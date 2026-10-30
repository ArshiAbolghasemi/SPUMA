"""Prompt composition (config/prompts fragments) and the nine dataset x architecture settings files: each agent's
system prompt, who may complete, and that no private grading field reaches any agent under any architecture."""

import dataclasses

import pytest

from spuma.agents.config import AgentConfig, compose_prompt
from spuma.agents.workflow import build_workflow
from spuma.config import PROJECT_ROOT, SETTINGS_FILE_ENV, get_settings
from spuma.dataset.musique import MuSiQueDataset
from spuma.dataset.tests.test_qa_datasets import musique_rows

ARCHITECTURES = ("feedbackloop", "star", "sequential")
DATASETS = ("mmlu", "musique", "morehopqa")
# the agents whose role prompt asks them to append TASK_DONE (everyone else is told not to)
COMPLETERS = {"feedbackloop": {"reviewer"}, "star": {"star"}, "sequential": {"finalizer"}}


def _activate(monkeypatch, dataset, architecture):
    path = PROJECT_ROOT / "config" / architecture / f"{dataset}_ma_qwen2_5_7b_{architecture}.yaml"
    monkeypatch.setenv(SETTINGS_FILE_ENV, str(path))
    get_settings.cache_clear()
    return get_settings()


def test_fragments_are_joined_in_order_and_checked():
    text = compose_prompt(["roles/general", "datasets/mmlu", "roles/sequential_solver", "schemas/mmlu"])
    parts = [compose_prompt([p]) for p in ("roles/general", "datasets/mmlu", "roles/sequential_solver", "schemas/mmlu")]
    assert text == "\n\n".join(parts) and text.index("GENERAL RULES") < text.index("You are the Solver")
    with pytest.raises(ValueError, match="not found"):
        compose_prompt(["roles/nope"])
    with pytest.raises(ValueError, match="not both"):
        AgentConfig(id="a", role="r", system_prompt="inline", prompt=["roles/general"])
    assert AgentConfig(id="a", role="r", prompt=["roles/general"]).system_prompt == compose_prompt(["roles/general"])


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("dataset", DATASETS)
def test_settings_compose_dataset_and_role_prompts(monkeypatch, dataset, architecture):
    settings = _activate(monkeypatch, dataset, architecture)
    dataset_text, schema = compose_prompt([f"datasets/{dataset}"]), compose_prompt([f"schemas/{dataset}"])
    for agent in settings.agents.enabled:
        prompt = agent.system_prompt
        assert prompt.startswith("GENERAL RULES") and dataset_text in prompt
        # a feedback reviewer critiques; it does not give the final answer, so it has no answer format
        assert prompt.endswith(schema) == (agent.id != "reviewer"), agent.id
        appends = "TASK_DONE" in prompt.replace("Do not output TASK_DONE", "")
        assert appends == (agent.id in COMPLETERS[architecture]), agent.id
    assert settings.chmm.frozen_from == [f"math_ma_qwen2_5_7b_{architecture}"]
    assert settings.experiment.name == f"ma_qwen_{dataset}_{architecture}"


SCRIPTS = {
    "sequential": ["ANSWER: Zed", "VERDICT: CORRECT\nVERIFIED_ANSWER: Zed", "ANSWER: Zed\nTASK_DONE"],
    "star": [
        "Analyst, solve it.",
        "ANSWER: Zed",
        "Verifier, check it.",
        "VERIFIED_ANSWER: Zed",
        "ANSWER: Zed\nTASK_DONE",
    ],
    "feedbackloop": ["ANSWER: Zed", "VERDICT: CORRECT\nSUGGESTED_ANSWER: Zed\nTASK_DONE"],
}


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_no_private_field_reaches_any_agent(monkeypatch, architecture, scripted_backend_cls):
    settings = _activate(monkeypatch, "musique", architecture)
    config = dataclasses.replace(settings.dataset, train_samples=0, validation_samples=0, test_samples=1)
    (problem,) = MuSiQueDataset(lambda _: musique_rows("dev", 1), config).load()["test"]
    backend = scripted_backend_cls(list(SCRIPTS[architecture]))
    traj = build_workflow(lambda _id: backend).run(problem.problem_id, problem.problem)
    assert traj.termination_reason == "task_done" and len(backend.calls) == len(SCRIPTS[architecture])
    prompts = "\n".join(backend.prompts)
    assert "[0] Club" in prompts
    for secret in ("Zed Zorin", "Z. Zorin", "SECRET_DECOMPOSITION", "is_supporting", "supporting_paragraphs"):
        assert secret not in prompts
