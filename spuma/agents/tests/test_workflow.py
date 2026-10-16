import dataclasses
import json

from spuma.agents.workflow import build_workflow


def _workflow(backend):
    """The workflow for the current get_settings() (tests install modified settings with `set_settings`)."""
    return build_workflow(lambda _id: backend)


def test_solver_then_reviewer_one_step_each(config, scripted_backend_cls):
    backend = scripted_backend_cls(["So x = 2. \\boxed{2}", "The solution is right. \\boxed{2}"])
    traj = _workflow(backend).run("p1", "Solve 2x + 3 = 7.")
    solver, reviewer = traj.agents
    assert [a.agent_id for a in traj.agents] == ["solver", "reviewer"]
    assert len(solver.steps) == len(reviewer.steps) == 1 and len(backend.calls) == 2
    assert solver.final_content == "So x = 2. \\boxed{2}"
    assert "So x = 2." in reviewer.steps[0].input_messages[1]["content"]
    assert traj.final_answer_text == "The solution is right. \\boxed{2}"


def test_reasoning_is_split_from_content(config, scripted_backend_cls):
    backend = scripted_backend_cls(["<think>2x = 4</think>x = 2. \\boxed{2}", "\\boxed{2}"])
    step = _workflow(backend).run("p3", "Solve 2x + 3 = 7.").agents[0].steps[0]
    assert step.reasoning == "2x = 4" and step.content == "x = 2. \\boxed{2}"
    assert step.text.startswith("<think>")


def test_step_records_reconstruct_model_call(config, set_settings, scripted_backend_cls):
    same = dataclasses.replace(config.model, backend="transformers", scoring_backend="same")
    set_settings(dataclasses.replace(config, model=same))
    backend = scripted_backend_cls(["\\boxed{1}", "\\boxed{1}"])
    traj = _workflow(backend).run("p2", "1?")
    step = traj.agents[0].steps[0]
    request = backend.calls[0]
    assert step.prompt_ids == request.prompt_ids
    assert step.seed == request.seed and step.generation_params["temperature"] == 0.7
    assert request.logprobs == "full" and step.scored_by == backend.scorer_id
    assert step.token_logprobs and len(step.token_logprobs) == len(step.token_entropies) == len(step.token_ids)
    assert step.generation_meta["finish_reason"] == "stop" and step.generation_meta["compat"]["model"]
    assert json.loads(traj.model_dump_json())["problem_id"] == "p2"


def test_hybrid_generation_requests_no_scores(config, set_settings, scripted_backend_cls):
    set_settings(
        dataclasses.replace(
            config, model=dataclasses.replace(config.model, **{"backend": "vllm", "scoring_backend": "transformers"})
        )
    )
    backend = scripted_backend_cls(["\\boxed{1}", "\\boxed{1}"], kind="vllm")
    step = _workflow(backend).run("p8", "1?").agents[0].steps[0]
    assert backend.calls[0].logprobs == "none" and step.token_logprobs is None and step.scored_by is None


def test_graph_never_sees_reference(config, scripted_backend_cls):
    backend = scripted_backend_cls(["\\boxed{5}", "\\boxed{5}"])
    _workflow(backend).run("p7", "What is 2+3?")
    assert not any("reference" in p.lower() or "ground truth" in p.lower() for p in backend.prompts)
