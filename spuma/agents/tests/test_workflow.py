import dataclasses
import json

import pytest

from spuma.agents.workflow import FeedbackWorkflow, MultiAgentWorkflow, build_workflow
from spuma.config import PROJECT_ROOT, SETTINGS_FILE_ENV, Settings, get_settings

A1, A2, A3 = "2x = 4, so \\boxed{2}", "Subtract 3: 2x = 4, so \\boxed{2}", "Indeed \\boxed{2}"
B1, B2 = "Why is 2x = 4?", "Can you justify the subtraction?"
DONE = "Clear now. TASK_DONE"


def _shared(backend):
    """One engine for both agents (as in the runner: agents on the same model share it)."""
    return lambda _id: backend


def _with_agents(config, set_settings, **fields):
    return set_settings(dataclasses.replace(config, agents=dataclasses.replace(config.agents, **fields)))


class RecordingUQ:
    """Stand-in for StepUncertainty: records what it is asked to measure, returns fixed scores."""

    def __init__(self, fail_at: int | None = None, value: float = 0.5):
        self.calls, self.fail_at, self.value = [], fail_at, value

    def measure(self, step, problem, marker=None):
        self.calls.append((step.agent_id, step.round, marker, step.text))
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise RuntimeError("scorer failed")
        metrics = {"mcsp": self.value, "se": self.value, "sd": self.value, "length": float(len(step.token_ids))}
        return {"metrics": metrics, "details": {"view": {"scored_text": step.text}}}


def test_feedback_loop_alternates_and_ends_at_task_done(config, scripted_backend_cls):
    backend = scripted_backend_cls([A1, B1, A2, DONE])
    workflow = build_workflow(lambda _id: backend)
    assert isinstance(workflow, FeedbackWorkflow)
    traj = workflow.run("p1", "Solve 2x + 3 = 7.")
    assert [(m["agent_id"], m["round"]) for m in traj.messages] == [
        ("mathematician", 1),
        ("enthusiast", 1),
        ("mathematician", 2),
        ("enthusiast", 2),
    ]
    assert [m["content"] for m in traj.messages] == [A1, B1, A2, DONE]
    assert traj.rounds == 2 and traj.termination_reason == "task_done" and traj.final_answer_text == A2
    mathematician, enthusiast = traj.agents
    assert [s.round for s in mathematician.steps] == [1, 2] and [s.step_id for s in enthusiast.steps] == [0, 1]
    assert enthusiast.steps[-1].text == DONE


def test_max_rounds_ends_the_loop(config, set_settings, scripted_backend_cls):
    _with_agents(config, set_settings, max_rounds=3)
    backend = scripted_backend_cls([A1, B1, A2, B2, A3, B1])
    traj = build_workflow(lambda _id: backend).run("p1", "q")
    assert traj.rounds == 3 and traj.termination_reason == "max_rounds" and traj.final_answer_text == A3
    assert len(backend.calls) == 6 and not backend.outputs


def test_only_the_enthusiast_can_complete(config, set_settings, scripted_backend_cls):
    _with_agents(config, set_settings, max_rounds=3)
    backend = scripted_backend_cls(["TASK_DONE, \\boxed{2}", B1, A2, DONE])
    traj = build_workflow(lambda _id: backend).run("p1", "q")
    assert traj.rounds == 2 and traj.termination_reason == "task_done"


def test_each_agent_sees_the_problem_and_the_conversation_from_its_side(config, scripted_backend_cls):
    traj = build_workflow(_shared(scripted_backend_cls([A1, B1, A2, DONE]))).run("p1", "Solve 2x + 3 = 7.")
    mathematician, enthusiast = traj.agents
    second = mathematician.steps[1].input_messages
    assert "Master Mathematician" in second[0]["content"] and "Solve 2x + 3 = 7." in second[1]["content"]
    assert second[2:] == [{"role": "assistant", "content": A1}, {"role": "user", "content": B1}]
    first_feedback = enthusiast.steps[0].input_messages
    assert "Curious Math Enthusiast" in first_feedback[0]["content"]
    assert "Solve 2x + 3 = 7." in first_feedback[1]["content"] and A1 in first_feedback[1]["content"]
    later = enthusiast.steps[1].input_messages
    assert [m["role"] for m in later] == ["system", "user", "assistant", "user"]
    assert later[1] == first_feedback[1] and later[2] == {"role": "assistant", "content": B1}
    assert later[3]["content"] == (
        f"Original mathematical problem:\nSolve 2x + 3 = 7.\n\nLatest response from the Master Mathematician:\n{A2}"
        "\n\nReview the response and provide your next question or feedback. If the solution is complete, end with "
        "TASK_DONE."
    )
    assert second[1]["content"] == "Solve the following mathematical problem:\n\nSolve 2x + 3 = 7."


def test_uq_measures_every_response_after_it_is_generated_and_never_routes(config, scripted_backend_cls):
    seen = []
    uq = RecordingUQ(value=0.0)
    traj = build_workflow(_shared(scripted_backend_cls([A1, B1, A2, DONE])), uq, seen.append).run("p1", "q")
    assert uq.calls == [
        ("mathematician", 1, None, A1),
        ("enthusiast", 1, "TASK_DONE", B1),
        ("mathematician", 2, None, A2),
        ("enthusiast", 2, "TASK_DONE", DONE),
    ]
    assert [s.metrics["mcsp"] for a in traj.agents for s in a.steps] == [0.0] * 4
    assert [(s.agent_id, s.round) for s in seen] == [(c[0], c[1]) for c in uq.calls] and seen[0].metrics
    assert all(set(m) == {"agent_id", "round", "content"} for m in traj.messages)
    unmeasured = build_workflow(_shared(scripted_backend_cls([A1, B1, A2, DONE]))).run("p1", "q")
    assert unmeasured.rounds == traj.rounds and unmeasured.messages == traj.messages


def test_reruns_are_reproducible_and_seeds_differ_per_agent_and_round(config, scripted_backend_cls):
    def run():
        traj = build_workflow(_shared(scripted_backend_cls([A1, B1, A2, DONE]))).run("p1", "q")
        return [s.seed for a in traj.agents for s in a.steps], len(traj.messages)

    (seeds, n), (again, m) = run(), run()
    assert seeds == again and len(set(seeds)) == 4 and n == m == 4


def test_a_failing_step_ends_the_run_as_error_and_keeps_earlier_steps(config, scripted_backend_cls):
    workflow = build_workflow(_shared(scripted_backend_cls([A1, B1, A2, DONE])), RecordingUQ(fail_at=3))
    traj = workflow.run("p1", "q")
    assert traj.termination_reason == "error" and "scorer failed" in traj.error
    assert [len(a.steps) for a in traj.agents] == [2, 1] and traj.agents[0].steps[0].metrics


def test_step_records_reconstruct_model_call(config, scripted_backend_cls):
    backend = scripted_backend_cls(["\\boxed{1}", "TASK_DONE"])
    traj = build_workflow(lambda _id: backend).run("p2", "1?")
    step = traj.agents[0].steps[0]
    request = backend.calls[0]
    assert step.prompt_ids == request.prompt_ids and step.seed == request.seed
    assert step.generation_params["temperature"] == 0.7 and request.logprobs == "token"
    assert step.token_logprobs and len(step.token_logprobs) == len(step.token_ids) and step.scored_by
    assert json.loads(traj.model_dump_json())["termination_reason"] == "task_done"


def test_graph_never_sees_reference(config, scripted_backend_cls):
    backend = scripted_backend_cls(["\\boxed{5}", "TASK_DONE"])
    build_workflow(lambda _id: backend).run("p7", "What is 2+3?")
    assert not any("reference" in p.lower() or "ground truth" in p.lower() for p in backend.prompts)


def test_sequential_workflow_one_measured_response_per_agent(monkeypatch, scripted_backend_cls):
    _sequential_settings(monkeypatch)
    uq = RecordingUQ()
    workflow = build_workflow(_shared(scripted_backend_cls([A1, B1, A3 + "\nTASK_DONE"])), uq)
    assert isinstance(workflow, MultiAgentWorkflow)
    traj = workflow.run("p1", "Solve 2x + 3 = 7.")
    assert [len(a.steps) for a in traj.agents] == [1, 1, 1] and traj.final_answer_text.startswith(A3)
    assert traj.termination_reason == "task_done" and [c[2] for c in uq.calls] == ["TASK_DONE"] * 3
    assert A1 in traj.agents[1].steps[0].input_messages[1]["content"]


def test_sequential_workflow_needs_three_agents(raw_config):
    raw_config["agents"]["workflow"] = "sequential"
    with pytest.raises(ValueError, match="exactly 3"):
        Settings.from_dict(raw_config)


def test_feedback_workflow_needs_two_agents(raw_config):
    raw_config["agents"]["items"][1]["enabled"] = False
    with pytest.raises(ValueError, match="exactly 2"):
        Settings.from_dict(raw_config)


def _sequential_settings(monkeypatch):
    path = PROJECT_ROOT / "config" / "sequential" / "math_ma_qwen2_5_7b_sequential.yaml"
    monkeypatch.setenv(SETTINGS_FILE_ENV, str(path))
    get_settings.cache_clear()


def test_sequential_experiment_runs_analyst_verifier_finalizer_once_each(monkeypatch, scripted_backend_cls):
    _sequential_settings(monkeypatch)
    final = A3 + "\nTASK_DONE"
    backend = scripted_backend_cls([A1, "VERDICT: CORRECT\nVERIFIED_ANSWER: \\boxed{2}", final])
    traj = build_workflow(lambda _id: backend).run("p1", "Solve 2x + 3 = 7.")
    assert [m["agent_id"] for m in traj.messages] == ["analyst", "verifier", "finalizer"]
    assert len(backend.calls) == 3 and traj.final_answer_text == final and traj.termination_reason == "task_done"
    assert traj.metadata == {"workflow": "sequential", "llm_calls": 3, "early_completion_token": []}
    analyst, verifier, finalizer = (a.steps[0].input_messages for a in traj.agents)
    assert "You are the Analyst" in analyst[0]["content"] and len(analyst) == len(verifier) == len(finalizer) == 2
    assert A1 in verifier[1]["content"] and "Solve 2x + 3 = 7." in verifier[1]["content"]
    assert A1 in finalizer[1]["content"] and "VERDICT: CORRECT" in finalizer[1]["content"]
    assert "TASK_DONE" in finalizer[0]["content"] and "TASK_DONE" not in analyst[0]["content"].replace(
        "Do not output TASK_DONE", ""
    )


def test_sequential_without_completion_token_still_ends_after_the_finalizer(monkeypatch, scripted_backend_cls):
    _sequential_settings(monkeypatch)
    backend = scripted_backend_cls([A1 + " TASK_DONE", B1, A3])
    traj = build_workflow(lambda _id: backend).run("p1", "q")
    assert len(backend.calls) == 3 and traj.termination_reason == "no_completion_token"
    assert traj.metadata["early_completion_token"] == ["analyst"]


def _qa_feedback_settings(monkeypatch):
    path = PROJECT_ROOT / "config" / "feedbackloop" / "mmlu_ma_qwen2_5_7b_feedbackloop.yaml"
    monkeypatch.setenv(SETTINGS_FILE_ENV, str(path))
    get_settings.cache_clear()


def test_reviewer_cannot_end_the_loop_while_reporting_an_error(monkeypatch, scripted_backend_cls):
    _qa_feedback_settings(monkeypatch)
    wrong = "VERDICT: INCORRECT\nFEEDBACK: B contradicts the premise.\nSUGGESTED_ANSWER: C\nTASK_DONE"
    right = "VERDICT: CORRECT\nFEEDBACK: Justified.\nSUGGESTED_ANSWER: C\nTASK_DONE"
    backend = scripted_backend_cls(["ANSWER: B", wrong, "ANSWER: C", right])
    traj = build_workflow(lambda _id: backend).run("p1", "Question:\nq?\n\nChoices:\nA. a\nB. b\nC. c\nD. d")
    assert traj.rounds == 2 and traj.termination_reason == "task_done" and traj.final_answer_text == "ANSWER: C"
    assert traj.metadata == {"ignored_completion_token": 1}
    revised = traj.agents[0].steps[1].input_messages
    assert revised[-1] == {"role": "user", "content": wrong}  # the solver sees the critique before revising


def test_qa_reviewer_prompt_reviews_rather_than_answers(monkeypatch):
    _qa_feedback_settings(monkeypatch)
    solver, reviewer = get_settings().agents.enabled
    assert "FINAL ANSWER FORMAT" in solver.system_prompt and "FINAL ANSWER FORMAT" not in reviewer.system_prompt
    assert "only\ntogether with VERDICT: CORRECT" in reviewer.system_prompt
