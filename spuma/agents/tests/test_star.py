import pytest

from spuma.agents.tests.test_workflow import RecordingUQ
from spuma.agents.workflow import FeedbackWorkflow, StarWorkflow, build_workflow
from spuma.config import PROJECT_ROOT, SETTINGS_FILE_ENV, Settings, get_settings

STAR_FILE = PROJECT_ROOT / "config" / "math_ma_qwen2_5_7b_star.yaml"
PROBLEM = "Solve 2x + 3 = 7."
REFERENCE = "so \\boxed{2} (reference)"
DELEGATE = "Analyst, please solve 2x + 3 = 7 completely."
SOLUTION = "2x = 4, so \\boxed{2}"
REQUEST = "The Analyst subtracted 3 and halved. Verifier, check it."
VERDICT = "VERDICT: CORRECT\nEXPLANATION: 2*2+3 = 7.\nVERIFIED_ANSWER: \\boxed{2}"
FINAL = "Both agree: x = 2. \\boxed{2}\nTASK_DONE"
SCRIPT = [DELEGATE, SOLUTION, REQUEST, VERDICT, FINAL]
NODES = ["star_delegate", "analyst", "star_request_verification", "verifier", "star_finalize"]


@pytest.fixture
def star(monkeypatch):
    monkeypatch.setenv(SETTINGS_FILE_ENV, str(STAR_FILE))
    get_settings.cache_clear()
    return get_settings()


def _run(backend, uq=None, problem=PROBLEM):
    return build_workflow(lambda _id: backend, uq).run("p1", problem)


def test_fixed_order_exactly_five_calls_and_termination(star, scripted_backend_cls):
    backend = scripted_backend_cls(list(SCRIPT))
    workflow = build_workflow(lambda _id: backend)
    assert isinstance(workflow, StarWorkflow)
    traj = workflow.run("p1", PROBLEM)
    assert [m["node"] for m in traj.messages] == NODES
    assert [m["agent_id"] for m in traj.messages] == ["star", "analyst", "star", "verifier", "star"]
    assert [m["phase"] for m in traj.messages] == ["delegation", None, "verification_request", None, "final_synthesis"]
    assert [m["content"] for m in traj.messages] == SCRIPT
    assert len(backend.calls) == 5 and not backend.outputs
    assert traj.termination_reason == "task_done" and traj.rounds == 1
    assert traj.final_answer_text == FINAL
    meta = traj.metadata
    assert meta["llm_calls"] == 5 and [u["node"] for u in meta["usage"]] == NODES
    assert meta["analyst_solution"] == SOLUTION and meta["verifier_assessment"] == VERDICT
    assert meta["phase"] == "final_synthesis" and meta["completed"] and meta["early_completion_token"] == []
    assert all(u["prompt_tokens"] > 0 and u["completion_tokens"] > 0 for u in meta["usage"])
    star_trace, analyst, verifier = traj.agents
    assert [s.step_id for s in star_trace.steps] == [0, 1, 2] and [s.round for s in star_trace.steps] == [1, 3, 5]
    assert analyst.steps[0].round == 2 and verifier.steps[0].round == 4


def test_always_ends_after_the_final_synthesis(star, scripted_backend_cls):
    # TASK_DONE too early never shortens the pipeline; a final synthesis without it still ends the run
    early = scripted_backend_cls([DELEGATE + " TASK_DONE", SOLUTION, REQUEST, VERDICT, "\\boxed{2}"])
    traj = _run(early)
    assert [m["node"] for m in traj.messages] == NODES and len(early.calls) == 5
    assert traj.termination_reason == "no_completion_token" and traj.metadata["early_completion_token"] == [
        "star_delegate"
    ]


def test_each_agent_sees_only_its_context(star, scripted_backend_cls):
    traj = _run(scripted_backend_cls(list(SCRIPT)))
    star_trace, analyst, verifier = traj.agents
    delegate, request, final = (s.input_messages for s in star_trace.steps)
    assert [m["role"] for m in delegate] == ["system", "user"] and PROBLEM in delegate[1]["content"]
    assert "PHASE 1" in delegate[1]["content"] and "You are Star" in delegate[0]["content"]
    # the coordinator accumulates its conversation, one phase prompt per call
    assert [m["role"] for m in request] == ["system", "user", "assistant", "user"]
    assert (
        request[2]["content"] == DELEGATE and SOLUTION in request[3]["content"] and "PHASE 2" in request[3]["content"]
    )
    assert [m["role"] for m in final] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert final[4]["content"] == REQUEST and VERDICT in final[5]["content"] and "PHASE 3" in final[5]["content"]
    a = analyst.steps[0].input_messages
    assert "You are the Analyst" in a[0]["content"] and PROBLEM in a[1]["content"] and DELEGATE in a[1]["content"]
    v = verifier.steps[0].input_messages
    assert "You are the Verifier" in v[0]["content"]
    assert all(x in v[1]["content"] for x in (PROBLEM, SOLUTION, REQUEST)) and DELEGATE not in v[1]["content"]


def test_no_agent_sees_the_reference_answer(star, scripted_backend_cls):
    backend = scripted_backend_cls(list(SCRIPT))
    _run(backend)
    assert not any(REFERENCE in p for p in backend.prompts)


def test_completion_marker_is_stripped_before_scoring_on_every_node(star, scripted_backend_cls):
    uq = RecordingUQ()
    traj = _run(scripted_backend_cls(list(SCRIPT)), uq)
    assert [(c[0], c[1], c[2]) for c in uq.calls] == [
        ("star", 1, "TASK_DONE"),
        ("analyst", 2, "TASK_DONE"),
        ("star", 3, "TASK_DONE"),
        ("verifier", 4, "TASK_DONE"),
        ("star", 5, "TASK_DONE"),
    ]
    assert all(s.metrics for a in traj.agents for s in a.steps)


def test_reruns_are_reproducible_with_a_distinct_seed_per_call(star, scripted_backend_cls):
    def run():
        traj = _run(scripted_backend_cls(list(SCRIPT)))
        return [m["content"] for m in traj.messages], [s.seed for a in traj.agents for s in a.steps]

    (texts, seeds), (again_texts, again_seeds) = run(), run()
    assert texts == again_texts and seeds == again_seeds and len(set(seeds)) == 5


def test_a_failing_call_ends_the_run_as_error(star, scripted_backend_cls):
    traj = _run(scripted_backend_cls(list(SCRIPT)), RecordingUQ(fail_at=3))
    assert traj.termination_reason == "error" and "scorer failed" in traj.error
    assert [len(a.steps) for a in traj.agents] == [2, 1, 0]


def test_star_settings_need_three_agents_and_the_phase_templates(star):
    raw = star.to_dict()
    raw["agents"]["items"][2]["enabled"] = False
    with pytest.raises(ValueError, match="exactly 3"):
        Settings.from_dict(raw)
    raw = star.to_dict()
    del raw["agents"]["items"][0]["templates"]["final_synthesis"]
    with pytest.raises(ValueError, match="final_synthesis"):
        Settings.from_dict(raw)


def test_feedback_loop_is_unchanged(config, scripted_backend_cls):
    assert config.agents.workflow == "feedback" and config.generation_fingerprint() == "73c7092a6566"
    assert isinstance(build_workflow(lambda _id: scripted_backend_cls([])), FeedbackWorkflow)
