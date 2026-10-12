import json

import pytest

from spuma.agents.workflow import build_workflow
from spuma.tools.registry import ToolExecutor, build_tool_registry, default_providers
from spuma.tools.worker import IsolatedRunner

SOLVE_CALL = (
    '<tool_call>{"name": "sympy_solve", "arguments": {"equations": ["2x + 3 = 7"], "variables": ["x"]}}</tool_call>'
)
FORBIDDEN_CALL = '<tool_call>{"name": "sympy_integrate", "arguments": {"expression": "x", "variable": "x"}}</tool_call>'


@pytest.fixture(scope="module")
def runner():
    r = IsolatedRunner()
    yield r
    r.close()


def _workflow(config, backend, runner, max_steps=None):
    if max_steps:
        config = config.replace(
            orchestration=config.orchestration.model_copy(update={"max_steps_per_agent": max_steps})
        )
    registry = build_tool_registry(config.tools, default_providers(runner))
    return build_workflow(config, registry, lambda _id: backend, ToolExecutor(registry, config.tools.execution))


def test_solver_tool_cycle_then_reviewer(config, runner, scripted_backend_cls):
    backend = scripted_backend_cls(
        [
            f"I will solve it. {SOLVE_CALL}",
            "So x = 2. \\boxed{2}",
            f"Let me integrate. {FORBIDDEN_CALL}",
            "The solution is right. \\boxed{2}",
        ]
    )
    traj = _workflow(config, backend, runner).run("p1", "Solve 2x + 3 = 7.")
    solver, reviewer = traj.agents
    assert [a.agent_id for a in traj.agents] == ["solver", "reviewer"]
    assert len(solver.steps) == 2 and solver.stop_reason == "final_message"
    rec = solver.steps[0].tool_records[0]
    assert rec["status"] == "success" and rec["result"]["result"] == [{"x": "2"}]
    assert rec["problem_id"] == "p1" and rec["agent_id"] == "solver" and rec["step_id"] == 0
    # the observation enters the next step's input as a tool message, not as generated text
    assert solver.steps[1].input_messages[-1] == {"role": "tool", "content": rec["observation"]}
    assert rec["observation"] not in solver.steps[1].text
    # reviewer receives the problem and the solver's final message, and is refused a non-allowed tool
    assert "So x = 2." in reviewer.steps[0].input_messages[1]["content"]
    assert reviewer.steps[0].tool_records[0]["status"] == "permission_denied"
    assert traj.final_answer_text == "The solution is right. \\boxed{2}"
    # tools offered = effective set: reviewer sees math_verify, solver does not; python_execute disabled
    assert "math_verify" in backend.tools_seen[2] and "math_verify" not in backend.tools_seen[0]
    assert all("python_execute" not in t for t in backend.tools_seen)


def test_step_records_reconstruct_model_call(config, runner, scripted_backend_cls):
    backend = scripted_backend_cls(["\\boxed{1}", "\\boxed{1}"])
    traj = _workflow(config, backend, runner).run("p2", "1?")
    step = traj.agents[0].steps[0]
    request = backend.calls[0]
    assert step.prompt_ids == request.prompt_ids
    assert step.seed == request.seed and step.generation_params["temperature"] == 0.7
    # scoring_backend: same -> raw full-vocabulary scores captured while generating
    assert request.logprobs == "full" and step.scored_by == backend.scorer_id
    assert step.token_logprobs and len(step.token_logprobs) == len(step.token_entropies) == len(step.token_ids)
    assert step.generation_meta["finish_reason"] == "stop" and step.generation_meta["compat"]["model"]
    assert json.loads(traj.model_dump_json())["problem_id"] == "p2"


def test_hybrid_generation_requests_no_scores(config, runner, scripted_backend_cls):
    hybrid = config.replace(
        model=config.model.model_copy(update={"backend": "vllm", "scoring_backend": "transformers"})
    )
    backend = scripted_backend_cls(["\\boxed{1}", "\\boxed{1}"], kind="vllm")
    step = _workflow(hybrid, backend, runner).run("p8", "1?").agents[0].steps[0]
    assert backend.calls[0].logprobs == "none" and step.token_logprobs is None and step.scored_by is None


def test_max_steps_stops_tool_loop(config, runner, scripted_backend_cls):
    backend = scripted_backend_cls([SOLVE_CALL, SOLVE_CALL, "\\boxed{2}"])
    traj = _workflow(config, backend, runner, max_steps=2).run("p3", "Solve 2x + 3 = 7.")
    solver = traj.agents[0]
    assert len(solver.steps) == 2 and solver.stop_reason == "max_steps"
    assert solver.steps[1].tool_records == []  # the last step's calls are not executed


def test_tool_budget_enforced(config, runner, scripted_backend_cls):
    raw = config.to_dict()
    raw["agents"]["items"][0]["max_tool_calls"] = 1
    cfg = type(config).from_dict(raw)
    backend = scripted_backend_cls([SOLVE_CALL + SOLVE_CALL, "\\boxed{2}", "\\boxed{2}"])
    traj = _workflow(cfg, backend, runner).run("p4", "q")
    statuses = [r["status"] for r in traj.agents[0].steps[0].tool_records]
    assert statuses == ["success", "budget_exceeded"]


def test_malformed_tool_call_gets_error_observation(config, runner, scripted_backend_cls):
    backend = scripted_backend_cls(["<tool_call>{bad json</tool_call>", "\\boxed{2}", "\\boxed{2}"])
    traj = _workflow(config, backend, runner).run("p5", "q")
    assert traj.agents[0].steps[0].tool_records[0]["status"] == "invalid_call"


def test_no_tools_variant_offers_nothing(config, runner, scripted_backend_cls):
    backend = scripted_backend_cls([SOLVE_CALL, "\\boxed{2}", "\\boxed{2}"])
    traj = _workflow(config.with_variant("no_tools"), backend, runner).run("p6", "q")
    assert backend.tools_seen[0] == []
    assert traj.agents[0].steps[0].tool_records[0]["status"] == "permission_denied"


def test_graph_never_sees_reference(config, runner, scripted_backend_cls):
    backend = scripted_backend_cls(["\\boxed{5}", "\\boxed{5}"])
    _workflow(config, backend, runner).run("p7", "What is 2+3?")
    assert not any("reference" in p.lower() or "ground truth" in p.lower() for p in backend.prompts)
