"""End-to-end: MATH-like problems -> Solver/Reviewer with tools -> step UQ -> RMS -> 8 features + label
-> SPN train/infer -> AUROC/AUARC, with a deterministic fake model and the real math tools."""

import json
import math
import re

import pytest

from spuma.conftest import ALL_CAPABILITIES
from spuma.dataset.base import Problem
from spuma.experiments.runner import ExperimentRunner
from spuma.models.backend import BackendInfo, GenerationResult, TokenScores, scorer_id
from spuma.models.chat import parse_qwen_output, to_chat_dicts
from spuma.tools.registry import default_providers
from spuma.tools.worker import IsolatedRunner


def _is_wrong(a: int, b: int) -> bool:
    return (7 * a + b) % 3 == 0


class FakeMathModel:
    """Solves 'What is A+B?' via sympy_simplify when offered tools; deliberately wrong on some problems,
    with lower confidence and more diverse samples there, so uncertainty carries signal.

    Plays the `kind` engine (transformers or vllm) of `model`; scores are a deterministic function of
    the prompt, so generation-time and teacher-forced scores agree like they do for real backends."""

    capabilities = ALL_CAPABILITIES

    def __init__(self, model, kind):
        self.name, self.calls, self.score_calls, self.closed = kind, 0, 0, False
        self.scorer_id = scorer_id(model, kind)
        self.info = BackendInfo(kind, model.name, None, model.name, None, "tmpl", 128, False, "bf16", None)

    def render(self, messages, tools):
        dicts = to_chat_dicts(messages)
        head = f"TOOLS={','.join(t['function']['name'] for t in tools or [])}\n"
        return [ord(c) for c in head + "\n".join(f"<{d['role']}>{d['content']}" for d in dicts)]

    @staticmethod
    def _wrong(prompt):
        a, b = map(int, re.search(r"What is (\d+)\+(\d+)\?", prompt).groups())
        return a, b, _is_wrong(a, b)

    def _answer(self, prompt):
        a, b, wrong = self._wrong(prompt)
        if "<tool>" not in prompt and "sympy_simplify" in prompt.split("\n")[0] and "Proposed solution" not in prompt:
            call = f'<tool_call>{{"name": "sympy_simplify", "arguments": {{"expression": "{a}+{b}"}}}}</tool_call>'
            return f"Use a tool. {call}", wrong
        return f"The sum is \\boxed{{{a + b + (1 if wrong else 0)}}}", wrong

    def _stats(self, prompt, n_tokens, full):
        _, _, wrong = self._wrong(prompt)
        lp, ent = (-1.2, 2.0) if wrong else (-0.1, 0.4)
        jitter = (len(prompt) % 100) / 1000
        return [lp - jitter] * n_tokens, ([ent + jitter] * n_tokens if full else None)

    def generate(self, request):
        self.calls += 1
        prompt = "".join(map(chr, request.prompt_ids))
        text, wrong = self._answer(prompt)
        texts = (
            [text.replace("\\boxed{", "\\boxed{1" if (wrong and i % 2) else "\\boxed{") for i in range(request.n)]
            if request.n > 1
            else [text]
        )
        out = []
        for t in texts:
            lps, ents = (None, None)
            if request.logprobs != "none":
                lps, ents = self._stats(prompt, len(t), request.logprobs == "full")
            out.append(
                GenerationResult(
                    [ord(c) for c in t],
                    t,
                    "stop",
                    len(request.prompt_ids),
                    self.name,
                    self.info.model,
                    self.info.tokenizer,
                    request.seed,
                    request.params.model_dump(),
                    request.logprobs,
                    lps,
                    ents,
                    self.scorer_id if lps else None,
                )
            )
        return out

    def generate_batch(self, requests):
        return [self.generate(r) for r in requests]

    def score(self, prompt_ids, token_ids, full_vocabulary):
        self.score_calls += 1
        lps, ents = self._stats("".join(map(chr, prompt_ids)), len(token_ids), full_vocabulary)
        return TokenScores(lps, ents, self.scorer_id)

    def parse(self, text):
        return parse_qwen_output(text)

    def count_tokens(self, text):
        return len(text)

    def truncate(self, text, n):
        return text[:n]

    def close(self):
        self.closed = True


class FakeDataset:
    def load(self):
        out = {}
        k = 0
        for split, size in (("train", 60), ("validation", 10), ("test", 30)):
            probs = []
            for _ in range(size):
                a, b = k % 17 + 1, k // 3 + 2
                k += 1
                probs.append(Problem(f"math/{split}/x/{k}", f"What is {a}+{b}?", f"so \\boxed{{{a + b}}}", split))
            out[split] = probs
        return out


@pytest.fixture(scope="module")
def runner_proc():
    r = IsolatedRunner()
    yield r
    r.close()


def _config(config, tmp_path, backend="transformers", scoring="same"):
    raw = config.to_dict()
    raw["experiment"]["output_dir"] = str(tmp_path)
    raw["evaluation"]["bootstrap"]["iterations"] = 50
    raw["spn"]["min_instances_slice"] = 20
    raw["experiment"]["spn_protocol"] = "cross_variant"
    raw["model"]["backend"], raw["model"]["scoring_backend"] = backend, scoring
    return type(config).from_dict(raw)


class Engines:
    """Backend factory + release hook that records the engine lifecycle."""

    def __init__(self):
        self.live, self.events, self.all = {}, [], []

    def factory(self, model_cfg, kind):
        if kind not in self.live:
            self.live[kind] = FakeMathModel(model_cfg, kind)
            self.all.append(self.live[kind])
            self.events.append(("load", kind, sorted(self.live)))
        return self.live[kind]

    def release(self):
        for e in self.live.values():
            e.close()
        self.events.append(("release", sorted(self.live)))
        self.live.clear()


@pytest.fixture
def fake_capabilities(monkeypatch):
    """vLLM is not installable here: report full capabilities for both kinds at validation time."""
    cls = type("C", (), {"capabilities_for": staticmethod(lambda config: ALL_CAPABILITIES)})
    monkeypatch.setattr("spuma.models.validation.backend_class", lambda kind: cls)


def test_full_pipeline(config, set_settings, tmp_path, runner_proc, fake_capabilities):
    cfg = set_settings(_config(config, tmp_path))
    engines = Engines()
    runner = ExperimentRunner(FakeDataset(), engines.factory, default_providers(runner_proc), release=engines.release)
    runner.run()
    store = runner.store
    assert [e[:2] for e in engines.events] == [("load", "transformers")]  # one engine shared by both agents
    model = engines.all[0]
    assert model.score_calls == 0  # scored at generation time (scoring_backend: same)

    # one shared split file, three variants evaluated side by side
    summary = store.read_json("summary.json")
    assert set(summary) == {"no_tools", "solver_tools", "both_tools"}
    for v, s in summary.items():
        assert s["n"] == 30 and 0 < s["accuracy"] < 1
        assert s["models"]["spn"]["auroc"] > 0.9, v
        assert s["models"]["spn"]["auarc"] > s["accuracy"]
        assert set(s["models"]) == {"spn", "spn_posterior_entropy"}  # only the configured risk model

    d = runner.scored_dir("solver_tools")
    feats = store.read_json(f"{d}/features.json")
    assert len(feats) == 100 and list(feats[0]["features"]) == cfg.features.order
    assert feats[0]["tool_metadata"]["solver"]["tool_calls"] == 1
    assert feats[0]["tool_metadata"]["reviewer"]["tool_calls"] == 0

    # trajectories: the tool observation is context for step 1, and SE sampled from that exact context
    pid = feats[0]["problem_id"].replace("/", "__")
    rec = store.read_json(f"{d}/steps/{pid}.json")
    solver = rec["trajectory"]["agents"][0]
    assert solver["steps"][0]["tool_records"][0]["status"] == "success"
    assert solver["steps"][1]["input_messages"][-1]["role"] == "tool"
    assert set(solver["steps"][1]["metrics"]) == {"pe", "se", "nll", "length"}
    assert len(solver["steps"][1]["uq_details"]["se_samples"]["texts"]) == 10
    assert 0 <= solver["steps"][1]["uq_details"]["se"]["semantic_negentropy"] <= 1
    assert solver["steps"][1]["scored_by"] == model.scorer_id
    assert solver["steps"][1]["metrics"]["length"] == len(solver["steps"][1]["token_ids"])
    assert rec["grade"]["label"] in (0, 1)

    no_tools = store.read_json(f"{runner.scored_dir('no_tools')}/features.json")
    assert all(r["tool_metadata"]["solver"]["tool_calls"] == 0 for r in no_tools)

    ev = store.read_json(f"{d}/evaluation.json")
    preds = ev["splits"]["test"]["predictions"]["spn"]
    assert all(0 <= p["p_incorrect"] <= 1 and 0 <= p["posterior_entropy"] <= math.log(2) + 1e-9 for p in preds)
    assert set(ev["cross_variant_test"]["spn"]) == {"no_tools", "both_tools"}
    assert "ci_low" in ev["splits"]["test"]["models"]["spn"]["auroc"]
    status = store.read_json(f"{d}/models/spn_status.json")
    assert status["status"] == "trained" and status["train_rows"] == 60

    meta = json.loads(next(store.path("runs").iterdir()).read_text())
    assert (
        meta["semantic_entropy_policy"]["n"] == 10
        and meta["semantic_entropy_policy"]["tool_call_policy"] == "no_execution"
    )
    assert store.read_json("config.json")["model"]["name"] == "Qwen/Qwen3-8B"

    assert meta["backends"]["plan"]["both_tools"]["default"]["generation_backend"] == "transformers"

    # resume: nothing is regenerated or rescored
    calls = model.calls
    ExperimentRunner(FakeDataset(), engines.factory, default_providers(runner_proc), release=engines.release).run()
    assert model.calls == calls and model.score_calls == 0


@pytest.mark.parametrize(("backend", "scoring"), [("vllm", "same"), ("vllm", "transformers")])
def test_backend_modes(config, set_settings, tmp_path, runner_proc, fake_capabilities, backend, scoring):
    """vLLM-only and hybrid (vLLM generation, Transformers scoring) produce the same downstream pipeline."""
    set_settings(_config(config, tmp_path, backend, scoring))
    engines = Engines()
    runner = ExperimentRunner(FakeDataset(), engines.factory, default_providers(runner_proc), release=engines.release)
    runner.run(variants=["both_tools"])
    gen = next(e for e in engines.all if e.name == "vllm")
    summary = runner.store.read_json("summary.json")["both_tools"]
    assert summary["models"]["spn"]["auroc"] > 0.9 and 0 < summary["accuracy"] < 1
    feats = runner.store.read_json(f"{runner.scored_dir('both_tools')}/features.json")
    assert len(feats) == 100 and all(len(r["features"]) == 8 for r in feats)
    rec = runner.store.read_json(
        f"{runner.scored_dir('both_tools')}/steps/{feats[0]['problem_id'].replace('/', '__')}.json"
    )
    step = rec["trajectory"]["agents"][0]["steps"][0]
    if scoring == "same":
        assert [e[:2] for e in engines.events] == [("load", "vllm")]
        assert gen.score_calls == 0 and step["scored_by"] == gen.scorer_id
    else:
        tf = next(e for e in engines.all if e.name == "transformers")
        # staged: vLLM loads, is released, and only then Transformers loads, alone on the "GPU"
        assert engines.events == [
            ("load", "vllm", ["vllm"]),
            ("release", ["vllm"]),
            ("load", "transformers", ["transformers"]),
        ]
        assert gen.closed and tf.score_calls > 0 and step["scored_by"] == tf.scorer_id
        raw = runner.store.read_json(
            f"{runner.variant_dir('both_tools')}/trajectories/{feats[0]['problem_id'].replace('/', '__')}.json"
        )
        assert raw["trajectory"]["agents"][0]["steps"][0]["token_logprobs"] is None  # vLLM did not score
    assert (
        len(step["uq_details"]["se_samples"]["texts"]) == 10 and step["uq_details"]["se_samples"]["backend"] == "vllm"
    )
