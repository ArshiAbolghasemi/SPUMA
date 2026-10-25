"""End-to-end: MATH-like problems -> Mathematician <-> Enthusiast feedback loop, UQ
after every response (MCSP, SE, SD, length) -> per-agent aggregation -> 8 features + label per run -> SPN
train/infer -> AUROC/AUARC, with a deterministic fake model."""

import json
import math
import re

import pytest

from spuma.common.config import to_plain
from spuma.dataset.base import Problem
from spuma.experiments.runner import ExperimentRunner
from spuma.models.backend import BackendCapabilityError, BackendInfo, GenerationResult, TokenScores, scorer_id
from spuma.models.chat import parse_qwen_output, to_chat_dicts
from spuma.models.tests.fixtures import ALL_CAPABILITIES

ROUNDS = 3
METRICS = {"mcsp", "se", "nll", "length"}


def _is_wrong(a: int, b: int) -> bool:
    return (7 * a + b) % 3 == 0


class FakeMathModel:
    """Plays both roles on 'What is A+B?'. The Mathematician answers, deliberately wrong on some problems, with
    lower confidence and more diverse samples there, so uncertainty carries signal; the Enthusiast asks once and
    then replies TASK_DONE.

    Scores are a deterministic function of the prompt, so generation-time and teacher-forced scores agree like
    they do for real backends."""

    capabilities = ALL_CAPABILITIES

    def __init__(self, model, kind):
        self.name, self.calls, self.score_calls, self.closed = kind, 0, 0, False
        self.fail_batches_for: set[str] = set()
        self.scorer_id = scorer_id(model, kind)
        self.info = BackendInfo(kind, model.name, None, model.name, None, "tmpl", 128, False, "bf16", None)

    def render(self, messages):
        dicts = to_chat_dicts(messages)
        return [ord(c) for c in "\n".join(f"<{d['role']}>{d['content']}" for d in dicts)]

    @staticmethod
    def _wrong(prompt):
        a, b = map(int, re.search(r"What is (\d+)\+(\d+)\?", prompt).groups())
        return a, b, _is_wrong(a, b)

    def _answer(self, prompt):
        a, b, wrong = self._wrong(prompt)
        if prompt.startswith("<system>You are the Curious Math Enthusiast"):
            return ("TASK_DONE" if "<assistant>" in prompt else "Why is that the sum?"), False
        return f"The sum is \\boxed{{{a + b + (1 if wrong else 0)}}}", wrong

    def _stats(self, prompt, n_tokens):
        _, _, wrong = self._wrong(prompt)
        lp = -1.2 if wrong else -0.1
        return [lp - (len(prompt) % 100) / 1000] * n_tokens

    def generate(self, request):
        self.calls += 1
        prompt = "".join(map(chr, request.prompt_ids))
        text, wrong = self._answer(prompt)
        texts = [
            text.replace("\\boxed{", "\\boxed{1" if (wrong and (request.seed + i) % 2) else "\\boxed{")
            for i in range(request.n)
        ]
        out = []
        for t in texts:
            lps = self._stats(prompt, len(t)) if request.logprobs != "none" else None
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
                    to_plain(request.params),
                    request.logprobs,
                    lps,
                    None,
                    self.scorer_id if lps else None,
                )
            )
        return out

    def generate_batch(self, requests):
        prompts = ["".join(map(chr, r.prompt_ids)) for r in requests]
        if any(p in prompt for p in self.fail_batches_for for prompt in prompts):
            raise RuntimeError("sampling failed")
        return [self.generate(r) for r in requests]

    def score(self, prompt_ids, token_ids, full_vocabulary):
        self.score_calls += 1
        return TokenScores(self._stats("".join(map(chr, prompt_ids)), len(token_ids)), None, self.scorer_id)

    def parse(self, text):
        return parse_qwen_output(text)

    def count_tokens(self, text):
        return len(text)

    def encode(self, text):
        return [ord(c) for c in text]

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


def _config(config, tmp_path, backend="transformers", scoring="same", concurrent=1):
    raw = config.to_dict()
    raw["experiment"]["output_dir"] = str(tmp_path)
    raw["experiment"]["concurrent_problems"] = concurrent
    raw["agents"]["max_rounds"] = ROUNDS
    raw["evaluation"]["bootstrap"]["iterations"] = 50
    raw["spn"]["min_instances_slice"] = 20
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


def _record(runner, problem_id):
    return runner.store.read_json(f"{runner.generation_dir()}/trajectories/{problem_id.replace('/', '__')}.json")


def test_full_pipeline(config, set_settings, tmp_path, fake_capabilities):
    cfg = set_settings(_config(config, tmp_path))
    engines = Engines()
    runner = ExperimentRunner(FakeDataset(), engines.factory, release=engines.release)
    runner.run()
    store = runner.store
    assert engines.events == [("load", "transformers", ["transformers"]), ("release", ["transformers"])]
    model = engines.all[0]
    assert model.score_calls == 0

    summary = store.read_json("summary.json")
    assert summary["n"] == 30 and 0 < summary["accuracy"] < 1
    assert summary["models"]["spn"]["auroc"] > 0.9
    assert summary["models"]["spn"]["auarc"] > summary["accuracy"]
    names = {"spn", "spn_class_conditional"}
    assert set(summary["models"]) == names | {f"{n}_posterior_entropy" for n in names}

    d = runner.scored_dir()
    feats = store.read_json(f"{d}/features.json")
    assert len(feats) == 100 and list(feats[0]["features"]) == cfg.features.order
    assert store.read_json(f"{d}/skipped.json") == []

    record = _record(runner, feats[0]["problem_id"])
    traj = record["trajectory"]
    assert record["grade"]["label"] in (0, 1)
    assert traj["rounds"] == 2 and traj["termination_reason"] == "task_done"
    assert [(m["agent_id"], m["round"]) for m in traj["messages"]] == [
        ("mathematician", 1),
        ("enthusiast", 1),
        ("mathematician", 2),
        ("enthusiast", 2),
    ]
    mathematician, enthusiast = traj["agents"]
    steps = mathematician["steps"] + enthusiast["steps"]
    assert all(set(s["metrics"]) == METRICS for s in steps)
    first = mathematician["steps"][0]
    assert "se_samples" not in first["uq_details"] and 0 <= first["metrics"]["se"]
    mc = first["uq_details"]["mc_samples"]
    assert len(mc["texts"]) == len(mc["token_logprobs"]) == 10 and len(set(mc["seeds"])) == 10
    assert 0 < first["metrics"]["mcsp"] <= 1 and first["metrics"]["nll"] > 0 and "sd" not in first["uq_details"]
    assert first["metrics"]["length"] == len(first["token_ids"]) and first["uq_details"]["nll"]["normalized_by_length"]
    done = enthusiast["steps"][-1]
    assert done["text"] == "TASK_DONE" and done["uq_details"]["view"]["marker_only_texts"] == 11
    progress_log = store.read_jsonl(
        f"{runner.generation_dir()}/trajectories/{feats[0]['problem_id'].replace('/', '__')}.steps.jsonl"
    )
    assert [(s["agent_id"], s["round"]) for s in progress_log] == [
        (m["agent_id"], m["round"]) for m in traj["messages"]
    ]

    lines = store.read_jsonl(f"{d}/steps.jsonl")
    assert len(lines) == 4 * 100
    line = lines[0]
    assert {"problem_id", "agent_id", "round", "input_messages", "text", "seed", "generation_params"} <= set(line)
    assert line["token_logprobs"] and set(line["metrics"]) == METRICS and line["uq_details"]["mc_samples"]["seeds"]
    rows = store.read_json(f"{d}/step_metrics.json")
    assert len(rows) == len(lines) and "round" in rows[0]

    ev = store.read_json(f"{d}/evaluation.json")
    preds = ev["splits"]["test"]["predictions"]["spn"]
    assert len(preds) == 30
    assert all(0 <= p["p_incorrect"] <= 1 and 0 <= p["posterior_entropy"] <= math.log(2) + 1e-9 for p in preds)
    assert "ci_low" in ev["splits"]["test"]["models"]["spn"]["auroc"]
    assert set(ev["splits"]["test"]["models"]["spn"]) >= {"auroc", "auprc", "auarc", "aurc", "brier", "ece", "nll"}
    status = store.read_json(f"{d}/models/spn_status.json")
    assert status["status"] == "trained" and status["train_rows"] == 60

    meta = json.loads(next(store.path("runs").iterdir()).read_text())
    assert (
        meta["monte_carlo_policy"]["num_generations"] == 10 and meta["monte_carlo_policy"]["uqlm_mode"] == "white_box"
    )
    assert meta["agents"]["max_rounds"] == ROUNDS and meta["packages"]["uqlm"] and meta["base_seed"] == cfg.project.seed
    assert store.read_json("config.json")["model"]["name"] == "Qwen/Qwen2.5-7B-Instruct"

    calls = model.calls
    ExperimentRunner(FakeDataset(), engines.factory, release=engines.release).run()
    assert engines.all[-1].calls == 0 and model.calls == calls


def test_vllm_generation_measures_with_the_same_engine(config, set_settings, tmp_path, fake_capabilities):
    set_settings(_config(config, tmp_path, "vllm", "same"))
    engines = Engines()
    runner = ExperimentRunner(FakeDataset(), engines.factory, release=engines.release)
    runner.run()
    assert engines.events == [("load", "vllm", ["vllm"]), ("release", ["vllm"])]
    feats = runner.store.read_json(f"{runner.scored_dir()}/features.json")
    assert len(feats) == 100 and all(len(r["features"]) == 8 for r in feats)
    step = _record(runner, feats[0]["problem_id"])["trajectory"]["agents"][0]["steps"][0]
    assert step["scored_by"] == engines.all[0].scorer_id and step["uq_details"]["mc_samples"]["backend"] == "vllm"


def test_separate_scoring_backend_is_rejected(config, set_settings, tmp_path, fake_capabilities):
    set_settings(_config(config, tmp_path, "vllm", "transformers"))
    engines = Engines()
    with pytest.raises(BackendCapabilityError, match="scoring_backend: same"):
        ExperimentRunner(FakeDataset(), engines.factory, release=engines.release).run()
    assert engines.events == []


def test_failed_problems_are_recorded_skipped_and_retried(config, set_settings, tmp_path, fake_capabilities):
    set_settings(_config(config, tmp_path))
    engines = Engines()
    runner = ExperimentRunner(FakeDataset(), engines.factory, release=engines.release)
    model = engines.factory(config.model, "transformers")
    model.fail_batches_for = {"What is 1+2?"}
    runner.run(phases=["generate", "score"])
    failed = _record(runner, "math/train/x/1")
    assert failed["trajectory"]["termination_reason"] == "error" and "sampling failed" in failed["trajectory"]["error"]
    assert failed["grade"] is None
    skipped = runner.store.read_json(f"{runner.scored_dir()}/skipped.json")
    assert [s["problem_id"] for s in skipped] == ["math/train/x/1"]
    assert len(runner.store.read_json(f"{runner.scored_dir()}/features.json")) == 99

    retry = Engines()
    ExperimentRunner(FakeDataset(), retry.factory, release=retry.release).run(phases=["generate", "score"])
    assert _record(runner, "math/train/x/1")["trajectory"]["termination_reason"] == "task_done"
    assert retry.all[0].calls > 0 and len(runner.store.read_json(f"{runner.scored_dir()}/features.json")) == 100


def test_concurrent_problems_give_identical_results(config, set_settings, tmp_path, fake_capabilities):
    """8 problems in flight (batched model calls, scorers on the main thread) produce exactly the features of the
    one-at-a-time run."""
    outputs = {}
    for concurrent in (1, 8):
        set_settings(_config(config, tmp_path / f"c{concurrent}", concurrent=concurrent))
        engines = Engines()
        runner = ExperimentRunner(FakeDataset(), engines.factory, release=engines.release)
        runner.run(phases=["generate", "score"])
        rows = runner.store.read_json(f"{runner.scored_dir()}/features.json")
        outputs[concurrent] = {r["problem_id"]: (r["features"], r["label"]) for r in rows}
    assert len(outputs[1]) == 100 and outputs[1] == outputs[8]


def test_star_pipeline_with_chmm(set_settings, tmp_path, fake_capabilities, monkeypatch):
    """Star: five measured calls per problem, 12 per-agent features, the risk models and all nine CHMM-study methods
    on the same trajectories; the final synthesis is the last step and the graded answer."""
    from spuma.chmm.experiment import STANDALONE, CHMMAggregation, chmm_method  # noqa: PLC0415
    from spuma.config import PROJECT_ROOT, SETTINGS_FILE_ENV, get_settings  # noqa: PLC0415

    monkeypatch.setenv(SETTINGS_FILE_ENV, str(PROJECT_ROOT / "config" / "star" / "math_ma_qwen2_5_7b_star.yaml"))
    get_settings.cache_clear()
    cfg = set_settings(_config(get_settings(), tmp_path))
    engines = Engines()
    runner = ExperimentRunner(FakeDataset(), engines.factory, release=engines.release)
    runner.run()
    store, d = runner.store, runner.scored_dir()
    feats = store.read_json(f"{d}/features.json")
    assert len(feats) == 100 and list(feats[0]["features"]) == cfg.features.order and len(cfg.features.order) == 12
    record = _record(runner, feats[0]["problem_id"])
    traj = record["trajectory"]
    assert [m["node"] for m in traj["messages"]] == [
        "star_delegate",
        "analyst",
        "star_request_verification",
        "verifier",
        "star_finalize",
    ]
    assert traj["final_answer_text"] == traj["messages"][-1]["content"] and record["grade"]["label"] in (0, 1)
    assert all(len(s["metrics"]) == 4 for a in traj["agents"] for s in a["steps"])
    references = {
        p["problem_id"]: p["reference_solution"] for ps in store.read_json("splits.json").values() for p in ps
    }
    for s in store.read_jsonl(f"{d}/steps.jsonl"):
        assert not any(references[s["problem_id"]] in m["content"] for m in s["input_messages"])

    summary = store.read_json("summary.json")
    assert {"spn", "spn_class_conditional"} <= set(summary["models"])
    methods = [*STANDALONE, "rms_spn", *(chmm_method(a) for a in cfg.chmm.alphas)]
    assert list(summary["chmm"]) == methods
    out = CHMMAggregation(runner).out
    steps = store.read_jsonl(f"{out}/steps.jsonl")
    first = [s for s in steps if s["problem_id"] == feats[0]["problem_id"]]
    assert [s["agent_id"] for s in first] == ["star", "analyst", "star", "verifier", "star"]
    assert [s["preceding"] for s in first] == [None, 0, 1, 2, 3]
    assert sum(s["weights"]["1"] for s in first) == pytest.approx(1)
