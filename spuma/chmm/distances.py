"""Situational distances of every agent response (SAUP-inspired), in [0, 1], higher = more deviation.

For response t of a trajectory (chronological order):
    question drift   d_q(t) = distance(original question, response t)
    interaction gap  d_i(t) = distance(response t, the relevant preceding response)
The relevant preceding response is the most recent response by a *different* agent: in the feedback loop the
other agent's last turn; in a star, a worker's coordinator message and the coordinator's latest worker reply.
The first response, and any response with no earlier response by another agent, has no interaction gap: it is
missing (None), never 0 or 1, and the CHMM integrates that dimension out instead of reading it as agreement or
disagreement.

`qa` (default, `deepset/roberta-base-squad2`): an extractive QA model reads response t as the context of the
original question. A SQuAD 2.0 model scores every answer span against its "no answer" option (the [CLS]
span); over the sliding windows of the response,
    p_answerable(t) = max_windows sigmoid(s_best_span - s_null),  s = start_logit + end_logit
is the probability that the response contains an answer to the question (SQuAD 2.0's null-score
difference, squashed). QA confidence is not a semantic distance, so the conversion is explicit:
    d_q(t) = 1 - p_answerable(t)                  the response no longer answers the question
    d_i(t) = 1 - F1(span_t, span_prev)            SQuAD token F1 of the two responses' best answer spans:
                                                  how far they disagree on the answer to the same question
Both are proxies: d_q is high for a response that only asks questions (an enthusiast turn) as well as for one
that drifted, and d_i measures span overlap, not entailment.

`lexical` (offline, no model): d_q = 1 - |question words in response| / |question words|,
d_i = 1 - Jaccard(word sets). For tests and as a model-free reference.

A new estimator implements `DistanceEstimator` and adds one `ESTIMATORS` entry.
"""

import collections
import logging
import math
import re
import string
from collections.abc import Callable
from typing import Any, Protocol

from spuma.chmm.config import DistanceConfig

logger = logging.getLogger(__name__)


class DistanceEstimator(Protocol):
    def read(self, question: str, texts: list[str]) -> list[dict[str, Any]]:
        """One record per text; each has `drift` = d_q in [0, 1] and whatever `gap` needs."""
        ...

    def gap(self, current: dict[str, Any], previous: dict[str, Any]) -> float:
        """d_i in [0, 1] between two records of the same trajectory."""
        ...


def _normalize(text: str) -> list[str]:
    """SQuAD answer normalization: lowercase, no punctuation or articles, whitespace tokens."""
    text = "".join(c for c in text.lower() if c not in set(string.punctuation))
    return re.sub(r"\b(a|an|the)\b", " ", text).split()


def token_f1(a: str, b: str) -> float:
    ta, tb = _normalize(a), _normalize(b)
    if not ta or not tb:
        return float(ta == tb)
    common = sum((collections.Counter(ta) & collections.Counter(tb)).values())
    if not common:
        return 0.0
    precision, recall = common / len(ta), common / len(tb)
    return 2 * precision * recall / (precision + recall)


class LexicalDistance:
    def __init__(self, config: DistanceConfig) -> None:
        pass

    def read(self, question: str, texts: list[str]) -> list[dict[str, Any]]:
        q = set(_normalize(question))
        out = []
        for text in texts:
            words = set(_normalize(text))
            out.append({"drift": 1 - len(q & words) / len(q) if q else 0.0, "words": sorted(words)})
        return out

    def gap(self, current: dict[str, Any], previous: dict[str, Any]) -> float:
        a, b = set(current["words"]), set(previous["words"])
        return 1 - len(a & b) / len(a | b) if a | b else 0.0


class QADistance:
    """Extractive QA (SQuAD 2.0) over sliding windows of each response; see the module docstring."""

    def __init__(self, config: DistanceConfig) -> None:
        import torch  # noqa: PLC0415 - heavy, only when this estimator is used
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer  # noqa: PLC0415

        self.config = config
        self.torch = torch
        device = config.device or (
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
        self.model = AutoModelForQuestionAnswering.from_pretrained(config.model, revision=config.revision)
        self.model.to(device).eval()
        self.device = device
        logger.info("qa distance model loaded", extra={"model": config.model, "device": device})

    def _question(self, question: str) -> str:
        ids = self.tokenizer(question, add_special_tokens=False)["input_ids"][: self.config.max_question_tokens]
        return self.tokenizer.decode(ids)

    def read(self, question: str, texts: list[str]) -> list[dict[str, Any]]:
        c, torch = self.config, self.torch
        q = self._question(question)
        enc = self.tokenizer(
            [q] * len(texts),
            [t if t.strip() else "." for t in texts],
            truncation="only_second",
            max_length=c.max_length,
            stride=c.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
            return_tensors="pt",
        )
        owner = enc.pop("overflow_to_sample_mapping").tolist()
        offsets = enc.pop("offset_mapping")
        starts, ends = [], []
        with torch.no_grad():
            for i in range(0, len(owner), c.batch_size):
                batch = {k: v[i : i + c.batch_size].to(self.device) for k, v in enc.items()}
                out = self.model(**batch)
                starts.append(out.start_logits.float().cpu())
                ends.append(out.end_logits.float().cpu())
        start, end = torch.cat(starts), torch.cat(ends)
        best: list[tuple[float, str]] = [(-math.inf, "")] * len(texts)
        n = start.shape[1]
        band = torch.triu(torch.ones(n, n, dtype=torch.bool)) & ~torch.triu(
            torch.ones(n, n, dtype=torch.bool), diagonal=c.max_answer_tokens
        )
        for w, t in enumerate(owner):
            context = torch.tensor([s == 1 for s in enc.sequence_ids(w)])
            valid = band & context[:, None] & context[None, :]
            scores = (start[w][:, None] + end[w][None, :]).masked_fill(~valid, -math.inf)
            k = int(scores.argmax())
            i, j = divmod(k, n)
            diff = float(scores[i, j]) - float(start[w, 0] + end[w, 0])
            if diff > best[t][0]:
                best[t] = (diff, texts[t][int(offsets[w, i, 0]) : int(offsets[w, j, 1])])
        out = []
        for diff, span in best:
            p = 1 / (1 + math.exp(-diff))
            out.append({"drift": 1 - p, "p_answerable": p, "span": span})
        return out

    def gap(self, current: dict[str, Any], previous: dict[str, Any]) -> float:
        return 1 - token_f1(current["span"], previous["span"])


ESTIMATORS: dict[str, Callable[[DistanceConfig], DistanceEstimator]] = {"qa": QADistance, "lexical": LexicalDistance}


def preceding(agents: list[str]) -> list[int | None]:
    """Index of each response's relevant preceding response: the latest earlier one by another agent."""
    out: list[int | None] = []
    for t, agent in enumerate(agents):
        out.append(next((j for j in range(t - 1, -1, -1) if agents[j] != agent), None))
    return out


def situational_distances(
    estimator: DistanceEstimator, question: str, agents: list[str], texts: list[str]
) -> list[dict[str, Any]]:
    """[{question_drift, interaction_gap (None when missing), preceding, ...estimator record}] per response."""
    records = estimator.read(question, texts)
    out = []
    for rec, prev in zip(records, preceding(agents), strict=True):
        gap = None if prev is None else float(estimator.gap(rec, records[prev]))
        extra = {k: v for k, v in rec.items() if k not in ("drift", "words")}
        out.append({"question_drift": float(rec["drift"]), "interaction_gap": gap, "preceding": prev, **extra})
    return out
