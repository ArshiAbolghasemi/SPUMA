"""Startup check: the capabilities this experiment needs vs. what the configured backends provide.

Runs before any model is loaded (capabilities come from config + installed library), so an
unsupported combination fails in seconds instead of after hours of generation.
"""

import logging
from typing import Any

from spuma.config import get_settings
from spuma.models.backend import BackendCapabilityError
from spuma.models.engines import backend_class

logger = logging.getLogger(__name__)


def execution_plan() -> dict[str, dict[str, Any]]:
    """Per model in use: generation backend, scoring backend, and the capabilities each must have."""
    config = get_settings()
    uq = config.uncertainty
    tools_on = config.tools.enabled and any(t.enabled for t in config.tools.registry.values())
    plan: dict[str, dict[str, Any]] = {}
    for agent in config.agents.enabled:
        key = agent.model
        model = config.model_for(agent)
        entry = plan.setdefault(
            key,
            {
                "model": model.name,
                "generation_backend": model.backend,
                "scoring_backend": model.scoring_kind,
                "scores_at_generation": model.scores_at_generation,
                "agents": [],
                "generation_requires": {"generation"},
                "scoring_requires": set(),
            },
        )
        entry["agents"].append(agent.id)
        gen, score = entry["generation_requires"], entry["scoring_requires"]
        if tools_on and agent.allowed_tools:
            gen.add("tool_calling")
        if uq.semantic_entropy.enabled:
            gen |= {"multiple_completions", "batch_generation"}
        if uq.predictive_entropy.enabled or uq.nll.enabled:
            if model.scores_at_generation:
                gen.add("generated_token_logprobs")
                if uq.predictive_entropy.enabled:
                    gen.add("full_vocabulary_logprobs")
            else:
                score.add("teacher_forced_scoring")
                if uq.predictive_entropy.enabled:
                    score.add("full_vocabulary_scoring")
    return plan


def validate_backends() -> dict[str, dict[str, Any]]:
    """Raise BackendCapabilityError listing every unmet requirement of the settings; returns the plan."""
    config = get_settings()
    plan = execution_plan()
    problems = []
    for key, entry in plan.items():
        model = config.all_models()[key]
        for role, kind in (("generation", entry["generation_backend"]), ("scoring", entry["scoring_backend"])):
            needed = sorted(entry[f"{role}_requires"])
            if not needed:
                continue
            try:
                caps = backend_class(kind).capabilities_for(model)
            except BackendCapabilityError as e:
                problems.append(f"model {key!r} {role} backend {kind!r}: {e}")
                continue
            if missing := caps.missing(*needed):
                problems.append(f"model {key!r} {role} backend {kind!r} lacks {missing}")
        if entry["scoring_backend"] == "vllm" and "full_vocabulary_scoring" in entry["scoring_requires"]:
            logger.warning(
                "PE via vLLM teacher-forced scoring transfers |V| x (prompt + continuation) floats per step; "
                "consider scoring_backend: transformers",
                extra={"model": key},
            )
    out = {k: {**v, **{r: sorted(v[r]) for r in ("generation_requires", "scoring_requires")}} for k, v in plan.items()}
    if problems:
        logger.error("backend capability validation failed", extra={"problems": problems, "plan": out})
        raise BackendCapabilityError("; ".join(problems))
    logger.info("backend capabilities validated", extra={"plan": out})
    return out
