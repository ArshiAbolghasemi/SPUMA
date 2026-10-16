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
        gen = entry["generation_requires"]
        if uq.monte_carlo_enabled:
            gen |= {"multiple_completions", "batch_generation"}
        if uq.token_logprobs_needed:
            gen |= {"generated_token_logprobs", "teacher_forced_scoring"}
    return plan


def validate_backends() -> dict[str, dict[str, Any]]:
    """Raise BackendCapabilityError listing every unmet requirement of the settings; returns the plan."""
    config = get_settings()
    plan = execution_plan()
    problems = []
    for key, entry in plan.items():
        model = config.all_models()[key]
        if config.uncertainty.token_logprobs_needed and not model.scores_at_generation:
            problems.append(
                f"model {key!r}: uncertainty is measured right after each response with the generating engine; "
                "set model.scoring_backend: same"
            )
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
    out = {k: {**v, **{r: sorted(v[r]) for r in ("generation_requires", "scoring_requires")}} for k, v in plan.items()}
    if problems:
        logger.error("backend capability validation failed", extra={"problems": problems, "plan": out})
        raise BackendCapabilityError("; ".join(problems))
    logger.info("backend capabilities validated", extra={"plan": out})
    return out
