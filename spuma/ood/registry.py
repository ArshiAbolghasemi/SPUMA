"""Registered OOD evaluations, by name. Each name maps to `config/<architecture>/<name>.yaml`, whose `ood.name` is
the same name. A new OOD evaluation adds its settings file and its (architecture, name) here."""

from pathlib import Path

from spuma.config import PROJECT_ROOT

OOD_EVALUATIONS: dict[str, Path] = {
    name: PROJECT_ROOT / "config" / architecture / f"{name}.yaml"
    for architecture, name in (("feedbackloop", "ood_morehopqa_math_ma_qwen2_5_7b_feedbackloop"),)
}
