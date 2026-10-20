"""`ood:` settings file: target dataset, frozen checkpoint and the source experiment it was trained in."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from spuma.common.config import as_path, build_config, check_choice
from spuma.config import PROJECT_ROOT
from spuma.dataset.config import DatasetConfig
from spuma.evaluation.metrics import METRICS


@dataclass(frozen=True)
class OODConfig:
    """`ood:` section.

    `source_config` is the experiment settings file the checkpoint was trained with; agents, prompts,
    generation, uncertainty, aggregation and features all come from it, and only `dataset` and the output
    location are replaced. `id_evaluation` defaults to the `evaluation.json` next to the checkpoint's
    `models/` directory.
    """

    name: str
    source_config: Path
    checkpoint: Path
    dataset: DatasetConfig
    risk_model: str = "spn"
    id_evaluation: Path | None = None
    output_dir: Path = Path("artifacts")
    metrics: list[str] = field(
        default_factory=lambda: ["auroc", "auprc", "auarc", "aurc", "brier", "ece", "nll", "accuracy"]
    )

    def __post_init__(self) -> None:
        for name in ("source_config", "checkpoint", "id_evaluation", "output_dir"):
            if getattr(self, name) is None:
                continue
            as_path(self, name)
            if not getattr(self, name).is_absolute():
                object.__setattr__(self, name, PROJECT_ROOT / getattr(self, name))
        check_choice("ood.risk_model", self.risk_model, ("spn", "spn_class_conditional"))
        for m in self.metrics:
            check_choice("ood.metrics[]", m, tuple(METRICS))
        if self.dataset.train_samples or self.dataset.validation_samples:
            raise ValueError("ood.dataset: an OOD set is test-only; set train_samples and validation_samples to 0")


def load_ood_config(path: str | Path) -> OODConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    if set(raw) != {"ood"}:
        raise ValueError(f"{path}: expected exactly one top-level section `ood:`, got {sorted(raw)}")
    config = build_config(OODConfig, raw["ood"], "ood.")
    if config.name != Path(path).stem:
        raise ValueError(f"{path}: ood.name {config.name!r} must equal the file name {Path(path).stem!r}")
    return config
