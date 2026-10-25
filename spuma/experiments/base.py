"""An experiment: a name plus the settings file it runs with.

The settings file holds everything that defines the experiment (`dataset:`, `model:`, `agents:`,
uncertainty, SPN, evaluation, and `experiment.name`, its artifact directory). `ExperimentRunner` is generic;
an experiment that needs code of its own (a dataset provider or grader the registries do not offer)
subclasses `Experiment` and overrides `runner()`.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from spuma.common.logs import set_run_id, setup_logging
from spuma.config import SETTINGS_FILE_ENV, Settings, get_settings
from spuma.experiments.runner import ExperimentRunner
from spuma.models.engines import release_backends

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Experiment:
    name: str
    config_file: Path
    description: str = ""
    architecture: str = ""  # feedbackloop | star | sequential
    dataset: str = ""

    def activate(self) -> Settings:
        """Make this experiment's settings file the process-wide settings (`get_settings()`)."""
        if not self.config_file.is_file():
            raise FileNotFoundError(f"experiment {self.name!r}: settings file {self.config_file} not found")
        os.environ[SETTINGS_FILE_ENV] = str(self.config_file)
        get_settings.cache_clear()
        return get_settings()

    def runner(self) -> ExperimentRunner:
        return ExperimentRunner()

    def run(self, phases: list[str] | None = None, limit: int | None = None) -> None:
        settings = self.activate()
        setup_logging(settings.experiment.name)
        runner = self.runner()
        set_run_id(runner.run_id)
        logger.info("experiment", extra={"experiment": self.name, "config": str(self.config_file), "phases": phases})
        try:
            runner.run(phases=phases, limit=limit)
        finally:
            release_backends()
