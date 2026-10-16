"""Logging setup: `spuma.*` loggers at the configured level, JSON or text, console and/or file.

Structured context goes in `extra={...}`; the JSON formatter emits every extra field, plus the
run id set by `set_run_id`. Third-party chatter is capped at WARNING.
"""

import json
import logging
import sys
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from spuma.config import get_settings

PROJECT_LOGGER = "spuma"
NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "filelock", "fsspec", "huggingface_hub", "asyncio", "datasets")
_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
_run_id = "-"


def set_run_id(run_id: str) -> None:
    global _run_id
    _run_id = run_id


class _RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload |= {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(run_name: str = "spuma") -> Path | None:
    """Configure the root logger from the `logging:` settings; returns the log file path when file logging is on."""
    config = get_settings().logging
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    formatter = (
        JsonFormatter()
        if config.format == "json"
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(run_id)s]: %(message)s")
    )
    handlers: list[logging.Handler] = []
    if config.console:
        handlers.append(logging.StreamHandler(sys.stderr))
    log_file = None
    if config.file:
        config.directory.mkdir(parents=True, exist_ok=True)
        log_file = config.directory / f"{run_name}-{datetime.now(UTC):%Y%m%dT%H%M%S}.log"
        handlers.append(logging.FileHandler(log_file))
    for h in handlers:
        h.setFormatter(formatter)
        h.addFilter(_RunIdFilter())
        root.addHandler(h)
    root.setLevel(logging.INFO)
    logging.getLogger(PROJECT_LOGGER).setLevel(config.level)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    return log_file


def progress[T](items: Iterable[T], total: int, desc: str) -> Iterator[T]:
    """Iterate with a tqdm progress bar (count, rate, ETA). Console log lines are printed above the bar
    instead of breaking it; the bar is hidden when stderr is not a terminal (log files, CI, tests)."""
    with logging_redirect_tqdm():
        yield from tqdm(items, total=total, desc=desc, unit="problem", dynamic_ncols=True, disable=None)


@contextmanager
def progress_counter(total: int, desc: str) -> Iterator[Callable[[], None]]:
    """A tqdm bar advanced by calling the yielded function (work finishing in any order)."""
    with logging_redirect_tqdm(), tqdm(total=total, desc=desc, unit="run", dynamic_ncols=True, disable=None) as bar:
        yield lambda: bar.update(1)
