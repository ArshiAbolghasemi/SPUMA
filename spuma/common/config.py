"""Config plumbing shared by every package: build frozen config dataclasses from plain mappings (Dynaconf
output), small validators for `__post_init__`, and the `logging:` section."""

import dataclasses
import types
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints


def _dataclass_in(tp: object) -> type | None:
    """The dataclass inside a type (`X` or `X | None`), if any."""
    return next((t for t in (tp, *get_args(tp)) if isinstance(t, type) and dataclasses.is_dataclass(t)), None)


def _strip_optional(tp: object) -> object:
    if get_origin(tp) in (Union, types.UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        return args[0] if len(args) == 1 else tp
    return tp


def _build_value(tp: object, value: Any, path: str) -> Any:
    """Build `value` for a field of type `tp`: dataclasses, and lists / str-keyed dicts of dataclasses."""
    if value is None:
        return None
    tp = _strip_optional(tp)
    origin, args = get_origin(tp), get_args(tp)
    if origin is list and args and (item := _dataclass_in(args[0])):
        if not isinstance(value, list):
            raise TypeError(f"setting {path.rstrip('.')} must be a list, got {type(value).__name__}")
        return [build_config(item, v, f"{path}{i}.") for i, v in enumerate(value)]
    if origin is dict and len(args) == 2 and (item := _dataclass_in(args[1])):
        if not isinstance(value, Mapping):
            raise TypeError(f"setting {path.rstrip('.')} must be a mapping, got {type(value).__name__}")
        return {k: build_config(item, v, f"{path}{k}.") for k, v in value.items()}
    if sub := _dataclass_in(tp):
        return build_config(sub, value, path)
    return value


def build_config[T](cls: type[T], raw: Mapping[str, Any] | T | None, path: str = "") -> T:
    """Recursively build `cls`; unknown or missing keys fail with the dotted key path."""
    if isinstance(raw, cls):
        return raw
    raw = {} if raw is None else raw
    if not isinstance(raw, Mapping):
        raise TypeError(f"setting {path.rstrip('.') or cls.__name__} must be a mapping, got {type(raw).__name__}")
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass")
    fields = dataclasses.fields(cls)
    names = {f.name for f in fields}
    unknown = sorted(set(raw) - names)
    if unknown:
        raise ValueError(f"unknown setting(s): {', '.join(path + k for k in unknown)}")
    missing = [
        f.name
        for f in fields
        if raw.get(f.name) is None  # absent or null
        and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    ]
    if missing:
        raise ValueError(f"missing required setting(s): {', '.join(path + k for k in missing)}")
    hints = get_type_hints(cls)
    kwargs = {name: _build_value(hints[name], value, f"{path}{name}.") for name, value in raw.items()}
    return cls(**kwargs)


def to_plain(value: Any) -> Any:
    """Config dataclasses -> JSON-able dicts/lists (Paths as strings)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


# --- validators for __post_init__ ------------------------------------------------------------------


def check_choice(key: str, value: Any, choices: Iterable[Any]) -> None:
    choices = tuple(choices)
    if value not in choices:
        raise ValueError(f"{key} must be one of {list(choices)}, got {value!r}")


def check_range(
    key: str,
    value: float | None,
    *,
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{key} must be a number, got {value!r}")
    if (
        (gt is not None and not value > gt)
        or (ge is not None and not value >= ge)
        or (lt is not None and not value < lt)
        or (le is not None and not value <= le)
    ):
        bounds = ", ".join(f"{op} {b}" for op, b in (("<", lt), (">", gt), (">=", ge), ("<=", le)) if b is not None)
        raise ValueError(f"{key} must be {bounds}, got {value!r}")


def as_path(obj: object, name: str) -> None:
    """Coerce a str field to Path on a frozen dataclass (YAML and env vars give strings)."""
    object.__setattr__(obj, name, Path(getattr(obj, name)))


@dataclass(frozen=True)
class LoggingConfig:
    """`logging:` section. `level` applies to SPUMA's own loggers."""

    level: str = "INFO"
    format: str = "json"
    console: bool = True
    file: bool = True
    directory: Path = Path("logs")

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", str(self.level).upper())
        check_choice("logging.level", self.level, ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"))
        check_choice("logging.format", self.format, ("json", "text"))
        as_path(self, "directory")
