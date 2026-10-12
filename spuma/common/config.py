"""Shared config base class and the `logging:` section."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class StrictConfig(BaseModel):
    """Base of every package config section: immutable, unknown keys rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class LoggingConfig(StrictConfig):
    """`logging:` section."""

    level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = "INFO"
    format: Literal["json", "text"] = "json"
    console: bool = True
    file: bool = True
    directory: Path = Path("logs")

    @field_validator("level", mode="before")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()
