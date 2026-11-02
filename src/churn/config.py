"""Config loading. One YAML file, resolved to absolute paths."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    """dict with attribute access for nested sections."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(value) if isinstance(value, dict) else value

    def path(self, name: str) -> Path:
        p = ROOT / self["paths"][name]
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def day_one(self) -> date:
        return date.fromisoformat(self["calendar"]["day_one"])

    def day_to_date(self, day: int) -> date:
        return self.day_one + timedelta(days=int(day) - 1)


@lru_cache
def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)
