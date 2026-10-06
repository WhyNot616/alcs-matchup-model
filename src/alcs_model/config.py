"""Configuration loading and project paths."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "output"
DOCS = ROOT / "docs"
DASHBOARD_TEMPLATE = ROOT / "dashboard" / "template.html"
DEFAULT_CONFIG = ROOT / "config" / "series.yaml"


@dataclass
class Config:
    raw: dict[str, Any]

    @property
    def season(self) -> int:
        return int(self.raw["season"])

    @property
    def teams(self) -> list[str]:
        return list(self.raw["teams"].keys())

    def team_name(self, abbr: str) -> str:
        return self.raw["teams"][abbr]["name"]

    def venue(self, abbr: str) -> str:
        return self.raw["teams"][abbr]["venue"]

    def opponent(self, abbr: str) -> str:
        a, b = self.teams
        return b if abbr == a else a

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def sim(self) -> dict[str, Any]:
        return self.raw["sim"]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if len(raw["teams"]) != 2:
        raise ValueError("config.teams must list exactly two teams")
    return Config(raw)


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_PROCESSED, OUTPUT, DOCS):
        d.mkdir(parents=True, exist_ok=True)
