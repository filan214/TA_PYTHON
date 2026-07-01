"""Loader config.yaml -> objek Config.

Path di config.yaml diselesaikan relatif terhadap lokasi file config,
sehingga pipeline bisa dijalankan dari direktori mana pun.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Paths:
    root: Path
    raw_csv: Path
    interim: Path
    processed: Path
    models: Path
    figures: Path
    results: Path

    def ensure(self) -> None:
        """Buat direktori output bila belum ada (raw_csv tidak dibuat)."""
        for p in (self.interim, self.processed, self.models, self.figures, self.results):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    seed: int
    paths: Paths
    raw: dict[str, Any]  # seluruh isi yaml untuk parameter lain (lags, grids, dst.)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    root = path.parent
    p = raw.get("paths", {})

    def resolve(key: str) -> Path:
        return (root / p[key]).resolve()

    paths = Paths(
        root=root,
        raw_csv=resolve("raw_csv"),
        interim=resolve("interim"),
        processed=resolve("processed"),
        models=resolve("models"),
        figures=resolve("figures"),
        results=resolve("results"),
    )
    return Config(seed=int(raw.get("seed", 42)), paths=paths, raw=raw)
