"""Utilitas I/O & reproducibility (§6, §9)."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Fiksasi semua sumber keacakan (numpy, random, PYTHONHASHSEED, tensorflow)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # tensorflow opsional pada tahap awal
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:  # pragma: no cover - tf belum terpasang pada Tahap 1
        logger.debug("tensorflow belum tersedia; lewati tf seed")


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
