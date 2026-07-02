"""Antarmuka model peramalan (Bab III §3.1.6).

Semua algoritma (SARIMAX/RF/LSTM) turun dari `Forecaster` agar Tahap 7 dapat
mengevaluasi seragam. Kontrak minimal: fit -> predict horizon -> name.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    @abstractmethod
    def fit(self, y, exog=None) -> "Forecaster":
        """Latih pada target latih `y` (+ eksogen opsional)."""

    @abstractmethod
    def predict(self, horizon: int, exog_future=None) -> np.ndarray:
        """Ramalkan `horizon` langkah ke depan (+ eksogen masa depan opsional)."""

    def name(self) -> str:
        return self.__class__.__name__
