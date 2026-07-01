"""Split temporal & CV deret waktu (D3, D4).

D3: split 80:20 tanpa shuffle -> 147 minggu latih / 37 minggu uji pada 184 minggu.
D4: TimeSeriesSplit (expanding window) untuk GridSearchCV di Tahap 6.

Split dilakukan berdasarkan URUTAN WAKTU (week_start), bukan posisi baris — supaya
partisi uji tetap 37 minggu terakhir meski baris warm-up (lag NaN) sudah di-drop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def train_cutoff_week(weeks: pd.DatetimeIndex, train_ratio: float = 0.8) -> pd.Timestamp:
    """Minggu pertama partisi UJI: elemen ke-`floor(n*ratio)` dari minggu terurut unik."""
    uniq = pd.DatetimeIndex(sorted(pd.Index(weeks).unique()))
    n_train = int(np.floor(len(uniq) * train_ratio))
    return uniq[n_train]


def temporal_masks(
    week_series: pd.Series,
    train_ratio: float = 0.8,
    cutoff: pd.Timestamp | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Kembalikan (mask_train, mask_test) boolean berdasar ambang waktu D3.

    Latih = minggu < cutoff; Uji = minggu >= cutoff. `cutoff` sebaiknya dihitung
    dari grid PENUH 184 minggu (bukan subset supervised pasca-drop warm-up), agar
    partisi uji tetap tepat 37 minggu terakhir. Bila None, dihitung dari week_series.
    """
    if cutoff is None:
        cutoff = train_cutoff_week(pd.DatetimeIndex(week_series), train_ratio)
    is_test = week_series >= cutoff
    return ~is_test, is_test


def time_series_splits(n_samples: int, n_splits: int = 5):
    """Bungkus sklearn TimeSeriesSplit (train selalu mendahului val)."""
    from sklearn.model_selection import TimeSeriesSplit

    return TimeSeriesSplit(n_splits=n_splits).split(np.arange(n_samples))
