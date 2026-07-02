"""Tahap 6d DoD — baseline naif (D10).

Naive one-step = lag-1; seasonal-naive = lag-52 dgn fallback lag-1 di awal deret.
Prediksi test sepadan (origin & y_true) dgn model utama -> valid utk MASE & DM.
"""
import numpy as np
import pandas as pd

from src.models import naive as N


def test_one_step_naive_lag1():
    u = [2, 3, 0, 5, 1]
    fc = N.one_step_naive(u, seasonal=False)
    # posisi 0 tak punya riwayat -> NaN; sisanya = nilai sebelumnya
    assert np.isnan(fc[0])
    assert list(fc[1:]) == [2, 3, 0, 5]


def test_seasonal_naive_fallbacks_then_lag_m():
    m = 3
    u = [1, 2, 3, 4, 5, 6, 7]
    fc = N.one_step_naive(u, seasonal=True, m=m)
    # t<m -> fallback lag-1 (t=0 NaN); t>=m -> lag-m
    assert np.isnan(fc[0])
    assert fc[1] == 1 and fc[2] == 2          # fallback lag-1
    assert fc[3] == 1 and fc[4] == 2 and fc[5] == 3 and fc[6] == 4  # lag-m


def test_naive_clips_negative_free():
    # tak ada nilai negatif pada units -> hasil tetap non-negatif
    fc = N.one_step_naive([0, 0, 4], seasonal=False)
    assert (fc[~np.isnan(fc)] >= 0).all()
