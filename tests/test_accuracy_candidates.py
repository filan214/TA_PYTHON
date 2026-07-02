"""Tahap 6e DoD (D11) — kandidat perbaikan akurasi (Croston/TSB & ensemble).

Uji sifat inti tanpa fit model berat: Croston/TSB memberi ramalan one-step non-negatif
& masuk akal pada deret intermittent; ensemble mean = rata-rata; invMAE bebas-leakage
(minggu pertama = rata-rata sederhana).
"""
import numpy as np
import pandas as pd
import pytest

from src.models import croston as C
from src.models import ensemble as E


# --- Croston / TSB ----------------------------------------------------------

def test_croston_intermittent_forecast_positive_rate():
    # permintaan tiap 3 minggu -> rate ~ ukuran/interval, non-negatif, tak meledak
    y = [0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3]
    fc = C.croston_forecast(y, alpha=0.2)
    assert (fc >= 0).all()
    assert fc[0] == 0.0                                  # belum ada state
    # setelah permintaan pertama (t=2), rate ~ 3/3 = 1 (ukuran 3 / interval 3)
    assert fc[5] == pytest.approx(1.0, abs=0.4)


def test_tsb_updates_probability_every_period():
    y = [0, 5, 0, 0, 5, 0, 0, 0, 5]
    fc = C.tsb_forecast(y, alpha=0.3)
    assert (fc >= 0).all()
    assert fc[0] == 0.0
    # rate = prob*size, harus di bawah ukuran permintaan (prob<1)
    assert fc.max() < 5.0


def test_tune_alpha_returns_grid_value():
    y = [0, 2, 0, 3, 0, 0, 4, 0, 1, 0, 2, 0]
    a = C.tune_alpha(y, n_train=9, method="croston")
    assert a in C.ALPHA_GRID


# --- Ensemble ---------------------------------------------------------------

def _pair(store, brand, y_true, rf, lstm):
    weeks = pd.date_range("2025-01-06", periods=len(y_true), freq="W-MON")
    return pd.DataFrame({"store": store, "brand": brand, "week_start": weeks,
                         "y_true_rf": np.asarray(y_true, float),
                         "y_true_lstm": np.asarray(y_true, float),
                         "y_pred_rf": np.asarray(rf, float),
                         "y_pred_lstm": np.asarray(lstm, float)})


def test_ensemble_mean_is_average():
    m = _pair("Toko_A", "Xiaomi", [1, 2, 3], [2, 2, 2], [4, 4, 4])
    out = E.ensemble_mean(m)
    assert list(out) == [3.0, 3.0, 3.0]


def test_ensemble_invmae_first_week_is_simple_mean():
    # minggu pertama tak punya riwayat galat -> rata-rata sederhana (bebas-leakage)
    out = E.ensemble_invmae_series([5, 5, 5], [4, 4, 4], [6, 8, 10])
    assert out[0] == pytest.approx(5.0)                 # (4+6)/2
    # setelah RF terbukti lebih baik, bobot condong ke RF -> prediksi mendekati RF
    assert out[2] < 7.0
