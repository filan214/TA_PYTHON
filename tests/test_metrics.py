"""Tahap 7 DoD — metrik evaluasi & agregasi (§7).

Uji nilai metrik pada contoh manual, keamanan sMAPE/MAPE pada minggu nol,
agregasi mean & weighted-by-volume, pemilihan varian terbaik, serta orkestrasi
(build summary + figur) — semua offline & deterministik (tanpa fit model).
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import metrics as M


def _preds(algo, variant, rng_seed=0):
    """Frame prediksi sintetis: 2 deret × 6 minggu."""
    rng = np.random.default_rng(rng_seed)
    weeks = pd.date_range("2025-01-06", periods=6, freq="W-MON")
    frames = []
    for store, brand in [("Toko_A", "Xiaomi"), ("Toko_B", "OPPO")]:
        yt = rng.integers(0, 5, size=6).astype(float)
        frames.append(pd.DataFrame({
            "store": store, "brand": brand, "week_start": weeks,
            "y_true": yt, "y_pred": yt + rng.normal(0, 1, size=6),
        }))
    return pd.concat(frames, ignore_index=True)


# --- Metrik dasar -----------------------------------------------------------

def test_mae_rmse_manual():
    a = [2, 0, 4, 10]
    p = [3, 1, 4, 8]
    assert M.mae(a, p) == pytest.approx(1.0)
    assert M.rmse(a, p) == pytest.approx(np.sqrt(1.5))


def test_mape_only_nonzero_actuals():
    # aktual 0 dikecualikan; sisanya: 1/2, 0/4, 2/10 -> mean *100
    a = [2, 0, 4, 10]
    p = [3, 1, 4, 8]
    assert M.mape(a, p) == pytest.approx((0.5 + 0.0 + 0.2) / 3 * 100)


def test_mape_all_zero_is_nan():
    assert np.isnan(M.mape([0, 0, 0], [1, 2, 3]))


def test_smape_safe_on_zero_actual():
    # 0/0 -> 0 (tak meledak); (0,5) -> 2*5/5 = 2.0
    val = M.smape([0, 0], [0, 5])
    assert np.isfinite(val)
    assert val == pytest.approx((0.0 + 2.0) / 2 * 100)


def test_smape_manual():
    a = [2, 0, 4, 10]
    p = [3, 1, 4, 8]
    expected = (0.4 + 2.0 + 0.0 + 4 / 18) / 4 * 100
    assert M.smape(a, p) == pytest.approx(expected)


# --- Agregasi ---------------------------------------------------------------

def test_wmean_ignores_nan_and_zero_weight():
    # nilai NaN & bobot 0 diabaikan; sisanya rata-rata berbobot
    assert M._wmean([1.0, np.nan, 3.0], [1.0, 5.0, 3.0]) == pytest.approx(
        (1 * 1 + 3 * 3) / (1 + 3))
    assert np.isnan(M._wmean([np.nan], [1.0]))
    assert np.isnan(M._wmean([1.0], [0.0]))


def test_series_metrics_counts_and_volume():
    m = M.series_metrics([0, 2, 0, 4], [0, 2, 1, 4])
    assert m["n"] == 4 and m["n_nonzero"] == 2 and m["volume"] == 6.0
    assert m["MAE"] == pytest.approx(0.25)


def test_summary_and_best_variant():
    preds = {("sarimax", "baseline"): _preds("sarimax", "baseline", 1),
             ("sarimax", "gt"): _preds("sarimax", "gt", 2),
             ("rf", "baseline"): _preds("rf", "baseline", 3),
             ("rf", "gt"): _preds("rf", "gt", 4)}
    ps = M.per_series_metrics(preds)
    assert len(ps) == 2 * 2 * 2  # (algo,variant,series)
    summary = M.summary_from_per_series(ps)
    assert set(summary["algo"]) == {"sarimax", "rf"}
    assert summary["is_best"].sum() == 1                # tepat satu terbaik
    best = M.best_variant_by_mae(summary)
    assert set(best) == {"sarimax", "rf"}
    assert all(v in ("baseline", "gt") for v in best.values())


def test_representative_series_by_volume():
    preds = {("rf", "gt"): _preds("rf", "gt", 5)}
    reps = M.representative_series(preds, n=1)
    assert len(reps) == 1 and reps[0] in [("Toko_A", "Xiaomi"), ("Toko_B", "OPPO")]


def test_plot_actual_vs_pred(tmp_path):
    preds = {a: _preds(a[0], a[1], i) for i, a in enumerate(
        [("sarimax", "gt"), ("rf", "gt"), ("lstm", "gt")])}
    saved = M.plot_actual_vs_pred(preds, [("Toko_A", "Xiaomi")], "gt", tmp_path)
    assert len(saved) == 1 and saved[0].exists()
