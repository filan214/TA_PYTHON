"""Tahap 7 DoD — uji Diebold-Mariano (§7).

Prediksi identik -> tak ada beda (p~1). Prediksi jelas berbeda -> signifikan &
`better` benar. Verifikasi prasyarat D9 (origin sepadan) menolak frame tak selaras.
Kedua kelompok perbandingan (antar-algoritma & ablation GT) terbentuk benar.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import diebold_mariano as DM


def _frame(store, brand, y_true, y_pred):
    weeks = pd.date_range("2025-01-06", periods=len(y_true), freq="W-MON")
    return pd.DataFrame({"store": store, "brand": brand, "week_start": weeks,
                         "y_true": np.asarray(y_true, float),
                         "y_pred": np.asarray(y_pred, float)})


def _preds_all(n=12, seed=0):
    """Enam set prediksi sintetis dgn origin identik (memenuhi D9)."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2025-01-06", periods=n, freq="W-MON")
    yt = rng.integers(0, 6, size=n).astype(float)
    out = {}
    for algo, noise in [("sarimax", 1.5), ("rf", 0.6), ("lstm", 1.0)]:
        for variant in ["baseline", "gt"]:
            eps = rng.normal(0, noise, size=n)
            out[(algo, variant)] = pd.DataFrame({
                "store": "Toko_A", "brand": "Xiaomi", "week_start": weeks,
                "y_true": yt, "y_pred": np.clip(yt + eps, 0, None)})
    return out


# --- Statistik DM -----------------------------------------------------------

def test_identical_predictions_pvalue_one():
    a = [1, 3, 0, 5, 2, 4, 1, 0]
    p = [2, 2, 1, 4, 3, 3, 2, 1]
    r = DM.diebold_mariano(a, p, p)              # model1 == model2
    assert r["dm_stat"] == 0.0
    assert r["p_value"] == pytest.approx(1.0)
    assert r["better"] == 0


def test_clear_difference_detected():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 8, size=60).astype(float)
    good = a + rng.normal(0, 0.3, size=60)       # nyaris tepat
    bad = a + rng.normal(0, 4.0, size=60)        # jauh
    r = DM.diebold_mariano(a, bad, good)
    assert r["better"] == 2                       # model2 (good) unggul
    assert r["p_value"] < 0.05


def test_better_flag_sign():
    a = [5, 5, 5, 5]
    r = DM.diebold_mariano(a, [5, 5, 5, 5], [0, 0, 0, 0])  # m1 sempurna
    assert r["better"] == 1


# --- Prasyarat D9: origin sepadan -------------------------------------------

def test_align_frames_rejects_mismatched_origins():
    df1 = _frame("Toko_A", "Xiaomi", [1, 2, 3], [1, 2, 3])
    df2 = _frame("Toko_A", "Xiaomi", [1, 2, 3], [2, 2, 2])
    df2["week_start"] = df2["week_start"] + pd.Timedelta(weeks=10)  # geser origin
    with pytest.raises(ValueError):
        DM.align_frames(df1, df2)


def test_align_frames_rejects_different_ytrue():
    df1 = _frame("Toko_A", "Xiaomi", [1, 2, 3], [1, 2, 3])
    df2 = _frame("Toko_A", "Xiaomi", [9, 9, 9], [1, 2, 3])         # y_true beda
    with pytest.raises(ValueError):
        DM.align_frames(df1, df2)


def test_dm_from_frames_pools_series():
    df1 = pd.concat([_frame("Toko_A", "Xiaomi", [1, 2, 3, 4], [1, 2, 3, 4]),
                     _frame("Toko_B", "OPPO", [0, 1, 2, 3], [0, 1, 2, 3])],
                    ignore_index=True)
    df2 = df1.copy()
    df2["y_pred"] = df2["y_true"] + 2.0                            # selalu lebih buruk
    r = DM.dm_from_frames(df1, df2)
    assert r["n"] == 8 and r["better"] == 1                        # df1 unggul


# --- Kelompok perbandingan --------------------------------------------------

def test_run_comparisons_two_groups():
    preds = _preds_all()
    best = {"sarimax": "gt", "rf": "gt", "lstm": "baseline"}
    dm = DM.run_comparisons(preds, best)
    groups = set(dm["group"])
    assert groups == {"antar_algoritma", "ablation_gt"}
    assert (dm["group"] == "antar_algoritma").sum() == 3           # 3 pasangan algo
    assert (dm["group"] == "ablation_gt").sum() == 3               # 3 algoritma
    assert {"dm_stat", "p_value", "significant", "better"} <= set(dm.columns)


def test_build_gt_ablation_series_and_summary():
    preds = _preds_all()
    tab = DM.build_gt_ablation(preds)
    assert set(tab["row_type"]) == {"deret", "ringkasan"}
    # satu deret per algoritma (data sintetis 1 deret) + 1 baris ringkasan/algo
    for algo in ["SARIMAX", "RF", "LSTM"]:
        sub = tab[(tab["algo"] == algo)]
        n_series = (sub["row_type"] == "deret").sum()
        summ = sub[sub["row_type"] == "ringkasan"].iloc[0]
        counts = summ[["n_membaik", "n_memburuk", "n_tak_signifikan"]].sum()
        assert counts == n_series                                   # cacah konsisten
