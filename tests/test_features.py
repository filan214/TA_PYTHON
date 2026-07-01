"""Tahap 5 DoD — rekayasa fitur, 2 varian, anti-leakage (§5, §7).

Verifikasi: make_supervised untuk baseline & gt (kolom gt_index hanya di gt),
tak ada NaN bocor, lag/rolling murni past-only, Fourier & kalender benar, dan
scaler di-fit hanya pada partisi latih (D3, uji=37 minggu terakhir).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.aggregate import aggregate_weekly
from src.data.clean import clean_transactions
from src.features import build
from src.utils.splits import temporal_masks, train_cutoff_week

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pos_transactions_raw.csv"


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "config.yaml")


@pytest.fixture(scope="module")
def weekly():
    return aggregate_weekly(clean_transactions(RAW))


@pytest.fixture(scope="module")
def gt(weekly, cfg):
    # GT sintetik cakupan penuh (offline) — cukup untuk menempelkan kolom gt_index.
    brands = sorted(weekly["brand"].unique())
    grid = pd.DatetimeIndex(sorted(weekly["week_start"].unique()))
    frames = [pd.DataFrame({"week_start": grid, "brand": b,
                            "gt_index": np.linspace(10, 90, len(grid))}) for b in brands]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def one_series_feats(weekly, gt, cfg):
    feats = build.build_all_features(weekly, gt, cfg)
    return feats["Toko_A|Xiaomi"]


# --- DoD: make_supervised 2 varian -----------------------------------------

def test_gt_variant_has_gt_index(one_series_feats, cfg):
    Xg, _, _ = build.make_supervised(one_series_feats, cfg, "gt")
    assert "gt_index" in Xg.columns


def test_baseline_variant_excludes_gt_index(one_series_feats, cfg):
    Xb, _, _ = build.make_supervised(one_series_feats, cfg, "baseline")
    assert "gt_index" not in Xb.columns


def test_variants_differ_only_by_gt_index(one_series_feats, cfg):
    Xb, _, _ = build.make_supervised(one_series_feats, cfg, "baseline")
    Xg, _, _ = build.make_supervised(one_series_feats, cfg, "gt")
    assert set(Xg.columns) - set(Xb.columns) == {"gt_index"}
    assert set(Xb.columns) - set(Xg.columns) == set()


def test_supervised_shapes(one_series_feats, cfg):
    Xb, yb, idx = build.make_supervised(one_series_feats, cfg, "baseline")
    # 184 minggu - 8 baris warm-up (lag_8/roll_mean_8) = 176 baris supervised.
    assert len(Xb) == len(yb) == len(idx) == 176
    # lag(8)+roll(3)+kalender(6)+fourier(4) = 21 fitur baseline; +gt_index = 22.
    assert Xb.shape[1] == 21


# --- DoD: anti-leakage ------------------------------------------------------

def test_no_nan_leaks_into_supervised(one_series_feats, cfg):
    for variant in ("baseline", "gt"):
        X, y, _ = build.make_supervised(one_series_feats, cfg, variant)
        assert not X.isna().any().any()
        assert not y.isna().any()


def test_warmup_drop_consistent_across_variants(one_series_feats, cfg):
    _, _, ib = build.make_supervised(one_series_feats, cfg, "baseline")
    _, _, ig = build.make_supervised(one_series_feats, cfg, "gt")
    assert list(ib) == list(ig)


def test_lag_and_rolling_are_past_only():
    # Deret dikenal: units = 0..19. lag_1[t]=units[t-1]; roll_mean_4[t]=mean(t-4..t-1).
    df = pd.DataFrame({
        "store": "S", "brand": "B",
        "week_start": pd.date_range("2022-01-03", periods=20, freq="W-MON"),
        "units": np.arange(20),
    })
    out = build.add_lag_roll_features(df.copy(), lags=8, rolling_windows=[4, 8])
    # Baris index 8 (units=8): lag_1=7, roll_mean_4=mean(4,5,6,7)=5.5 (TIDAK memuat 8).
    assert out.loc[8, "lag_1"] == 7
    assert out.loc[8, "roll_mean_4"] == pytest.approx(5.5)
    assert out.loc[8, "roll_mean_8"] == pytest.approx(np.mean([0, 1, 2, 3, 4, 5, 6, 7]))


def test_scaler_fit_on_train_only(one_series_feats, cfg, weekly):
    X, _, index = build.make_supervised(one_series_feats, cfg, "baseline")
    grid = pd.DatetimeIndex(sorted(weekly["week_start"].unique()))
    cutoff = train_cutoff_week(grid, float(cfg["train_ratio"]))
    scaler = build.fit_scaler_on_train(X, index, cutoff)
    mask_train, _ = temporal_masks(pd.Series(index), cutoff=cutoff)
    # data_max_ scaler harus = max partisi latih saja (bukan seluruh deret).
    train_max = X.loc[mask_train.values].max().values
    assert np.allclose(scaler.data_max_, train_max)


# --- Kalender & Fourier -----------------------------------------------------

def test_calendar_flags_lebaran_and_harbolnas(one_series_feats):
    df = one_series_feats
    lebaran_monday = pd.Timestamp("2022-05-02")  # sudah Senin
    row = df.loc[df["week_start"] == lebaran_monday]
    assert int(row["is_lebaran_window"].iloc[0]) == 1
    # Minggu memuat 11 Nov 2022 -> is_harbolnas.
    nov_monday = pd.Timestamp("2022-11-11").to_period("W").start_time
    hb = df.loc[df["week_start"] == nov_monday, "is_harbolnas"]
    assert int(hb.iloc[0]) == 1


def test_fourier_columns_and_range(one_series_feats, cfg):
    K = int(cfg["fourier_harmonics"])
    cols = [f"fourier_{fn}_{k}" for k in range(1, K + 1) for fn in ("sin", "cos")]
    for c in cols:
        assert c in one_series_feats.columns
        assert one_series_feats[c].between(-1.0, 1.0).all()


# --- Panel & split ----------------------------------------------------------

def test_build_all_features_covers_20_series_with_gt(weekly, gt, cfg):
    feats = build.build_all_features(weekly, gt, cfg)
    assert len(feats) == 20
    for df in feats.values():
        assert df["gt_index"].notna().all()


def test_temporal_split_is_147_37(weekly, cfg):
    grid = pd.DatetimeIndex(sorted(weekly["week_start"].unique()))
    cutoff = train_cutoff_week(grid, float(cfg["train_ratio"]))
    mask_train, mask_test = temporal_masks(pd.Series(grid), cutoff=cutoff)
    assert int(mask_train.sum()) == 147
    assert int(mask_test.sum()) == 37


def test_resolve_unknown_column_raises(one_series_feats, cfg):
    bad = {"feature_variants": {"x": ["does_not_exist"]}}

    class _Cfg:
        def __getitem__(self, k):
            return bad[k]

    with pytest.raises(KeyError):
        build.resolve_feature_columns(one_series_feats.columns, _Cfg(), "x")
