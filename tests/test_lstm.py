"""Tahap 6c DoD — LSTM per deret & varian (§5, §7).

Uji cepat: units kecil + epochs kecil + subset deret. Verifikasi konstruksi
kanal/sekuens, one-step-ahead (origin uji sepadan SARIMAX/RF -> D9), kanal
gt_index hanya di varian gt, dan skema prediksi.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.aggregate import aggregate_weekly
from src.data.clean import clean_transactions
from src.features import build
from src.models import lstm as L
from src.utils.splits import train_cutoff_week

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pos_transactions_raw.csv"

TINY = dict(units=8, epochs=3, patience=2)


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "config.yaml")


@pytest.fixture(scope="module")
def feats(cfg):
    weekly = aggregate_weekly(clean_transactions(RAW))
    brands = sorted(weekly["brand"].unique())
    grid = pd.DatetimeIndex(sorted(weekly["week_start"].unique()))
    gt = pd.concat(
        [pd.DataFrame({"week_start": grid, "brand": b,
                       "gt_index": np.linspace(10, 90, len(grid))}) for b in brands],
        ignore_index=True,
    )
    return build.build_all_features(weekly, gt, cfg)


@pytest.fixture(scope="module")
def cutoff(feats, cfg):
    grid = pd.DatetimeIndex(sorted(next(iter(feats.values()))["week_start"].unique()))
    return train_cutoff_week(grid, float(cfg["train_ratio"]))


# --- Kanal & sekuens --------------------------------------------------------

def test_channels_units_first_no_lag_roll(feats, cfg):
    ch = L.lstm_channels(next(iter(feats.values())).columns, cfg, "baseline")
    assert ch[0] == "units"
    assert not any(c.startswith(("lag_", "roll_")) for c in ch)


def test_gt_channel_only_in_gt(feats, cfg):
    cols = next(iter(feats.values())).columns
    cb = L.lstm_channels(cols, cfg, "baseline")
    cg = L.lstm_channels(cols, cfg, "gt")
    assert set(cg) - set(cb) == {"gt_index"}


def test_make_sequences_shapes():
    mat = np.arange(20 * 3, dtype=float).reshape(20, 3)
    y = np.arange(20, dtype=float)
    weeks = pd.date_range("2022-01-03", periods=20, freq="W-MON")
    X, ys, idx = L.make_sequences(mat, y, weeks, L=8)
    assert X.shape == (12, 8, 3) and len(ys) == 12 and len(idx) == 12
    assert np.allclose(X[0], mat[0:8]) and ys[0] == y[8]   # jendela [t-L..t-1] -> t


def test_build_sequences_split_and_d9(feats, cfg, cutoff):
    X_tr, y_tr, X_te, test_index, y_true, umin, urange = L.build_series_sequences(
        feats["Toko_A|Xiaomi"], cfg, "baseline", cutoff
    )
    assert X_tr.shape == (139, 8, 11) and X_te.shape == (37, 8, 11)
    assert len(y_true) == 37 and not np.isnan(y_true).any()
    grid = pd.DatetimeIndex(sorted(feats["Toko_A|Xiaomi"]["week_start"].unique()))
    assert list(test_index) == list(grid[-37:])            # one-step origin sepadan (D9)


def test_gt_variant_extra_channel(feats, cfg, cutoff):
    _, _, X_te_b, *_ = L.build_series_sequences(feats["Toko_A|Xiaomi"], cfg, "baseline", cutoff)
    _, _, X_te_g, *_ = L.build_series_sequences(feats["Toko_A|Xiaomi"], cfg, "gt", cutoff)
    assert X_te_g.shape[2] == X_te_b.shape[2] + 1           # +gt_index


# --- Fit/predict (tiny) -----------------------------------------------------

def test_fit_predict_tiny(feats, cfg, cutoff):
    X_tr, y_tr, X_te, test_index, y_true, umin, urange = L.build_series_sequences(
        feats["Toko_A|Xiaomi"], cfg, "gt", cutoff
    )
    fc = L.LSTMForecaster(seed=42, **TINY).fit(X_tr, y_tr)
    y_pred = np.clip(fc.predict(X_te) * urange + umin, 0.0, None)
    assert len(y_pred) == 37 and np.isfinite(y_pred).all() and (y_pred >= 0).all()


def test_train_all_schema_subset(monkeypatch, feats, cfg):
    subset = {k: feats[k] for k in ["Toko_A|Xiaomi", "Toko_D|Vivo"]}
    monkeypatch.setattr("src.models.sarimax._load_series_features", lambda cfg: subset)
    preds = L.train_all(cfg, "baseline", save=False, **TINY)
    assert list(preds.columns) == ["store", "brand", "week_start", "y_true", "y_pred"]
    assert len(preds) == 2 * 37 and (preds["y_pred"] >= 0).all()
    assert np.isfinite(preds["y_pred"]).all()
