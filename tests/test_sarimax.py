"""Tahap 6a DoD — SARIMA/SARIMAX per deret & varian (§5, §7).

Uji cepat: seasonal=False (hindari m=52 yang lambat) + subset deret. Memverifikasi
pemilihan exog (tanpa lag/roll; gt_index hanya di gt), split 147/37, prediksi 37
non-negatif, skema orders (has_seasonal_terms), dan fallback non-seasonal.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.aggregate import aggregate_weekly
from src.data.clean import clean_transactions
from src.features import build
from src.models import sarimax as sx
from src.utils.splits import train_cutoff_week

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pos_transactions_raw.csv"


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


# --- Pemilihan eksogen ------------------------------------------------------

def test_exog_excludes_lag_roll(feats, cfg):
    cols = sx.sarimax_exog_columns(next(iter(feats.values())).columns, cfg, "baseline")
    assert not any(c.startswith(("lag_", "roll_")) for c in cols)
    assert "fourier_sin_1" in cols and "is_lebaran_window" in cols


def test_exog_gt_has_one_more_column(feats, cfg):
    cols_b = sx.sarimax_exog_columns(next(iter(feats.values())).columns, cfg, "baseline")
    cols_g = sx.sarimax_exog_columns(next(iter(feats.values())).columns, cfg, "gt")
    assert set(cols_g) - set(cols_b) == {"gt_index"}


def test_split_147_37(feats, cfg, cutoff):
    y_tr, y_te, X_tr, X_te, idx, _ = sx.split_endog_exog(
        feats["Toko_A|Xiaomi"], cfg, "baseline", cutoff
    )
    assert len(y_tr) == 147 and len(y_te) == 37
    assert len(X_tr) == 147 and len(X_te) == 37 and len(idx) == 37


# --- Fit / predict (cepat, non-seasonal) ------------------------------------

def test_fit_predict_nonseasonal(feats, cfg, cutoff):
    fc, y_te, y_pred, idx, _ = sx.fit_one(
        feats["Toko_A|Xiaomi"], cfg, "baseline", cutoff, seasonal=False
    )
    assert len(y_pred) == 37 == len(y_te)
    assert np.isfinite(y_pred).all()
    assert (y_pred >= 0).all()               # clip permintaan >= 0
    assert fc.has_seasonal_terms is False    # non-seasonal -> tak ada suku musiman


def test_predictions_clipped_nonnegative(feats, cfg, cutoff):
    _, _, y_pred, _, _ = sx.fit_one(
        feats["Toko_D|Vivo"], cfg, "gt", cutoff, seasonal=False
    )
    assert (y_pred >= 0).all()


# --- has_seasonal_terms property -------------------------------------------

class _Stub:
    def __init__(self, order, seasonal_order):
        self.order, self.seasonal_order = order, seasonal_order

    def predict(self, n_periods, X=None):
        return np.zeros(n_periods)

    def aic(self):
        return 100.0


def test_has_seasonal_terms_true_and_false():
    fc = sx.SarimaxForecaster()
    fc._sorder = (0, 0, 0, 52)
    assert fc.has_seasonal_terms is False
    fc._sorder = (1, 0, 0, 52)
    assert fc.has_seasonal_terms is True


# --- D9 walk-forward --------------------------------------------------------

def test_fit_fixed_and_walk_forward(feats, cfg, cutoff):
    y_tr, y_te, X_tr, X_te, idx, _ = sx.split_endog_exog(
        feats["Toko_A|Xiaomi"], cfg, "baseline", cutoff
    )
    fc = sx.SarimaxForecaster(m=52).fit_fixed(y_tr, X_tr, (1, 0, 1), (0, 0, 0, 52))
    wf = fc.walk_forward(y_te, X_te)
    assert len(wf) == 37 and np.isfinite(wf).all() and (wf >= 0).all()
    assert fc.order == (1, 0, 1) and fc.seasonal_order == (0, 0, 0, 52)


def test_walk_forward_is_rolling_not_multistep(feats, cfg, cutoff):
    # AR(1) punya dinamika -> one-step (pakai aktual t-1) BEDA dari fixed-origin 37-langkah.
    y_tr, y_te, X_tr, X_te, idx, _ = sx.split_endog_exog(
        feats["Toko_A|Xiaomi"], cfg, "gt", cutoff
    )
    fc = sx.SarimaxForecaster(m=52).fit_fixed(y_tr, X_tr, (1, 0, 0), (0, 0, 0, 52))
    wf = fc.walk_forward(y_te, X_te)
    res = fc._statsmodels_result()
    Xs = fc._prep_exog(X_te, fit=False)
    multistep = np.asarray(res.forecast(steps=37, exog=Xs), dtype=float)
    assert not np.allclose(wf, multistep)  # rezim rolling != fixed-origin (D9)


# --- Fallback non-seasonal --------------------------------------------------

def test_fit_falls_back_when_seasonal_fails(monkeypatch, feats, cfg, cutoff):
    def fake_auto_arima(y, seasonal=True, **k):
        if seasonal:
            raise RuntimeError("seasonal m=52 non-konvergen (simulasi)")
        return _Stub((1, 0, 0), (0, 0, 0, 0))

    monkeypatch.setattr("pmdarima.auto_arima", fake_auto_arima)
    y_tr, _, X_tr, X_te, _, _ = sx.split_endog_exog(
        feats["Toko_A|Xiaomi"], cfg, "baseline", cutoff
    )
    fc = sx.SarimaxForecaster(seasonal=True, m=52)
    fc.fit(y_tr, X_tr)
    assert fc.fell_back_ is True
    assert len(fc.predict(37, X_te)) == 37


# --- train_all schema (subset, non-seasonal) -------------------------------

def test_train_all_schema_subset(monkeypatch, feats, cfg):
    subset = {k: feats[k] for k in ["Toko_A|Xiaomi", "Toko_B|OPPO"]}
    monkeypatch.setattr(sx, "_load_series_features", lambda cfg: subset)
    preds, orders = sx.train_all(cfg, "baseline", save=False, seasonal=False)

    assert list(preds.columns) == ["store", "brand", "week_start", "y_true", "y_pred"]
    assert len(preds) == 2 * 37
    assert (preds["y_pred"] >= 0).all()

    need = {"store", "brand", "variant", "p", "d", "q", "P", "D", "Q", "m",
            "has_seasonal_terms", "fell_back", "aic"}
    assert need <= set(orders.columns)
    assert len(orders) == 2
    assert orders["variant"].eq("baseline").all()
