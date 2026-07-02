"""Tahap 6b DoD — Random Forest per deret & varian (§5, §7).

Uji cepat: grid hyperparameter kecil + subset deret. Verifikasi one-step-ahead
(origin uji sepadan SARIMAX -> D9), split 139/37, importance mengandung gt_index
di varian gt, dan skema prediksi.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.aggregate import aggregate_weekly
from src.data.clean import clean_transactions
from src.features import build
from src.models import random_forest as rf
from src.utils.splits import train_cutoff_week

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pos_transactions_raw.csv"

SMALL_GRID = {"n_estimators": [50], "max_depth": [None],
              "max_features": ["sqrt"], "min_samples_leaf": [1]}


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


def test_split_139_37_and_test_is_last37(feats, cfg, cutoff):
    X, y, index = build.make_supervised(feats["Toko_A|Xiaomi"], cfg, "baseline")
    X_tr, y_tr, X_te, y_te, test_index = rf.split_supervised(X, y, index, cutoff)
    assert len(X_tr) == 139 and len(X_te) == 37       # 176 supervised - 37 uji
    grid = pd.DatetimeIndex(sorted(feats["Toko_A|Xiaomi"]["week_start"].unique()))
    assert list(test_index) == list(grid[-37:])       # origin sepadan SARIMAX (D9)


def test_fit_predict_shapes(feats, cfg, cutoff):
    X, y, index = build.make_supervised(feats["Toko_A|Xiaomi"], cfg, "baseline")
    X_tr, y_tr, X_te, y_te, _ = rf.split_supervised(X, y, index, cutoff)
    fc = rf.RandomForestForecaster(SMALL_GRID, cv_splits=3, random_state=42).fit(X_tr, y_tr)
    y_pred = fc.predict(X_te)
    assert len(y_pred) == 37 and np.isfinite(y_pred).all() and (y_pred >= 0).all()
    assert fc.best_params_ is not None
    assert len(fc.feature_importances_) == X_tr.shape[1]


def test_gt_variant_importance_has_gt_index(feats, cfg, cutoff):
    X, y, index = build.make_supervised(feats["Toko_A|Xiaomi"], cfg, "gt")
    X_tr, y_tr, X_te, y_te, _ = rf.split_supervised(X, y, index, cutoff)
    fc = rf.RandomForestForecaster(SMALL_GRID, cv_splits=3, random_state=42).fit(X_tr, y_tr)
    imp = pd.Series(fc.feature_importances_, index=fc.feature_names_)
    assert "gt_index" in imp.index
    assert (imp >= 0).all()


def test_reproducible(feats, cfg, cutoff):
    X, y, index = build.make_supervised(feats["Toko_A|Xiaomi"], cfg, "baseline")
    X_tr, y_tr, X_te, y_te, _ = rf.split_supervised(X, y, index, cutoff)
    p1 = rf.RandomForestForecaster(SMALL_GRID, cv_splits=3, random_state=42).fit(X_tr, y_tr).predict(X_te)
    p2 = rf.RandomForestForecaster(SMALL_GRID, cv_splits=3, random_state=42).fit(X_tr, y_tr).predict(X_te)
    assert np.allclose(p1, p2)


def test_train_all_schema_subset(monkeypatch, feats, cfg):
    subset = {k: feats[k] for k in ["Toko_A|Xiaomi", "Toko_B|OPPO"]}
    monkeypatch.setattr("src.models.sarimax._load_series_features", lambda cfg: subset)
    preds, params = rf.train_all(cfg, "gt", save=False, param_grid=SMALL_GRID, cv_splits=3)

    assert list(preds.columns) == ["store", "brand", "week_start", "y_true", "y_pred"]
    assert len(preds) == 2 * 37 and (preds["y_pred"] >= 0).all()
    assert {"store", "brand", "variant", "n_estimators", "gt_importance"} <= set(params.columns)
    assert len(params) == 2
