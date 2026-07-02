"""Tahap 6b — Random Forest per deret & per varian (Bab III §3.1.6, D5/D9).

RandomForestRegressor pada matriks supervised Tahap 5 (lag+rolling+kalender+Fourier
[+gt_index]). Orde hyperparameter dipilih via GridSearchCV + TimeSeriesSplit(5) pada
data latih saja (anti-leakage). Dua varian (ablation GT, D5): baseline vs gt.

**Rezim evaluasi (D9):** RF **alami one-step-ahead** — fitur lag/rolling tiap minggu uji
dihitung dari permintaan aktual hingga t-1 (make_supervised). Jadi memprediksi 37 baris
uji sekaligus = 37 ramalan one-step-ahead (BUKAN rekursif/multi-step). Origin ramalan
sepadan dengan SARIMAX walk-forward -> valid untuk uji DM (Tahap 7).

Output:
    reports/results/predictions_rf_<variant>.parquet   [store,brand,week_start,y_true,y_pred]
    reports/results/rf_best_params.csv                 (hyperparameter terpilih per deret×varian)
    reports/figures/rf_importance_<store>_<brand>_<variant>.png
    models/random_forest/<variant>/<store>_<brand>.pkl

DoD: 20 artefak per varian (40 total) tanpa error; prediksi 37 minggu per deret×varian.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features.build import make_supervised  # noqa: E402
from src.models.base import Forecaster  # noqa: E402
from src.utils.splits import temporal_masks, train_cutoff_week  # noqa: E402

logger = logging.getLogger(__name__)


def _param_grid(cfg) -> dict:
    rf = cfg["random_forest"]
    return {
        "n_estimators": rf["n_estimators"],
        "max_depth": rf["max_depth"],          # null yaml -> None
        "max_features": rf["max_features"],
        "min_samples_leaf": rf["min_samples_leaf"],
    }


def split_supervised(X, y, index, cutoff):
    """Bagi matriks supervised jadi train/test lewat ambang waktu global (D3)."""
    mask_train, mask_test = temporal_masks(pd.Series(index), cutoff=cutoff)
    return (X.loc[mask_train.values], y.loc[mask_train.values],
            X.loc[mask_test.values], y.loc[mask_test.values],
            pd.DatetimeIndex(index[mask_test.values]))


class RandomForestForecaster(Forecaster):
    """RF + GridSearchCV(TimeSeriesSplit) untuk pemilihan hyperparameter (train saja)."""

    def __init__(self, param_grid: dict, cv_splits: int = 5, random_state: int = 42,
                 scoring: str = "neg_mean_absolute_error", n_jobs: int = -1):
        self.param_grid = param_grid
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.model_ = None
        self.best_params_ = None
        self.feature_names_ = None

    def fit(self, X, y=None) -> "RandomForestForecaster":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

        self.feature_names_ = list(X.columns)
        gs = GridSearchCV(
            RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs),
            self.param_grid, cv=TimeSeriesSplit(n_splits=self.cv_splits),
            scoring=self.scoring, n_jobs=self.n_jobs,
        )
        gs.fit(X, y)
        self.model_ = gs.best_estimator_
        self.best_params_ = gs.best_params_
        return self

    def predict(self, X, exog_future=None) -> np.ndarray:
        return np.clip(self.model_.predict(X), 0.0, None)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model_.feature_importances_

    def name(self) -> str:
        return "RandomForest"


def _plot_importance(fc: RandomForestForecaster, key: str, variant: str, out_path: Path):
    imp = pd.Series(fc.feature_importances_, index=fc.feature_names_).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(imp))))
    colors = ["#C44E52" if n == "gt_index" else "#4C72B0" for n in imp.index]
    ax.barh(imp.index, imp.values, color=colors)
    ax.set_title(f"RF feature importance — {key} [{variant}]")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def train_all(cfg, variant: str, series: list[str] | None = None,
              save: bool = True, param_grid: dict | None = None, **rf_kwargs):
    """Latih RF 20 deret satu varian (one-step-ahead D9); (predictions, best_params_df)."""
    from src.models.sarimax import _load_series_features  # reuse loader

    feats = _load_series_features(cfg)
    if series is not None:
        feats = {k: v for k, v in feats.items() if k in series}
    grid = pd.DatetimeIndex(sorted(next(iter(feats.values()))["week_start"].unique()))
    cutoff = train_cutoff_week(grid, float(cfg["train_ratio"]))
    param_grid = param_grid if param_grid is not None else _param_grid(cfg)

    pred_rows, param_rows = [], []
    for key, df in feats.items():
        store, brand = key.split("|")
        X, y, index = make_supervised(df, cfg, variant)
        X_tr, y_tr, X_te, y_te, test_index = split_supervised(X, y, index, cutoff)

        fc = RandomForestForecaster(param_grid, random_state=int(cfg.seed), **rf_kwargs)
        fc.fit(X_tr, y_tr)
        y_pred = fc.predict(X_te)

        pred_rows.append(pd.DataFrame({
            "store": store, "brand": brand, "week_start": test_index,
            "y_true": y_te.to_numpy(), "y_pred": y_pred,
        }))
        gt_imp = (float(pd.Series(fc.feature_importances_, index=fc.feature_names_)["gt_index"])
                  if "gt_index" in fc.feature_names_ else np.nan)
        param_rows.append({"store": store, "brand": brand, "variant": variant,
                           **fc.best_params_, "gt_importance": gt_imp})

        if save:
            import joblib

            mdir = cfg.paths.models / "random_forest" / variant
            mdir.mkdir(parents=True, exist_ok=True)
            joblib.dump(fc.model_, mdir / f"{store}_{brand}.pkl")
            cfg.paths.figures.mkdir(parents=True, exist_ok=True)
            _plot_importance(fc, key, variant,
                             cfg.paths.figures / f"rf_importance_{store}_{brand}_{variant}.png")
        logger.info("RF[%s] %s: best=%s", variant, key, fc.best_params_)

    predictions = pd.concat(pred_rows, ignore_index=True)
    params = pd.DataFrame(param_rows)
    if save:
        cfg.paths.results.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(cfg.paths.results / f"predictions_rf_{variant}.parquet", index=False)
        logger.info("Tersimpan prediksi RF (one-step D9): predictions_rf_%s.parquet (%d baris)",
                    variant, len(predictions))
    return predictions, params


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="Tahap 6b — Random Forest (baseline & gt)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--variant", choices=["baseline", "gt", "both"], default="both")
    args = ap.parse_args(argv)

    from src.config import load_config
    from src.utils.io import set_seed

    cfg = load_config(args.config)
    cfg.paths.ensure()
    set_seed(cfg.seed)
    variants = ["baseline", "gt"] if args.variant == "both" else [args.variant]

    all_params = []
    for variant in variants:
        _, params = train_all(cfg, variant)
        all_params.append(params)
    params_df = pd.concat(all_params, ignore_index=True)
    out = cfg.paths.results / "rf_best_params.csv"
    params_df.to_csv(out, index=False)
    print("Stage 6b:", {"variants": variants, "n_models": len(params_df),
                        "params_csv": str(out)})


if __name__ == "__main__":
    main()
