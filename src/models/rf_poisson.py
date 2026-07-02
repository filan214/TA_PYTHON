"""Tahap 6e-1 — Model pohon dengan objective Poisson (D11, prioritas 1).

Model existing (RF/HGB default) meminimalkan *squared error* -> asumsi implisit data
kontinu ber-noise Gaussian. Data ini adalah *count data* diskrit λ≈1,9 dgn varians≈mean
(karakteristik Poisson) & ~26% minggu nol. Menyelaraskan loss dengan distribusi data:
    - `RandomForestRegressor(criterion='poisson')`
    - `HistGradientBoostingRegressor(loss='poisson')`
adalah perbaikan berbasis teori terkuat dgn biaya terkecil — feature matrix Tahap 5
dipakai APA ADANYA (kedua varian baseline/gt), grid tuning ringan dgn TimeSeriesSplit(5)
yang sama seperti 6b (anti-leakage, train saja).

**Catatan target = MAE/MASE, BUKAN sMAPE.** sMAPE tak akan turun ke <15% (lantai
struktural ~65–70% untuk data ini, lih. D7/D10) — itu ekspektasi, bukan kegagalan.

**Rezim evaluasi (D9):** sama seperti RF — one-step-ahead alami (fitur lag/rolling dari
aktual hingga t-1). Prediksi keluar sebagai *rate* kontinu positif; **jangan dibulatkan**
sebelum evaluasi (agar sepadan dgn model lain). Clip >= 0 saja.

Output (masuk kelompok DM ke-4 Tahap 7, D11):
    reports/results/predictions_rf_poisson_<variant>.parquet
    reports/results/predictions_hgb_poisson_<variant>.parquet
    reports/results/poisson_best_params.csv
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src.features.build import make_supervised
from src.models.random_forest import split_supervised
from src.models.sarimax import _load_series_features
from src.utils.splits import train_cutoff_week

logger = logging.getLogger(__name__)

KINDS = ("rf_poisson", "hgb_poisson")


def _estimator_and_grid(kind: str, cfg, seed: int):
    """(estimator, param_grid) untuk GridSearchCV — objective Poisson."""
    if kind == "rf_poisson":
        from sklearn.ensemble import RandomForestRegressor

        rf = cfg["random_forest"]
        est = RandomForestRegressor(criterion="poisson", random_state=seed, n_jobs=-1)
        grid = {"n_estimators": rf["n_estimators"], "max_depth": rf["max_depth"],
                "max_features": rf["max_features"], "min_samples_leaf": rf["min_samples_leaf"]}
        return est, grid
    if kind == "hgb_poisson":
        from sklearn.ensemble import HistGradientBoostingRegressor

        est = HistGradientBoostingRegressor(loss="poisson", random_state=seed,
                                            early_stopping=False, max_iter=400)
        grid = {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [8, 15],
                "min_samples_leaf": [10, 20]}
        return est, grid
    raise ValueError(f"kind tak dikenal: {kind}")


def _fit_predict(kind: str, X_tr, y_tr, X_te, cfg, seed: int):
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

    est, grid = _estimator_and_grid(kind, cfg, seed)
    gs = GridSearchCV(est, grid, cv=TimeSeriesSplit(n_splits=5),
                      scoring="neg_mean_absolute_error", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    y_pred = np.clip(gs.best_estimator_.predict(X_te), 0.0, None)  # rate kontinu, tak dibulatkan
    return y_pred, gs.best_params_


def train_all(cfg, kind: str, variant: str, series: list[str] | None = None,
              save: bool = True):
    """Latih model pohon Poisson 20 deret satu (kind, varian) — one-step D9."""
    feats = _load_series_features(cfg)
    if series is not None:
        feats = {k: v for k, v in feats.items() if k in series}
    grid_weeks = pd.DatetimeIndex(sorted(next(iter(feats.values()))["week_start"].unique()))
    cutoff = train_cutoff_week(grid_weeks, float(cfg["train_ratio"]))
    seed = int(cfg.seed)

    pred_rows, param_rows = [], []
    for key, df in feats.items():
        store, brand = key.split("|")
        X, y, index = make_supervised(df, cfg, variant)
        X_tr, y_tr, X_te, y_te, test_index = split_supervised(X, y, index, cutoff)
        y_pred, best = _fit_predict(kind, X_tr, y_tr, X_te, cfg, seed)
        pred_rows.append(pd.DataFrame({
            "store": store, "brand": brand, "week_start": test_index,
            "y_true": y_te.to_numpy(), "y_pred": y_pred,
        }))
        param_rows.append({"kind": kind, "store": store, "brand": brand,
                           "variant": variant, **best})
        logger.info("%s[%s] %s: best=%s", kind, variant, key, best)

    predictions = pd.concat(pred_rows, ignore_index=True)
    params = pd.DataFrame(param_rows)
    if save:
        cfg.paths.results.mkdir(parents=True, exist_ok=True)
        out = cfg.paths.results / f"predictions_{kind}_{variant}.parquet"
        predictions.to_parquet(out, index=False)
        logger.info("Tersimpan prediksi %s (one-step D9): %s (%d baris)", kind, out.name, len(predictions))
    return predictions, params


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Tahap 6e-1 — RF/HGB objective Poisson (D11)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--kind", choices=[*KINDS, "both"], default="both")
    ap.add_argument("--variant", choices=["baseline", "gt", "both"], default="both")
    args = ap.parse_args(argv)

    from src.config import load_config
    from src.utils.io import set_seed

    cfg = load_config(args.config)
    cfg.paths.ensure()
    set_seed(cfg.seed)
    kinds = list(KINDS) if args.kind == "both" else [args.kind]
    variants = ["baseline", "gt"] if args.variant == "both" else [args.variant]

    all_params = []
    for kind in kinds:
        for variant in variants:
            _, params = train_all(cfg, kind, variant)
            all_params.append(params)
    params_df = pd.concat(all_params, ignore_index=True)
    out = cfg.paths.results / "poisson_best_params.csv"
    params_df.to_csv(out, index=False)
    print("Stage 6e-1 (Poisson):", {"kinds": kinds, "variants": variants,
                                     "n_models": len(params_df), "params_csv": str(out)})


if __name__ == "__main__":
    main()
