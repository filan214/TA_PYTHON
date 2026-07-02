"""Tahap 6e-2 — Croston & TSB untuk intermittent demand (D11, prioritas 2).

Metode klasik yang didesain khusus untuk *intermittent demand* (masih ada ~26% minggu
nol). Memisahkan estimasi "ukuran permintaan saat terjadi" dari "seberapa sering terjadi":
    - **Croston**: SES pada ukuran permintaan (z) & interval antar-permintaan (p),
      di-update HANYA saat ada permintaan. Ramalan rate = z / p.
    - **TSB** (Teunter-Syntetos-Babai): SES pada ukuran (z) & *probabilitas* permintaan
      (prob), di-update SETIAP periode (menangani obsolescence). Ramalan = prob * z.

Satu varian saja (tanpa GT — metode ini tak menerima eksogen). Smoothing α di-tune pada
data latih via grid kecil (0.05–0.5) dgn kriteria MAE one-step di train (anti-leakage).

**Rezim evaluasi (D9):** recursion berjalan atas deret aktual penuh; ramalan periode t
memakai state yang di-update hingga aktual t-1 -> one-step-ahead walk-forward, sepadan
dgn model lain. α dipilih dari train saja; ramalan test memakai aktual s/d t-1 (premis D9).

Output (kelompok DM ke-4 Tahap 7, D11):
    reports/results/predictions_croston.parquet
    reports/results/predictions_tsb.parquet
    reports/results/croston_params.csv   (α terpilih per deret × metode)
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src.utils.splits import train_cutoff_week

logger = logging.getLogger(__name__)

METHODS = ("croston", "tsb")
ALPHA_GRID = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)


def croston_forecast(y, alpha: float) -> np.ndarray:
    """One-step forecast Croston klasik untuk tiap posisi (rate = z/p, state s/d t-1)."""
    y = np.asarray(y, float)
    n = len(y)
    fc = np.zeros(n)
    z = p = None
    q = 0                                   # interval berjalan sejak permintaan terakhir
    for t in range(n):
        fc[t] = (z / p) if (z is not None and p and p > 0) else 0.0
        q += 1
        if y[t] > 0:
            if z is None:                   # inisialisasi pada permintaan pertama
                z, p = y[t], float(q)
            else:
                z = alpha * y[t] + (1 - alpha) * z
                p = alpha * q + (1 - alpha) * p
            q = 0
    return np.clip(fc, 0.0, None)


def tsb_forecast(y, alpha: float, beta: float | None = None) -> np.ndarray:
    """One-step forecast TSB (rate = prob*z). beta default = alpha (satu parameter di-tune)."""
    y = np.asarray(y, float)
    beta = alpha if beta is None else beta
    n = len(y)
    fc = np.zeros(n)
    z = None
    prob = 0.0
    for t in range(n):
        fc[t] = prob * z if z is not None else 0.0
        if y[t] > 0:
            z = y[t] if z is None else alpha * y[t] + (1 - alpha) * z
            prob = beta * 1.0 + (1 - beta) * prob
        else:
            prob = (1 - beta) * prob        # peluang meluruh saat tak ada permintaan
    return np.clip(fc, 0.0, None)


_FORECASTERS = {"croston": croston_forecast, "tsb": tsb_forecast}


def tune_alpha(y_full, n_train: int, method: str) -> float:
    """Pilih α (grid) yang meminimalkan MAE one-step di partisi latih (anti-leakage)."""
    fn = _FORECASTERS[method]
    best_alpha, best_mae = ALPHA_GRID[0], np.inf
    for a in ALPHA_GRID:
        fc = fn(y_full, a)
        # skip posisi 0 (belum ada state); nilai train = indeks 1..n_train-1
        err = np.abs(np.asarray(y_full, float)[1:n_train] - fc[1:n_train])
        m = float(np.mean(err)) if err.size else np.inf
        if m < best_mae:
            best_alpha, best_mae = a, m
    return best_alpha


def _load_weekly(cfg) -> pd.DataFrame:
    path = cfg.paths.interim / "weekly_store_brand.parquet"
    if not path.exists():
        from src.data.aggregate import _load_clean, aggregate_weekly

        aggregate_weekly(_load_clean(cfg), path)
    return pd.read_parquet(path)


def train_all(cfg, method: str, save: bool = True):
    """Croston/TSB 20 deret (one-step D9); α di-tune per deret di train."""
    if method not in METHODS:
        raise ValueError(f"metode tak dikenal: {method}")
    weekly = _load_weekly(cfg)
    grid = pd.DatetimeIndex(sorted(weekly["week_start"].unique()))
    cutoff = train_cutoff_week(grid, float(cfg["train_ratio"]))
    n_train = int((grid < cutoff).sum())
    fn = _FORECASTERS[method]

    pred_rows, param_rows = [], []
    for (store, brand), sub in weekly.groupby(["store", "brand"]):
        sub = sub.sort_values("week_start").reset_index(drop=True)
        y = sub["units"].to_numpy(float)
        alpha = tune_alpha(y, n_train, method)
        fc = fn(y, alpha)
        test = (sub["week_start"] >= cutoff).to_numpy()
        pred_rows.append(pd.DataFrame({
            "store": store, "brand": brand,
            "week_start": sub.loc[test, "week_start"].to_numpy(),
            "y_true": y[test], "y_pred": fc[test],
        }))
        param_rows.append({"method": method, "store": store, "brand": brand, "alpha": alpha})
        logger.info("%s %s|%s: alpha=%.2f", method, store, brand, alpha)

    predictions = pd.concat(pred_rows, ignore_index=True)
    params = pd.DataFrame(param_rows)
    if save:
        cfg.paths.results.mkdir(parents=True, exist_ok=True)
        out = cfg.paths.results / f"predictions_{method}.parquet"
        predictions.to_parquet(out, index=False)
        logger.info("Tersimpan prediksi %s (one-step D9): %s (%d baris)", method, out.name, len(predictions))
    return predictions, params


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Tahap 6e-2 — Croston & TSB (D11)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--method", choices=[*METHODS, "both"], default="both")
    args = ap.parse_args(argv)

    from src.config import load_config

    cfg = load_config(args.config)
    cfg.paths.ensure()
    methods = list(METHODS) if args.method == "both" else [args.method]

    all_params = []
    for method in methods:
        _, params = train_all(cfg, method)
        all_params.append(params)
    params_df = pd.concat(all_params, ignore_index=True)
    out = cfg.paths.results / "croston_params.csv"
    params_df.to_csv(out, index=False)
    print("Stage 6e-2 (Croston/TSB):", {"methods": methods, "n_models": len(params_df),
                                        "params_csv": str(out)})


if __name__ == "__main__":
    main()
