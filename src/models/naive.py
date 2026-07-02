"""Tahap 6d — Baseline naif (D10, backfill).

Dua metode baseline, TANPA training/tuning, dihitung langsung dari data aktual:
    - `naive`  : ramalan minggu t = aktual minggu t-1 (random walk).
    - `snaive` : ramalan minggu t = aktual minggu t-52 (seasonal-naive m=52);
                 fallback ke naive biasa bila t-52 belum tersedia (awal deret).

**Rezim evaluasi (D9) — sama persis dengan SARIMAX/RF/LSTM:** one-step-ahead
walk-forward pada 37 minggu uji. Di tiap minggu t model "tahu" aktual hingga t-1,
jadi lag-1 dan lag-52 diambil dari deret aktual penuh (termasuk minggu uji yang
sudah terlewati). Origin ramalan sepadan dengan model utama -> galat naif bisa
dibandingkan lewat uji DM (Tahap 7) & dipakai sebagai denominator MASE (D10/9b).

**Bukan kandidat "algoritma terbaik"** (D10) — perannya murni pembanding/denominator,
tidak masuk ranking utama Tahap 7. Satu varian saja (naif tak memakai eksogen).

Output:
    reports/results/predictions_naive.parquet   [store,brand,week_start,y_true,y_pred]
    reports/results/predictions_snaive.parquet  (seasonal-naive m=52)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.splits import train_cutoff_week

logger = logging.getLogger(__name__)

NAIVE_METHODS = ("naive", "snaive")


def one_step_naive(units, seasonal: bool, m: int = 52) -> np.ndarray:
    """Ramalan one-step untuk SETIAP posisi t dari deret aktual penuh (NaN bila tak ada riwayat).

    seasonal=False -> lag-1 (random walk). seasonal=True -> lag-m, fallback lag-1.
    """
    u = np.asarray(units, dtype=float)
    fc = np.full(len(u), np.nan)
    for t in range(len(u)):
        if seasonal and t - m >= 0:
            fc[t] = u[t - m]
        elif t - 1 >= 0:
            fc[t] = u[t - 1]
    return np.clip(fc, 0.0, None)


def _load_weekly(cfg) -> pd.DataFrame:
    path = cfg.paths.interim / "weekly_store_brand.parquet"
    if not path.exists():
        from src.data.aggregate import _load_clean, aggregate_weekly

        aggregate_weekly(_load_clean(cfg), path)
    return pd.read_parquet(path)


def build_naive_predictions(cfg, method: str) -> pd.DataFrame:
    """Prediksi walk-forward (D9) untuk metode naif pada 37 minggu uji, 20 deret."""
    if method not in NAIVE_METHODS:
        raise ValueError(f"metode naif tak dikenal: {method}")
    seasonal = method == "snaive"
    m = int(cfg["seasonal_period"])

    weekly = _load_weekly(cfg)
    grid = pd.DatetimeIndex(sorted(weekly["week_start"].unique()))
    cutoff = train_cutoff_week(grid, float(cfg["train_ratio"]))

    rows = []
    for (store, brand), sub in weekly.groupby(["store", "brand"]):
        sub = sub.sort_values("week_start").reset_index(drop=True)
        fc = one_step_naive(sub["units"].to_numpy(), seasonal=seasonal, m=m)
        test = sub["week_start"] >= cutoff
        rows.append(pd.DataFrame({
            "store": store, "brand": brand,
            "week_start": sub.loc[test, "week_start"].to_numpy(),
            "y_true": sub.loc[test, "units"].astype(float).to_numpy(),
            "y_pred": fc[test.to_numpy()],
        }))
    return pd.concat(rows, ignore_index=True)


def train_all(cfg, save: bool = True) -> dict[str, pd.DataFrame]:
    """Hasilkan predictions_naive.parquet & predictions_snaive.parquet (D9)."""
    out = {}
    for method in NAIVE_METHODS:
        preds = build_naive_predictions(cfg, method)
        out[method] = preds
        if save:
            cfg.paths.results.mkdir(parents=True, exist_ok=True)
            path = cfg.paths.results / f"predictions_{method}.parquet"
            preds.to_parquet(path, index=False)
            logger.info("Tersimpan baseline %s (walk-forward D9): %s (%d baris)",
                        method, path.name, len(preds))
    return out


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Tahap 6d — baseline naif/seasonal-naif (D10)")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)

    from src.config import load_config

    cfg = load_config(args.config)
    cfg.paths.ensure()
    out = train_all(cfg)
    print("Stage 6d (naive, D10):", {
        m: {"rows": len(df), "n_series": df.groupby(["store", "brand"]).ngroups,
            "MAE": round(float(np.mean(np.abs(df["y_true"] - df["y_pred"]))), 4)}
        for m, df in out.items()
    })


if __name__ == "__main__":
    main()
