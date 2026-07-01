"""Tahap 5 — Rekayasa fitur supervised + 2 varian (Bab III §3.1.5, D5 & D6).

Ubah tiap deret mingguan gerai×merek menjadi matriks supervised untuk RF & LSTM,
dan siapkan `exog` untuk SARIMAX. Dua **varian fitur** per deret:
    - `baseline` : tanpa Google Trends
    - `gt`       : + kolom gt_index (ablation study D5)
Fourier(m=52) ada di KEDUA varian (D6) — bukan bagian ablation GT.

Output: data/processed/features_<store>_<brand>.parquet
    kolom: store, brand, week_start, units (target), + fitur, + gt_index.
    (baris warm-up ber-NaN lag TETAP disimpan di parquet; di-drop saat make_supervised.)

Anti-leakage (WAJIB):
    - Semua lag/rolling dihitung dari masa lalu saja: rolling di-`shift(1)` agar
      jendela berakhir di t-1 (tidak menyentuh units[t] = target).
    - Scaler MinMax di-`fit` HANYA pada partisi latih (147 minggu pertama, D3),
      terpisah per varian.

DoD (§5 Tahap 5):
    - Tak ada baris lag-NaN yang bocor ke train/test (di-drop konsisten kedua varian).
    - `make_supervised(df, cfg, variant) -> (X, y, index)`; kolom gt_index ADA di
      varian `gt`, TIDAK ADA di `baseline`.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.splits import temporal_masks

logger = logging.getLogger(__name__)

# Idulfitri (Lebaran) Indonesia — referensi fitur kalender (bukan pemodelan ARIMA).
LEBARAN_DATES = pd.to_datetime(["2022-05-02", "2023-04-22", "2024-04-10", "2025-03-31"])
RAMADAN_WEEKS_BEFORE = 4          # ~4 minggu menjelang Lebaran = bulan Ramadan
LEBARAN_WINDOW_WEEKS = 1          # ±1 minggu di sekitar minggu Lebaran
HARBOLNAS_DATES_MMDD = [(11, 11), (12, 12)]  # 11.11 & 12.12


def _week_monday(ts: pd.Timestamp) -> pd.Timestamp:
    """Senin awal minggu (konvensi sama dengan Tahap 1/2)."""
    return pd.Timestamp(ts).to_period("W").start_time


# --- Blok fitur -------------------------------------------------------------

def add_lag_roll_features(df: pd.DataFrame, lags: int, rolling_windows: list[int]) -> pd.DataFrame:
    """lag_1..lag_L dan roll_mean_w (semua w) + roll_std_w (w pertama), past-only."""
    u = df["units"]
    for k in range(1, lags + 1):
        df[f"lag_{k}"] = u.shift(k)
    for w in rolling_windows:
        df[f"roll_mean_{w}"] = u.rolling(w).mean().shift(1)   # jendela berakhir di t-1
    df[f"roll_std_{rolling_windows[0]}"] = u.rolling(rolling_windows[0]).std().shift(1)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """weekofyear, month, is_ramadan, is_lebaran_window, is_harbolnas, year_trend."""
    ws = df["week_start"]
    df["weekofyear"] = ws.dt.isocalendar().week.astype(int)
    df["month"] = ws.dt.month.astype(int)

    lebaran_mondays = [_week_monday(d) for d in LEBARAN_DATES]
    lebaran_window, ramadan = set(), set()
    for lm in lebaran_mondays:
        for j in range(-LEBARAN_WINDOW_WEEKS, LEBARAN_WINDOW_WEEKS + 1):
            lebaran_window.add(lm + pd.Timedelta(weeks=j))
        for j in range(1, RAMADAN_WEEKS_BEFORE + 1):
            ramadan.add(lm - pd.Timedelta(weeks=j))

    years = range(int(ws.dt.year.min()), int(ws.dt.year.max()) + 1)
    harbolnas = {
        _week_monday(pd.Timestamp(year=y, month=m, day=d))
        for y in years for (m, d) in HARBOLNAS_DATES_MMDD
    }

    df["is_ramadan"] = ws.isin(ramadan).astype(int)
    df["is_lebaran_window"] = ws.isin(lebaran_window).astype(int)
    df["is_harbolnas"] = ws.isin(harbolnas).astype(int)
    df["year_trend"] = np.arange(len(df))          # indeks minggu berurutan (tren)
    return df


def add_fourier_features(df: pd.DataFrame, period: int, harmonics: int) -> pd.DataFrame:
    """Pasangan sin/cos Fourier musiman (m=period) dari indeks minggu berurutan."""
    t = np.arange(len(df))
    for k in range(1, harmonics + 1):
        df[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        df[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return df


# --- Perakitan per deret ----------------------------------------------------

def build_features_for_series(series_df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Rakit seluruh fitur untuk satu deret (sudah terurut week_start, berisi gt_index)."""
    df = series_df.sort_values("week_start").reset_index(drop=True).copy()
    df = add_lag_roll_features(df, int(cfg["lags"]), list(cfg["rolling_windows"]))
    df = add_calendar_features(df)
    df = add_fourier_features(df, int(cfg["seasonal_period"]), int(cfg["fourier_harmonics"]))
    return df


def build_all_features(
    weekly: pd.DataFrame, gt: pd.DataFrame, cfg, processed_dir: Path | None = None
) -> dict[str, pd.DataFrame]:
    """Bangun fitur untuk 20 deret; opsional tulis parquet per deret."""
    merged = weekly.merge(gt[["week_start", "brand", "gt_index"]],
                          on=["brand", "week_start"], how="left")
    out: dict[str, pd.DataFrame] = {}
    for (store, brand), sub in merged.groupby(["store", "brand"]):
        feats = build_features_for_series(sub, cfg)
        key = f"{store}|{brand}"
        out[key] = feats
        if processed_dir is not None:
            processed_dir.mkdir(parents=True, exist_ok=True)
            feats.to_parquet(processed_dir / f"features_{store}_{brand}.parquet", index=False)
    logger.info("Fitur terbangun untuk %d deret.", len(out))
    return out


# --- Pemilihan varian & supervised -----------------------------------------

def resolve_feature_columns(all_cols, cfg, variant: str) -> list[str]:
    """Resolusi daftar kolom varian dari config; '*' = cocok-awalan; jaga urutan df."""
    spec = cfg["feature_variants"][variant]
    resolved: list[str] = []
    for entry in spec:
        if entry.endswith("*"):
            prefix = entry[:-1]
            resolved.extend([c for c in all_cols if c.startswith(prefix)])
        elif entry in all_cols:
            resolved.append(entry)
        else:
            raise KeyError(f"Kolom fitur '{entry}' (varian {variant}) tak ada di features.")
    seen, ordered = set(), []
    for c in resolved:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def make_supervised(
    features_df: pd.DataFrame, cfg, variant: str = "baseline"
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Matriks supervised (X, y, index) untuk satu deret & varian.

    Drop baris warm-up (lag/rolling NaN) — konsisten antar-varian (lag identik).
    """
    cols = resolve_feature_columns(features_df.columns, cfg, variant)
    sub = features_df.dropna(subset=cols).reset_index(drop=True)
    X = sub[cols].copy()
    y = sub["units"].astype(float).copy()
    index = pd.DatetimeIndex(sub["week_start"].values)
    return X, y, index


def fit_scaler_on_train(
    X: pd.DataFrame, index: pd.DatetimeIndex, cutoff: pd.Timestamp
):
    """Fit MinMaxScaler HANYA pada partisi latih (minggu < cutoff, D3) — anti-leakage.

    `cutoff` dihitung dari grid PENUH 184 minggu supaya uji tetap 37 minggu terakhir.
    """
    from sklearn.preprocessing import MinMaxScaler

    mask_train, _ = temporal_masks(pd.Series(index), cutoff=cutoff)
    scaler = MinMaxScaler()
    scaler.fit(X.loc[mask_train.values])
    return scaler


# --- Orkestrasi -------------------------------------------------------------

def _load_inputs(cfg):
    weekly_path = cfg.paths.interim / "weekly_store_brand.parquet"
    if not weekly_path.exists():
        from src.data.aggregate import _load_clean, aggregate_weekly

        aggregate_weekly(_load_clean(cfg), weekly_path)
    weekly = pd.read_parquet(weekly_path)

    # Tahap 5 memakai cache GT apa adanya (TANPA fetch ulang; §15). build_google_trends
    # akan memuat cache bila ada (source=cache) tanpa network.
    from src.data.google_trends import build_google_trends

    gt_path = cfg.paths.interim / "google_trends.csv"
    week_grid = pd.DatetimeIndex(sorted(weekly["week_start"].unique()))
    gt = build_google_trends(
        keywords_by_brand=cfg["keywords_by_brand"], week_grid=week_grid,
        timeframe=cfg["trends_timeframe"], geo=cfg["trends_geo"], out_path=gt_path,
    )
    return weekly, gt


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="Tahap 5 — rekayasa fitur (baseline & gt)")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)

    from src.config import load_config

    cfg = load_config(args.config)
    cfg.paths.ensure()
    weekly, gt = _load_inputs(cfg)
    feats = build_all_features(weekly, gt, cfg, processed_dir=cfg.paths.processed)

    # Ambang split D3 dihitung sekali dari grid PENUH 184 minggu (uji = 37 terakhir).
    from src.utils.splits import train_cutoff_week

    week_grid = pd.DatetimeIndex(sorted(weekly["week_start"].unique()))
    cutoff = train_cutoff_week(week_grid, float(cfg["train_ratio"]))

    # Simpan scaler per varian per deret (fit di train saja).
    import joblib

    n_scalers = 0
    for key, df in feats.items():
        store, brand = key.split("|")
        for variant in ("baseline", "gt"):
            X, _, index = make_supervised(df, cfg, variant)
            scaler = fit_scaler_on_train(X, index, cutoff)
            sdir = cfg.paths.models / "scalers" / variant
            sdir.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, sdir / f"{store}_{brand}.pkl")
            n_scalers += 1

    # Ringkasan cepat satu deret.
    any_key = next(iter(feats))
    Xb, yb, idx = make_supervised(feats[any_key], cfg, "baseline")
    Xg, _, _ = make_supervised(feats[any_key], cfg, "gt")
    print("Stage 5:", {
        "n_series": len(feats),
        "n_scalers": n_scalers,
        "supervised_rows_per_series": len(Xb),
        "n_features_baseline": Xb.shape[1],
        "n_features_gt": Xg.shape[1],
        "gt_only_cols": sorted(set(Xg.columns) - set(Xb.columns)),
    })


if __name__ == "__main__":
    main()
