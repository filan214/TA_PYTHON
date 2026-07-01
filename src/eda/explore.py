"""Tahap 4 — Analisis Data Eksploratif (Bab III §3.1.5).

Menghasilkan figure untuk skripsi + ringkasan yang memandu pemodelan SARIMAX/RF/LSTM.

Output:
    reports/figures/series_grid.png              (plot 4x5 deret store x brand)
    reports/figures/acf_pacf_<series>.png        (deret representatif)
    reports/figures/decompose_<series>.png       (deret terpadat, period=52)
    reports/figures/seasonality_month.png        (rata-rata unit per bulan)
    reports/results/eda_summary.json             (ADF per 20 deret + rekomendasi d/D)

DoD (§5 Tahap 4):
    - Semua figure ter-generate tanpa error.
    - eda_summary.json memuat p-value ADF per 20 deret + rekomendasi d/D awal.

Catatan:
- Backend matplotlib = 'Agg' (non-interaktif) -> aman untuk headless/CI.
- seasonal_decompose model='additive' (bukan multiplicative): deret memuat nol,
  multiplicative akan gagal/inf pada nilai 0.
- Tanggal Lebaran (Idulfitri) di-hardcode sebagai referensi anotasi musiman saja
  (BUKAN dipakai untuk pemodelan). Harbolnas ~ Nov/Des (11.11, 12.12).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # harus sebelum import pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # noqa: E402
from statsmodels.tsa.seasonal import seasonal_decompose  # noqa: E402
from statsmodels.tsa.stattools import adfuller  # noqa: E402

logger = logging.getLogger(__name__)

SEASONAL_PERIOD = 52          # D6 (mingguan tahunan)
ADF_ALPHA = 0.05              # ambang stasioneritas
SEASONAL_STRENGTH_THRESH = 0.64  # heuristik Hyndman: Fs>=0.64 -> perlu D=1
MAX_D = 2

# Idulfitri (Lebaran) di Indonesia — referensi anotasi saja.
LEBARAN_DATES = ["2022-05-02", "2023-04-22", "2024-04-10", "2025-03-31"]


# --- Pemuatan & bentuk deret ------------------------------------------------

def load_weekly(path: str | Path) -> pd.DataFrame:
    """Muat panel mingguan Tahap 2 (long: store, brand, week_start, units)."""
    return pd.read_parquet(path)


def series_key(store: str, brand: str) -> str:
    return f"{store}|{brand}"


def to_series(weekly: pd.DataFrame, store: str, brand: str) -> pd.Series:
    """Ekstrak satu deret waktu (index = week_start Senin, values = units)."""
    sub = weekly[(weekly["store"] == store) & (weekly["brand"] == brand)]
    return sub.set_index("week_start")["units"].sort_index().astype(float)


def series_list(weekly: pd.DataFrame) -> list[tuple[str, str]]:
    return sorted(weekly.groupby(["store", "brand"]).groups.keys())


# --- ADF & rekomendasi orde -------------------------------------------------

def adf_pvalue(series: pd.Series) -> float | None:
    """p-value ADF (autolag=AIC); None bila deret konstan/gagal."""
    s = series.dropna()
    if s.nunique() <= 1 or len(s) < 12:
        return None
    try:
        return float(adfuller(s, autolag="AIC")[1])
    except Exception as exc:  # noqa: BLE001
        logger.warning("ADF gagal: %s", exc)
        return None


def seasonal_strength(series: pd.Series, period: int = SEASONAL_PERIOD) -> float | None:
    """Kekuatan musiman Fs = max(0, 1 - Var(resid)/Var(resid+seasonal)) (Hyndman)."""
    s = series.dropna()
    if len(s) < 2 * period or s.nunique() <= 1:
        return None
    try:
        dec = seasonal_decompose(s, model="additive", period=period)
        resid = dec.resid.dropna()
        seas = dec.seasonal.reindex(resid.index)
        denom = float(np.var(resid + seas))
        if denom <= 0:
            return 0.0
        return float(max(0.0, 1.0 - np.var(resid) / denom))
    except Exception as exc:  # noqa: BLE001
        logger.warning("seasonal_decompose gagal: %s", exc)
        return None


def recommend_orders(series: pd.Series, period: int = SEASONAL_PERIOD) -> dict:
    """Rekomendasi d (dari ADF berjenjang) & D (dari kekuatan musiman)."""
    p_level = adf_pvalue(series)
    # d: bedakan sampai stasioner, maksimal MAX_D.
    d = 0
    if p_level is not None and p_level >= ADF_ALPHA:
        p1 = adf_pvalue(series.diff().dropna())
        d = 1 if (p1 is not None and p1 < ADF_ALPHA) else MAX_D
    fs = seasonal_strength(series, period)
    D = 1 if (fs is not None and fs >= SEASONAL_STRENGTH_THRESH) else 0
    return {
        "p_value_level": None if p_level is None else round(p_level, 4),
        "stationary_level": None if p_level is None else bool(p_level < ADF_ALPHA),
        "seasonal_strength": None if fs is None else round(fs, 4),
        "recommend_d": d,
        "recommend_D": D,
    }


def run_adf_all(weekly: pd.DataFrame) -> dict[str, dict]:
    """ADF + rekomendasi orde untuk seluruh 20 deret."""
    out: dict[str, dict] = {}
    for store, brand in series_list(weekly):
        out[series_key(store, brand)] = recommend_orders(to_series(weekly, store, brand))
    return out


# --- Figure -----------------------------------------------------------------

def plot_series_grid(weekly: pd.DataFrame, out_path: Path) -> Path:
    """Grid 4x5 (store x brand) deret waktu units mingguan."""
    stores = sorted(weekly["store"].unique())
    brands = sorted(weekly["brand"].unique())
    fig, axes = plt.subplots(len(stores), len(brands), figsize=(20, 12),
                             sharex=True, squeeze=False)
    for i, store in enumerate(stores):
        for j, brand in enumerate(brands):
            ax = axes[i][j]
            s = to_series(weekly, store, brand)
            ax.plot(s.index, s.values, lw=0.8)
            if i == 0:
                ax.set_title(brand, fontsize=11)
            if j == 0:
                ax.set_ylabel(store, fontsize=11)
    fig.suptitle("Deret mingguan units — gerai × merek (4×5)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_acf_pacf(series: pd.Series, name: str, out_path: Path, lags: int = 60) -> Path:
    """ACF & PACF berdampingan untuk satu deret."""
    lags = min(lags, len(series.dropna()) // 2 - 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(), ax=ax1, lags=lags)
    ax1.set_title(f"ACF — {name}")
    plot_pacf(series.dropna(), ax=ax2, lags=lags, method="ywm")
    ax2.set_title(f"PACF — {name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_decompose(series: pd.Series, name: str, out_path: Path,
                   period: int = SEASONAL_PERIOD) -> Path:
    """seasonal_decompose additive (period=52) untuk deret terpadat."""
    dec = seasonal_decompose(series.dropna(), model="additive", period=period)
    fig = dec.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(f"Dekomposisi musiman (period={period}) — {name}", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_seasonality_month(weekly: pd.DataFrame, out_path: Path) -> Path:
    """Rata-rata units per bulan (agregat lintas deret) — soroti musim."""
    monthly = weekly.assign(month=weekly["week_start"].dt.month)
    m = monthly.groupby("month")["units"].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(m.index, m.values, color="#4C72B0")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Rata-rata units / minggu-deret")
    ax.set_title("Musiman: rata-rata units per bulan (Lebaran ~Mar–Mei, Harbolnas Nov–Des)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# --- Ringkasan musiman ------------------------------------------------------

def seasonal_summary(weekly: pd.DataFrame) -> dict:
    """Ringkasan musiman: rata-rata per bulan & minggu-tahun; anotasi Lebaran/Harbolnas."""
    w = weekly.copy()
    w["month"] = w["week_start"].dt.month
    w["woy"] = w["week_start"].dt.isocalendar().week.astype(int)
    by_month = w.groupby("month")["units"].mean().round(3)
    by_woy = w.groupby("woy")["units"].mean().round(3)

    # Petakan tanggal Lebaran ke week_start Senin dari grid.
    grid = pd.DatetimeIndex(sorted(w["week_start"].unique()))
    lebaran_weeks = []
    for d in pd.to_datetime(LEBARAN_DATES):
        monday = d.to_period("W").start_time
        if monday in grid:
            lebaran_weeks.append(monday.date().isoformat())
    return {
        "mean_units_by_month": {int(k): float(v) for k, v in by_month.items()},
        "mean_units_by_weekofyear": {int(k): float(v) for k, v in by_woy.items()},
        "lebaran_weeks": lebaran_weeks,
        "harbolnas_note": "Puncak belanja daring Nov–Des (11.11, 12.12); cek bulan 11–12.",
        "peak_month": int(by_month.idxmax()),
        "trough_month": int(by_month.idxmin()),
    }


# --- Orkestrasi -------------------------------------------------------------

def run_eda(weekly: pd.DataFrame, figures_dir: Path, results_dir: Path) -> dict:
    """Jalankan seluruh EDA; tulis figure + eda_summary.json; kembalikan ringkasan."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Deret terpadat & representatif berdasarkan total units.
    totals = weekly.groupby(["store", "brand"])["units"].sum().sort_values(ascending=False)
    keys = list(totals.index)
    densest = keys[0]
    representative = [keys[0], keys[len(keys) // 2], keys[-1]]  # padat, tengah, jarang

    figures: list[str] = []
    figures.append(str(plot_series_grid(weekly, figures_dir / "series_grid.png")))
    for store, brand in representative:
        name = series_key(store, brand)
        fp = figures_dir / f"acf_pacf_{store}_{brand}.png"
        plot_acf_pacf(to_series(weekly, store, brand), name, fp)
        figures.append(str(fp))
    d_store, d_brand = densest
    figures.append(str(plot_decompose(
        to_series(weekly, d_store, d_brand), series_key(d_store, d_brand),
        figures_dir / f"decompose_{d_store}_{d_brand}.png",
    )))
    figures.append(str(plot_seasonality_month(weekly, figures_dir / "seasonality_month.png")))

    adf = run_adf_all(weekly)
    n_stat = sum(1 for v in adf.values() if v["stationary_level"] is True)
    d_vals = [v["recommend_d"] for v in adf.values()]
    D_vals = [v["recommend_D"] for v in adf.values()]

    summary = {
        "n_series": len(adf),
        "n_weeks": int(weekly.groupby(["store", "brand"]).size().iloc[0]),
        "densest_series": series_key(*densest),
        "representative_series": [series_key(*k) for k in representative],
        "adf": adf,
        "adf_summary": {
            "n_stationary_level": n_stat,
            "n_nonstationary_level": len(adf) - n_stat,
        },
        "recommended_orders": {
            "d_mode": int(pd.Series(d_vals).mode().iloc[0]),
            "D_mode": int(pd.Series(D_vals).mode().iloc[0]),
        },
        "seasonality": seasonal_summary(weekly),
        "figures": figures,
    }

    out_json = results_dir / "eda_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(
        "EDA: %d figure, %d/%d deret stasioner (level), d_mode=%d D_mode=%d -> %s",
        len(figures), n_stat, len(adf),
        summary["recommended_orders"]["d_mode"],
        summary["recommended_orders"]["D_mode"], out_json,
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="Tahap 4 — EDA (figures + eda_summary.json)")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)

    from src.config import load_config

    cfg = load_config(args.config)
    cfg.paths.ensure()
    weekly_path = cfg.paths.interim / "weekly_store_brand.parquet"
    if not weekly_path.exists():
        from src.data.aggregate import _load_clean, aggregate_weekly

        aggregate_weekly(_load_clean(cfg), weekly_path)
    weekly = load_weekly(weekly_path)
    summary = run_eda(weekly, cfg.paths.figures, cfg.paths.results)
    print("Stage 4:", {k: summary[k] for k in
                       ("n_series", "adf_summary", "recommended_orders", "densest_series")})


if __name__ == "__main__":
    main()
