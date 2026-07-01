"""Tahap 3 — Google Trends mingguan sebagai eksogen (Bab III §3.1.3, keputusan D5).

Ambil indeks minat pencarian `geo='ID'` per merek, selaraskan ke grid minggu
Senin (grid Tahap 2), simpan sebagai cache CSV.

Output: data/interim/google_trends.csv
    kolom: week_start (Timestamp Senin), brand, gt_index (float 0..100)

Kontrak (§9 gotcha pytrends — API tak resmi, rentan rate-limit):
- **Cache idempoten**: bila file sudah ada, muat dari cache — TANPA memanggil network.
  `--force` memaksa fetch ulang (mis. menyegarkan cache fallback).
- **Fallback `--no-trends`**: hasilkan kolom gt_index = NaN agar pipeline tetap lulus
  sampai akhir (downstream memperlakukan gt_index yang seluruhnya NaN sebagai
  "tanpa eksogen" -> SARIMAX turun menjadi SARIMA).
- Bila fetch gagal (rate-limit/network/pytrends tak terpasang): tulis fallback NaN
  + WARNING jelas, sehingga pipeline tetap jalan; jalankan ulang `--force` untuk retry.

DoD (§5 Tahap 3):
- File tercache; run kedua tidak memanggil network.
- gt_index dalam [0,100]; cakupan minggu >= 95% grid (sisanya boleh di-ffill).
- Bila fallback aktif, pipeline tetap lulus sampai akhir.

Catatan penyelarasan waktu:
- Titik mingguan Google Trends ber-anchor Minggu (awal minggu ala Google). Grid
  demand kita ber-anchor Senin. Penyelarasan memakai reindex `nearest`
  (toleransi 3 hari) ke grid Senin lalu ffill maksimal 1 minggu — memetakan tiap
  titik GT ke label minggu Senin yang sama dengan deret permintaan.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GT_MIN, GT_MAX = 0.0, 100.0
MIN_COVERAGE = 0.95          # DoD: cakupan minggu >= 95% grid
_ALIGN_TOLERANCE = pd.Timedelta(days=3)
_FFILL_LIMIT = 1             # ffill maksimal 1 minggu (§ steps 2)

COLUMNS = ["week_start", "brand", "gt_index"]


# --- Penyelarasan & indeks -------------------------------------------------

def align_to_grid(
    brand_series: pd.Series,
    week_grid: pd.DatetimeIndex,
    tolerance: pd.Timedelta = _ALIGN_TOLERANCE,
    ffill_limit: int = _FFILL_LIMIT,
) -> pd.Series:
    """Selaraskan satu deret GT (index tanggal) ke grid minggu Senin.

    Reindex `nearest` (toleransi 3 hari) memetakan tiap titik GT ke Senin
    terdekat pada grid; sisa minggu tanpa titik di-ffill maksimal 1 minggu.
    """
    s = brand_series.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    aligned = s.reindex(week_grid, method="nearest", tolerance=tolerance)
    if ffill_limit:
        aligned = aligned.ffill(limit=ffill_limit)
    return aligned


def _rescale_0_100(s: pd.Series) -> pd.Series:
    """Skala per merek ke [0,100] dengan membagi puncaknya (nol sejati dijaga)."""
    peak = s.max()
    if not np.isfinite(peak) or peak <= 0:
        return s
    return (s / peak * GT_MAX).clip(GT_MIN, GT_MAX)


# --- Pengambilan (network) -------------------------------------------------

def fetch_trends(
    keywords_by_brand: dict[str, list[str]],
    timeframe: str,
    geo: str,
    week_grid: pd.DatetimeIndex,
    pause: float = 1.0,
) -> pd.DataFrame:
    """Ambil GT per merek via pytrends dan kembalikan long df [week_start, brand, gt_index].

    All-or-nothing: kegagalan merek mana pun memunculkan exception agar pemanggil
    dapat jatuh ke fallback dengan skema konsisten.
    """
    from pytrends.request import TrendReq  # impor lokal: opsional sampai fetch nyata

    pytrends = TrendReq(hl="id-ID", tz=420)  # tz=420 menit = WIB (UTC+7)
    frames: list[pd.DataFrame] = []
    for brand, keywords in keywords_by_brand.items():
        pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
        raw = pytrends.interest_over_time()
        if raw is None or raw.empty:
            raise RuntimeError(f"Google Trends kosong untuk merek {brand} ({keywords}).")
        if "isPartial" in raw.columns:
            raw = raw.drop(columns="isPartial")
        combined = raw.mean(axis=1)             # gabung kata kunci -> satu deret
        combined = _rescale_0_100(combined)
        aligned = align_to_grid(combined, week_grid)
        frames.append(
            pd.DataFrame({"week_start": week_grid, "brand": brand, "gt_index": aligned.values})
        )
        logger.info("GT fetched: %s (%d kata kunci)", brand, len(keywords))
        time.sleep(pause)                       # sopan terhadap rate-limit
    return pd.concat(frames, ignore_index=True)[COLUMNS]


def fallback_trends(
    keywords_by_brand: dict[str, list[str]], week_grid: pd.DatetimeIndex
) -> pd.DataFrame:
    """Fallback: long df dengan gt_index = NaN untuk tiap (minggu, merek)."""
    frames = [
        pd.DataFrame({"week_start": week_grid, "brand": brand, "gt_index": np.nan})
        for brand in keywords_by_brand
    ]
    return pd.concat(frames, ignore_index=True)[COLUMNS]


# --- Validasi --------------------------------------------------------------

def coverage_by_brand(trends: pd.DataFrame, week_grid: pd.DatetimeIndex) -> pd.Series:
    """Fraksi minggu grid dengan gt_index non-NaN, per merek."""
    n = len(week_grid)
    return trends.groupby("brand")["gt_index"].apply(lambda s: s.notna().sum() / n)


def validate_trends(
    trends: pd.DataFrame, week_grid: pd.DatetimeIndex, min_coverage: float = MIN_COVERAGE
) -> dict:
    """Validasi DoD untuk data GT nyata (dilewati untuk fallback all-NaN)."""
    vals = trends["gt_index"].dropna()
    in_range = bool(((vals >= GT_MIN) & (vals <= GT_MAX)).all()) if len(vals) else True
    cov = coverage_by_brand(trends, week_grid)
    return {
        "n_rows": len(trends),
        "n_brands": trends["brand"].nunique(),
        "in_range": in_range,
        "min_coverage": float(cov.min()) if len(cov) else 0.0,
        "coverage_by_brand": {b: round(float(v), 4) for b, v in cov.items()},
        "min_coverage_ok": bool(len(cov) and cov.min() >= min_coverage),
    }


# --- Orkestrasi (cache/fetch/fallback) -------------------------------------

def _read_cache(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["week_start"])
    return df[COLUMNS]


def build_google_trends(
    keywords_by_brand: dict[str, list[str]],
    week_grid: pd.DatetimeIndex,
    timeframe: str,
    geo: str,
    out_path: str | Path | None = None,
    no_trends: bool = False,
    force: bool = False,
    pause: float = 1.0,
) -> pd.DataFrame:
    """Bangun/muat data GT mingguan; kembalikan long df [week_start, brand, gt_index].

    `df.attrs['stats']['source']` in {cache, fetched, fallback}.
    """
    out_path = Path(out_path) if out_path is not None else None
    week_grid = pd.DatetimeIndex(week_grid)

    # 1) Cache idempoten: file ada + bukan force -> TANPA network.
    if out_path is not None and out_path.exists() and not force:
        trends = _read_cache(out_path)
        source = "cache"
        logger.info("GT cache dipakai (tanpa network): %s", out_path)
    elif no_trends:
        # 2) Fallback eksplisit diminta pengguna.
        trends = fallback_trends(keywords_by_brand, week_grid)
        source = "fallback"
        logger.warning("--no-trends: gt_index = NaN (SARIMAX turun ke SARIMA di hilir).")
    else:
        # 3) Coba fetch nyata; gagal apa pun -> fallback (pipeline tetap jalan).
        try:
            trends = fetch_trends(keywords_by_brand, timeframe, geo, week_grid, pause=pause)
            source = "fetched"
        except Exception as exc:  # noqa: BLE001 — sengaja luas: network/rate-limit/impor
            logger.warning(
                "Fetch Google Trends GAGAL (%s: %s). Memakai fallback gt_index=NaN. "
                "Jalankan ulang dengan --force untuk retry.",
                type(exc).__name__, exc,
            )
            trends = fallback_trends(keywords_by_brand, week_grid)
            source = "fallback"

    # Validasi hanya untuk data nyata (fallback all-NaN dikecualikan).
    stats = validate_trends(trends, week_grid)
    stats["source"] = source
    trends.attrs["stats"] = stats

    if source in ("cache", "fetched"):
        assert stats["in_range"], "gt_index di luar [0,100] — data GT tak valid."
        assert stats["min_coverage_ok"], (
            f"cakupan minggu {stats['min_coverage']:.3f} < {MIN_COVERAGE} "
            "untuk >=1 merek; sesuaikan kata kunci/timeframe atau ffill."
        )

    logger.info(
        "GT: %d baris, %d merek, source=%s, min_coverage=%.3f, in_range=%s",
        stats["n_rows"], stats["n_brands"], source,
        stats["min_coverage"], stats["in_range"],
    )

    # Tulis cache. Fallback tetap ditulis agar pipeline punya file & lulus end-to-end;
    # warning di atas + --force menyediakan jalur retry (tak mengunci fallback selamanya).
    if out_path is not None and source != "cache":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        trends.to_csv(out_path, index=False)
        logger.info("Tersimpan cache GT: %s (source=%s)", out_path, source)

    return trends


def _week_grid_from_stage2(cfg) -> pd.DatetimeIndex:
    """Ambil grid minggu Senin dari artefak Tahap 2 (jalankan Tahap 2 bila perlu)."""
    weekly_path = cfg.paths.interim / "weekly_store_brand.parquet"
    if not weekly_path.exists():
        from src.data.aggregate import _load_clean, aggregate_weekly

        aggregate_weekly(_load_clean(cfg), weekly_path)
    weekly = pd.read_parquet(weekly_path, columns=["week_start"])
    return pd.DatetimeIndex(sorted(weekly["week_start"].unique()))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="Tahap 3 — Google Trends eksogen (per merek)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-trends", action="store_true", help="fallback: gt_index=NaN")
    ap.add_argument("--force", action="store_true", help="fetch ulang walau cache ada")
    args = ap.parse_args(argv)

    from src.config import load_config

    cfg = load_config(args.config)
    cfg.paths.ensure()
    week_grid = _week_grid_from_stage2(cfg)
    out = cfg.paths.interim / "google_trends.csv"
    trends = build_google_trends(
        keywords_by_brand=cfg["keywords_by_brand"],
        week_grid=week_grid,
        timeframe=cfg["trends_timeframe"],
        geo=cfg["trends_geo"],
        out_path=out,
        no_trends=args.no_trends,
        force=args.force,
    )
    print("Stage 3 stats:", trends.attrs["stats"])


if __name__ == "__main__":
    main()
