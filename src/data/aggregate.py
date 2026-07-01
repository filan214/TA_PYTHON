"""Tahap 2 — Agregasi mingguan gerai x merek (Bab III §3.1.4).

Ubah transaksi bersih menjadi 20 deret mingguan (4 gerai x 5 merek) pada grid
waktu penuh. Minggu tanpa penjualan diisi 0 (nilai valid, bukan missing — §2).

Output: data/interim/weekly_store_brand.parquet
    kolom: store, brand, week_start (Timestamp Senin), units (int >= 0)

DoD (§5 Tahap 2):
    - 20 kombinasi x 184 minggu = 3680 baris; tak ada NaN; units int >= 0
    - Σunits == 6907 (konsisten dengan Tahap 1)
    - mean fraksi minggu-nol per deret <= 0.30 (validasi keputusan desain D1)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

N_WEEKS_EXPECTED = 184  # grid penuh (§1.2, diverifikasi pada data riil)
MAX_ZERO_WEEK_FRAC = 0.30  # ambang D1


def zero_week_fraction(weekly: pd.DataFrame) -> pd.Series:
    """Fraksi minggu dengan units==0 per deret (store, brand), urut menurun."""
    return (
        weekly.assign(is_zero=weekly["units"].eq(0))
        .groupby(["store", "brand"])["is_zero"]
        .mean()
        .sort_values(ascending=False)
    )


def aggregate_weekly(
    clean: pd.DataFrame,
    out_path: str | Path | None = None,
    max_zero_week_frac: float = MAX_ZERO_WEEK_FRAC,
) -> pd.DataFrame:
    """Agregasi transaksi bersih -> deret mingguan gerai x merek pada grid penuh."""
    # Langkah 1: jumlahkan qty per (store, brand, minggu).
    g = (
        clean.groupby(["store", "brand", "week_start"], as_index=False)["qty"]
        .sum()
        .rename(columns={"qty": "units"})
    )

    # Langkah 2: grid penuh = kartesian {store} x {brand} x {semua minggu}, fill 0.
    stores = sorted(clean["store"].unique())
    brands = sorted(clean["brand"].unique())
    weeks = pd.date_range(
        clean["week_start"].min(), clean["week_start"].max(), freq="W-MON"
    )
    full_index = pd.MultiIndex.from_product(
        [stores, brands, weeks], names=["store", "brand", "week_start"]
    )
    weekly = (
        g.set_index(["store", "brand", "week_start"])
        .reindex(full_index, fill_value=0)
        .reset_index()
        .sort_values(["store", "brand", "week_start"])
        .reset_index(drop=True)
    )
    # Langkah 3: units integer (week_start sudah Timestamp Senin dari Tahap 1).
    weekly["units"] = weekly["units"].astype(int)

    # --- Validasi DoD ---
    n_series = weekly.groupby(["store", "brand"]).ngroups
    zero_frac = zero_week_fraction(weekly)
    mean_zero_frac = float(zero_frac.mean())

    # attrs harus JSON-serializable: to_parquet menyimpan df.attrs sebagai metadata
    # JSON, jadi jangan menaruh objek Series/ndarray di sini (hanya primitif/dict/list).
    stats = {
        "n_rows": len(weekly),
        "n_series": int(n_series),
        "n_weeks": len(weeks),
        "units": int(weekly["units"].sum()),
        "mean_zero_week_frac": mean_zero_frac,
        "zero_week_frac_by_series": {
            f"{s}|{b}": round(float(v), 4) for (s, b), v in zero_frac.items()
        },
    }
    weekly.attrs["stats"] = stats

    logger.info(
        "aggregate: %d baris (%d deret x %d minggu), Σunits=%d, mean_zero_frac=%.4f",
        stats["n_rows"], stats["n_series"], stats["n_weeks"], stats["units"], mean_zero_frac,
    )
    logger.info("5 deret dengan fraksi minggu-nol tertinggi:")
    for (s, b), v in zero_frac.head(5).items():
        logger.info("    %s x %s: %.3f", s, b, v)

    # Guard D1 (§4 instruksi: STOP bila data menyalahi asumsi desain).
    assert weekly["units"].notna().all(), "Ada NaN pada units setelah reindex/fill."
    assert (weekly["units"] >= 0).all(), "units negatif — tak seharusnya."
    assert len(weeks) == N_WEEKS_EXPECTED, (
        f"Grid minggu = {len(weeks)}, diharapkan {N_WEEKS_EXPECTED} (§1.2)."
    )
    assert mean_zero_frac <= max_zero_week_frac, (
        f"mean fraksi minggu-nol = {mean_zero_frac:.4f} > {max_zero_week_frac} "
        "-> keputusan D1 (gerai x merek) tidak valid pada data ini; STOP & lapor."
    )

    # Langkah 4: simpan.
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        weekly.to_parquet(out_path, index=False)
        logger.info("Tersimpan: %s", out_path)

    return weekly


def _load_clean(cfg) -> pd.DataFrame:
    """Muat transaksi bersih dari interim; bila belum ada, jalankan Tahap 1."""
    clean_path = cfg.paths.interim / "transactions_clean.parquet"
    if clean_path.exists():
        return pd.read_parquet(clean_path)
    from src.data.clean import clean_transactions

    return clean_transactions(cfg.paths.raw_csv, clean_path)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="Tahap 2 — agregasi mingguan gerai x merek")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)

    from src.config import load_config

    cfg = load_config(args.config)
    cfg.paths.ensure()
    clean = _load_clean(cfg)
    out = cfg.paths.interim / "weekly_store_brand.parquet"
    weekly = aggregate_weekly(clean, out)
    print("Stage 2 stats:", weekly.attrs["stats"])


if __name__ == "__main__":
    main()
