"""Tahap 1 — Pembersihan transaksi POS mentah (Bab III §3.1.4).

CSV mentah -> DataFrame transaksi bersih. Kontrak §1.2 (sudah diverifikasi pada
data riil pos_transactions_raw.csv):

    raw=6320 | Void=76 | duplikat penuh (Paid)=19 | clean=6225 | Σqty=6907
    brand=5 | sku=15 | store=4 | rentang 2022-01-02 .. 2025-06-30

Catatan desain:
- `sales_no` BUKAN primary key (§9): dedup memakai baris identik penuh, bukan sales_no.
- `week_start` = Senin awal ISO-week. Pandas period 'W' = 'W-SUN' (Senin..Minggu),
  jadi start_time = Senin. Minggu tanpa penjualan adalah nilai valid (0 unit) yang
  ditangani di Tahap 2 (jangan diinterpolasi — lihat catatan intermittency §2).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# §1.3 Mapping merek dari product_name (case-insensitive). Redmi = sub-brand Xiaomi.
_BRAND_KEYS: list[tuple[str, str]] = [
    ("xiaomi", "Xiaomi"),
    ("redmi", "Xiaomi"),
    ("oppo", "OPPO"),
    ("samsung", "Samsung"),
    ("realme", "Realme"),
    ("vivo", "Vivo"),
]


def brand_from_product_name(name: object) -> str:
    """Kembalikan merek dari product_name; 'Other' jika tak dikenali."""
    n = str(name).lower()
    for key, brand in _BRAND_KEYS:
        if key in n:
            return brand
    return "Other"


def clean_transactions(raw_path: str | Path, out_path: str | Path | None = None) -> pd.DataFrame:
    """Bersihkan CSV POS mentah menjadi DataFrame transaksi Paid, terdedup, ber-merek.

    Statistik ringkas tersimpan di ``df.attrs['stats']`` untuk assertion DoD.
    """
    raw_path = Path(raw_path)
    raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    n_raw = len(raw)

    # Parse waktu. created_at = acuan resampling (§1.1); paid_at null pada Void.
    raw["created_at"] = pd.to_datetime(raw["created_at"])
    raw["paid_at"] = pd.to_datetime(raw["paid_at"], errors="coerce")

    # Langkah 2: hanya transaksi Paid (buang 76 Void: qty==0, grand_total==0).
    n_void = int((raw["status"] == "Void").sum())
    paid = raw[raw["status"] == "Paid"].copy()

    # Langkah 3: buang duplikat baris identik penuh (§9 — bukan berdasarkan sales_no).
    n_before = len(paid)
    clean = paid.drop_duplicates().reset_index(drop=True)
    n_dup = n_before - len(clean)

    # Langkah 4: kolom brand + assert tak ada 'Other'.
    clean["brand"] = clean["product_name"].map(brand_from_product_name)
    n_other = int((clean["brand"] == "Other").sum())
    assert n_other == 0, (
        f"{n_other} baris ter-map ke 'Other'; product_name tak dikenal BRAND_MAP: "
        f"{list(clean.loc[clean['brand'] == 'Other', 'product_name'].unique()[:5])}"
    )

    # Langkah 5: bucket mingguan -> week_start (Timestamp Senin), parquet-friendly.
    clean["week_start"] = clean["created_at"].dt.to_period("W").dt.start_time

    stats = {
        "n_raw": n_raw,
        "n_void": n_void,
        "n_duplicates": n_dup,
        "n_clean": len(clean),
        "units": int(clean["qty"].sum()),
    }
    clean.attrs["stats"] = stats
    logger.info(
        "clean: raw=%d void=%d dup=%d -> clean=%d (Σqty=%d)",
        n_raw, n_void, n_dup, len(clean), stats["units"],
    )

    # Langkah 6: simpan parquet.
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        clean.to_parquet(out_path, index=False)
        logger.info("Tersimpan: %s", out_path)

    return clean


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="Tahap 1 — clean POS transactions")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)

    from src.config import load_config

    cfg = load_config(args.config)
    cfg.paths.ensure()
    out = cfg.paths.interim / "transactions_clean.parquet"
    df = clean_transactions(cfg.paths.raw_csv, out)
    print("Stage 1 stats:", df.attrs["stats"])


if __name__ == "__main__":
    main()
