"""Tahap 1 DoD — assertion kontrak data §1.2 / §5 (Bab III §3.1.4).

Menjalankan `clean_transactions` pada data riil dan memverifikasi setiap fakta
yang tercantum di Data Contract. `pytest -q` harus hijau sebelum lanjut Tahap 2.
"""
from pathlib import Path

import pandas as pd
import pytest

from src.data.clean import brand_from_product_name, clean_transactions

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "pos_transactions_raw.csv"


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_csv(RAW, encoding="utf-8-sig")


@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    return clean_transactions(RAW)


# --- DoD §5 Tahap 1 ---------------------------------------------------------

def test_raw_row_count(raw):
    assert len(raw) == 6320


def test_void_removed(clean):
    assert clean.attrs["stats"]["n_void"] == 76


def test_duplicates_removed(clean):
    assert clean.attrs["stats"]["n_duplicates"] == 19


def test_clean_row_count(clean):
    assert len(clean) == 6225
    assert clean.attrs["stats"]["n_clean"] == 6225


def test_units_sum(clean):
    assert int(clean["qty"].sum()) == 6907


def test_brand_count(clean):
    assert clean["brand"].nunique() == 5


def test_no_other_brand(clean):
    assert not (clean["brand"] == "Other").any()


def test_sku_count(clean):
    assert clean["sku"].nunique() == 15


def test_store_count(clean):
    assert clean["store"].nunique() == 4


# --- Invarian tambahan (memperkuat keyakinan) -------------------------------

def test_only_paid_status(clean):
    assert (clean["status"] == "Paid").all()


def test_qty_positive(clean):
    # qty==0 hanya pada Void (§1.1); setelah dibersihkan harus >= 1.
    assert (clean["qty"] >= 1).all()


def test_week_start_is_monday(clean):
    # Period 'W' (W-SUN) -> week_start harus Senin (weekday == 0).
    assert (clean["week_start"].dt.weekday == 0).all()


def test_brand_map_examples():
    assert brand_from_product_name("Xiaomi Redmi Note 12") == "Xiaomi"
    assert brand_from_product_name("Redmi 10C") == "Xiaomi"
    assert brand_from_product_name("OPPO A57") == "OPPO"
    assert brand_from_product_name("Samsung Galaxy A14") == "Samsung"
    assert brand_from_product_name("realme C55") == "Realme"
    assert brand_from_product_name("vivo Y16") == "Vivo"
