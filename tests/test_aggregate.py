"""Tahap 2 DoD — agregasi mingguan gerai x merek (§5, §7).

3680 baris, no-NaN, units int >= 0, Σunits konsisten, fraksi minggu-nol <= 0.30.
"""
from pathlib import Path

import pandas as pd
import pytest

from src.data.aggregate import aggregate_weekly
from src.data.clean import clean_transactions

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "pos_transactions_raw.csv"


@pytest.fixture(scope="module")
def clean():
    return clean_transactions(RAW)


@pytest.fixture(scope="module")
def weekly(clean):
    return aggregate_weekly(clean)


def test_row_count(weekly):
    assert len(weekly) == 3680  # 20 deret x 184 minggu


def test_columns(weekly):
    assert list(weekly.columns) == ["store", "brand", "week_start", "units"]


def test_n_series(weekly):
    assert weekly.groupby(["store", "brand"]).ngroups == 20


def test_each_series_184_weeks(weekly):
    counts = weekly.groupby(["store", "brand"]).size()
    assert (counts == 184).all()


def test_no_nan(weekly):
    assert weekly.notna().all().all()


def test_units_integer_nonneg(weekly):
    assert weekly["units"].dtype.kind in "iu"  # integer
    assert (weekly["units"] >= 0).all()


def test_units_sum_consistent(weekly):
    # Konsisten dengan Tahap 1 (Σqty = 6907).
    assert int(weekly["units"].sum()) == 6907


def test_zero_week_fraction_validates_d1(weekly):
    mean_frac = weekly.attrs["stats"]["mean_zero_week_frac"]
    assert mean_frac <= 0.30, f"mean zero-week frac {mean_frac:.4f} > 0.30 (D1)"


def test_week_grid_contiguous_monday(weekly):
    # Semua week_start Senin dan grid kontigu 7 hari untuk tiap deret.
    assert (weekly["week_start"].dt.weekday == 0).all()
    one = weekly[(weekly["store"] == "Toko_A") & (weekly["brand"] == "Xiaomi")]
    diffs = one["week_start"].sort_values().diff().dropna().dt.days.unique()
    assert list(diffs) == [7]


def test_parquet_roundtrip(clean, tmp_path):
    # Menjaga regresi: df.attrs harus JSON-serializable agar to_parquet tak gagal.
    out = tmp_path / "weekly.parquet"
    aggregate_weekly(clean, out)
    back = pd.read_parquet(out)
    assert len(back) == 3680
    assert list(back.columns) == ["store", "brand", "week_start", "units"]
    assert int(back["units"].sum()) == 6907
