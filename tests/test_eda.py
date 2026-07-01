"""Tahap 4 DoD — EDA figures + eda_summary.json (§5, §7).

Menjalankan run_eda pada data riil (clean -> aggregate) satu kali, lalu memverifikasi:
figure ter-generate tanpa error, dan eda_summary.json memuat p-value ADF per 20
deret + rekomendasi d/D. Backend matplotlib 'Agg' (headless).
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.aggregate import aggregate_weekly
from src.data.clean import clean_transactions
from src.eda import explore

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "pos_transactions_raw.csv"


@pytest.fixture(scope="module")
def weekly() -> pd.DataFrame:
    return aggregate_weekly(clean_transactions(RAW))


@pytest.fixture(scope="module")
def eda_out(weekly, tmp_path_factory):
    base = tmp_path_factory.mktemp("eda")
    figs, res = base / "figures", base / "results"
    summary = explore.run_eda(weekly, figs, res)
    return summary, figs, res


# --- DoD: figures ter-generate ---------------------------------------------

def test_series_grid_figure_exists(eda_out):
    _, figs, _ = eda_out
    f = figs / "series_grid.png"
    assert f.exists() and f.stat().st_size > 0


def test_decompose_figure_exists_for_densest(eda_out):
    summary, figs, _ = eda_out
    store, brand = summary["densest_series"].split("|")
    f = figs / f"decompose_{store}_{brand}.png"
    assert f.exists() and f.stat().st_size > 0


def test_acf_pacf_figures_for_representative(eda_out):
    summary, figs, _ = eda_out
    for key in summary["representative_series"]:
        store, brand = key.split("|")
        f = figs / f"acf_pacf_{store}_{brand}.png"
        assert f.exists() and f.stat().st_size > 0


def test_all_listed_figures_on_disk(eda_out):
    summary, _, _ = eda_out
    for fp in summary["figures"]:
        assert Path(fp).exists() and Path(fp).stat().st_size > 0


# --- DoD: eda_summary.json isi ADF per 20 deret + rekomendasi d/D -----------

def test_summary_json_written(eda_out):
    _, _, res = eda_out
    assert (res / "eda_summary.json").exists()


def test_adf_covers_all_20_series(eda_out):
    summary, _, res = eda_out
    disk = json.loads((res / "eda_summary.json").read_text(encoding="utf-8"))
    assert len(disk["adf"]) == 20
    assert summary["n_series"] == 20


def test_each_series_has_pvalue_and_orders(eda_out):
    summary, _, _ = eda_out
    for key, rec in summary["adf"].items():
        assert set(rec) >= {"p_value_level", "recommend_d", "recommend_D", "seasonal_strength"}
        # p-value None hanya jika deret konstan; di data ini harus numerik.
        assert rec["p_value_level"] is None or 0.0 <= rec["p_value_level"] <= 1.0
        assert rec["recommend_d"] in (0, 1, 2)
        assert rec["recommend_D"] in (0, 1)


def test_recommended_orders_present(eda_out):
    summary, _, _ = eda_out
    assert summary["recommended_orders"]["d_mode"] in (0, 1, 2)
    assert summary["recommended_orders"]["D_mode"] in (0, 1)


def test_seasonality_summary_shape(eda_out):
    summary, _, _ = eda_out
    seas = summary["seasonality"]
    assert len(seas["mean_units_by_month"]) == 12
    assert 1 <= seas["peak_month"] <= 12
    # Semua 4 tanggal Lebaran jatuh dalam rentang grid -> harus terpetakan.
    assert len(seas["lebaran_weeks"]) == 4


# --- Unit: fungsi inti ------------------------------------------------------

def test_adf_pvalue_none_for_constant():
    assert explore.adf_pvalue(pd.Series([3.0] * 100)) is None


def test_adf_pvalue_numeric_for_varying(weekly):
    s = explore.to_series(weekly, "Toko_A", "Xiaomi")
    p = explore.adf_pvalue(s)
    assert p is not None and 0.0 <= p <= 1.0


def test_recommend_orders_keys(weekly):
    rec = explore.recommend_orders(explore.to_series(weekly, "Toko_A", "Xiaomi"))
    assert rec["recommend_d"] in (0, 1, 2) and rec["recommend_D"] in (0, 1)
