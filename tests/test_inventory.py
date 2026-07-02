"""Tahap 8 DoD — optimasi inventori (§8, D8).

Verifikasi formula parameter inventori dari galat peramalan one-step (D9):
- z & safety stock NAIK saat service_level naik (DoD utama).
- ROP >= SS dan OUL >= ROP (konsistensi tingkatan).
- sigma galat = 0 saat ramalan sempurna -> SS = 0.
- Simulasi kebijakan base-stock (order-up-to) deterministik pada contoh manual:
  tanpa stockout saat S besar; menghitung stockout & fill-rate saat S kurang.
- Tabel parameter satu baris per deret; tabel dampak biaya satu baris per algoritma.
Semua offline & deterministik (tanpa fit model).
"""
import numpy as np
import pandas as pd
import pytest

from src.inventory import optimize as INV


def _preds(store, brand, y_true, y_pred):
    weeks = pd.date_range("2025-01-06", periods=len(y_true), freq="W-MON")
    return pd.DataFrame({"store": store, "brand": brand, "week_start": weeks,
                         "y_true": np.asarray(y_true, float),
                         "y_pred": np.asarray(y_pred, float)})


# --- Faktor keamanan z ------------------------------------------------------

def test_z_increases_with_service_level():
    assert INV.z_from_service_level(0.95) == pytest.approx(1.6448536, abs=1e-4)
    assert (INV.z_from_service_level(0.90)
            < INV.z_from_service_level(0.95)
            < INV.z_from_service_level(0.99))


# --- Formula parameter (DoD) ------------------------------------------------

def test_safety_stock_increases_with_service_level():
    sigma, L = 2.0, 2
    ss_90 = INV.safety_stock(sigma, INV.z_from_service_level(0.90), L)
    ss_95 = INV.safety_stock(sigma, INV.z_from_service_level(0.95), L)
    ss_99 = INV.safety_stock(sigma, INV.z_from_service_level(0.99), L)
    assert ss_90 < ss_95 < ss_99                     # DoD: SS naik dgn service_level
    assert ss_95 == pytest.approx(1.6448536 * 2.0 * np.sqrt(2))


def test_reorder_point_geq_safety_stock():
    ss = INV.safety_stock(1.5, INV.z_from_service_level(0.95), 2)
    rop = INV.reorder_point(mean_weekly_demand=3.0, lead_time_weeks=2, ss=ss)
    assert rop >= ss                                 # DoD: ROP >= SS
    assert rop == pytest.approx(3.0 * 2 + ss)


def test_order_up_to_geq_reorder_point():
    ss = INV.safety_stock(1.5, INV.z_from_service_level(0.95), 2)
    rop = INV.reorder_point(3.0, 2, ss)
    oul = INV.order_up_to_level(rop, mean_weekly_demand=3.0, review_period_weeks=1)
    assert oul >= rop                                # review demand >= 0
    assert oul == pytest.approx(rop + 3.0 * 1)


def test_forecast_error_sigma_zero_when_perfect():
    assert INV.forecast_error_sigma([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(0.0)
    # ramalan sempurna -> SS = 0
    params = INV.series_params([1, 2, 3, 4], [1, 2, 3, 4],
                               z=INV.z_from_service_level(0.95),
                               lead_time_weeks=2, review_period_weeks=1)
    assert params["safety_stock"] == pytest.approx(0.0)
    assert params["mean_weekly_demand"] == pytest.approx(2.5)


# --- Simulasi kebijakan base-stock ------------------------------------------

def test_simulate_no_stockout_when_S_large():
    r = INV.simulate_base_stock([2, 3, 1], S=100.0, lead_time_weeks=2,
                                review_period_weeks=1)
    assert r["stockout_weeks"] == 0
    assert r["shortage_units"] == pytest.approx(0.0)
    assert r["fill_rate"] == pytest.approx(1.0)
    assert r["holding_unit_weeks"] > 0


def test_simulate_counts_stockout_when_starved():
    # S=0 & mulai kosong -> tiap minggu kehabisan, fill-rate 0
    r = INV.simulate_base_stock([5, 5], S=0.0, lead_time_weeks=1,
                                review_period_weeks=1, init_inventory=0.0)
    assert r["stockout_weeks"] == 2
    assert r["shortage_units"] == pytest.approx(10.0)
    assert r["fill_rate"] == pytest.approx(0.0)


def test_simulate_manual_replenishment():
    # demand konstan 2, S=10, L=1, R=1, mulai penuh (10):
    # tiap minggu on_hand berakhir 8 -> holding 4*8=32, tanpa stockout
    r = INV.simulate_base_stock([2, 2, 2, 2], S=10.0, lead_time_weeks=1,
                                review_period_weeks=1)
    assert r["stockout_weeks"] == 0
    assert r["holding_unit_weeks"] == pytest.approx(32.0)


# --- Tabel keluaran ---------------------------------------------------------

def test_inventory_params_table_one_row_per_series():
    preds = pd.concat([_preds("Toko_A", "Xiaomi", [1, 2, 0, 3, 2, 1],
                              [1, 1, 1, 2, 2, 1]),
                       _preds("Toko_B", "OPPO", [0, 1, 2, 1, 0, 2],
                              [1, 1, 1, 1, 1, 1])], ignore_index=True)
    tab = INV.inventory_params_table(preds, service_level=0.95,
                                     lead_time_weeks=2, review_period_weeks=1)
    assert len(tab) == 2                             # 1 baris per deret
    for col in ["store", "brand", "sigma_error", "mean_weekly_demand",
                "safety_stock", "reorder_point", "order_up_to_level"]:
        assert col in tab.columns
    assert (tab["safety_stock"] >= 0).all()
    assert (tab["order_up_to_level"] >= tab["reorder_point"]).all()
    assert (tab["reorder_point"] >= tab["safety_stock"]).all()


def test_cost_impact_one_row_per_algo():
    preds_by_algo = {
        "SARIMAX": _preds("Toko_A", "Xiaomi", [1, 2, 0, 3, 2, 1], [2, 1, 1, 2, 3, 0]),
        "RF": _preds("Toko_A", "Xiaomi", [1, 2, 0, 3, 2, 1], [1, 2, 0, 3, 2, 1]),
        "LSTM": _preds("Toko_A", "Xiaomi", [1, 2, 0, 3, 2, 1], [1, 1, 1, 1, 1, 1]),
    }
    tab = INV.cost_impact_table(preds_by_algo, service_level=0.95,
                                lead_time_weeks=2, review_period_weeks=1,
                                holding_cost=0.02)
    assert len(tab) == 3
    for col in ["algo", "holding_cost_total", "stockout_weeks", "fill_rate",
                "safety_stock_mean"]:
        assert col in tab.columns
    assert np.isfinite(tab["holding_cost_total"]).all()
    # ramalan sempurna (RF) -> sigma 0 -> SS mean 0
    assert tab.set_index("algo").loc["RF", "safety_stock_mean"] == pytest.approx(0.0)
