"""Tahap 9 DoD — logika DSS pengadaan (§9).

Kebijakan order-up-to: pesan hingga OUL saat stok <= ROP; kuantitas dibulatkan ke
atas (unit utuh). Status tiga tingkat (aman/mendekati ROP/di bawah ROP). Konteks
ramalan JUJUR: menyurfacekan prediksi walk-forward tervalidasi TERAKHIR (Tahap 7)
beserta minggu & tanggal as-of — bukan forecast live (prototipe tak melatih ulang).
Semua offline & deterministik (tanpa Streamlit).
"""
import numpy as np
import pandas as pd
import pytest

from src.dss import recommend as R


def _series(store, brand, weeks, y_true, y_pred):
    return pd.DataFrame({"store": store, "brand": brand,
                         "week_start": pd.to_datetime(weeks),
                         "y_true": np.asarray(y_true, float),
                         "y_pred": np.asarray(y_pred, float)})


# --- Rekomendasi order-up-to ------------------------------------------------

def test_order_below_rop_fills_to_oul():
    rec = R.order_recommendation(current_stock=5, reorder_point=12.6,
                                 order_up_to_level=16.1)
    assert rec["status"] == R.STATUS_BELOW
    assert rec["order_qty"] == 12                # ceil(16.1 - 5) = 12, unit utuh
    assert isinstance(rec["order_qty"], int)


def test_no_order_when_stock_safe():
    rec = R.order_recommendation(current_stock=20, reorder_point=12.6,
                                 order_up_to_level=16.1)
    assert rec["status"] == R.STATUS_SAFE
    assert rec["order_qty"] == 0


def test_near_rop_band():
    # rop=10, near_frac=0.15 -> ambang aman di 11.5
    assert R.order_recommendation(10, 10, 14)["status"] == R.STATUS_BELOW   # tepat di ROP
    assert R.order_recommendation(11, 10, 14)["status"] == R.STATUS_NEAR
    assert R.order_recommendation(12, 10, 14)["status"] == R.STATUS_SAFE
    assert R.order_recommendation(11, 10, 14)["order_qty"] == 0             # belum pesan


def test_recommend_text_mentions_action():
    order = R.recommend_text("Toko_A", "Xiaomi", current_stock=5,
                             reorder_point=12.6, order_up_to_level=16.1)
    assert "pesan" in order.lower() and "12" in order
    safe = R.recommend_text("Toko_A", "Xiaomi", current_stock=30,
                            reorder_point=12.6, order_up_to_level=16.1)
    assert "belum" in safe.lower()


# --- Konteks ramalan jujur (load-only proxy) --------------------------------

def test_forecast_context_uses_last_validated_week():
    df = _series("Toko_A", "Xiaomi",
                 ["2025-06-16", "2025-06-23", "2025-06-30"],
                 y_true=[3, 2, 4], y_pred=[2.5, 2.1, 3.2])
    ctx = R.forecast_context(df)
    assert ctx["forecast_week"] == pd.Timestamp("2025-06-30")
    assert ctx["as_of"] == pd.Timestamp("2025-06-23")     # minggu sebelum ramalan
    assert ctx["forecast_value"] == pytest.approx(3.2)
    assert ctx["n_weeks"] == 3


def test_forecast_label_is_honest_about_no_live_refit():
    ctx = R.forecast_context(_series("Toko_A", "Xiaomi",
                             ["2025-06-23", "2025-06-30"], [2, 4], [2.1, 3.2]))
    label = R.forecast_label(ctx)
    assert "2025" in label
    assert "tervalidasi" in label.lower()                 # jujur: prediksi tervalidasi
    assert "30 Jun 2025" in label and "23 Jun 2025" in label


# --- Tabel DSS gabungan (prediksi + parameter inventori) --------------------

def test_build_dss_table_joins_preds_and_params():
    preds = pd.concat([
        _series("Toko_A", "Xiaomi", ["2025-06-23", "2025-06-30"], [2, 4], [2.1, 3.2]),
        _series("Toko_A", "OPPO", ["2025-06-23", "2025-06-30"], [1, 0], [1.0, 0.5]),
    ], ignore_index=True)
    params = pd.DataFrame({
        "store": ["Toko_A", "Toko_A"], "brand": ["Xiaomi", "OPPO"],
        "mean_weekly_demand": [3.5, 1.0], "safety_stock": [5.6, 3.9],
        "reorder_point": [12.6, 8.6], "order_up_to_level": [16.1, 10.9]})
    tab = R.build_dss_table(preds, params)
    assert len(tab) == 2
    for col in ["store", "brand", "forecast_week", "as_of", "forecast_value",
                "reorder_point", "order_up_to_level", "safety_stock"]:
        assert col in tab.columns
    row = tab.set_index("brand").loc["Xiaomi"]
    assert row["reorder_point"] == pytest.approx(12.6)
    assert row["forecast_value"] == pytest.approx(3.2)
    assert row["forecast_week"] == pd.Timestamp("2025-06-30")
