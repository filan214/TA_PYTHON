"""Tahap 9 — Logika DSS pengadaan stok (Bab III §3.1.9).

Lapisan keputusan murni (tanpa Streamlit) yang dipakai dashboard. Dua fungsi inti:

1. **Rekomendasi order-up-to** (`order_recommendation`): membandingkan stok terkini
   pemilik toko dengan `reorder_point` (ROP) & `order_up_to_level` (OUL) dari Tahap 8.
   Bila stok <= ROP → pesan hingga OUL (kuantitas dibulatkan ke ATAS, unit utuh).
   Status tiga tingkat untuk indikator warna: aman / mendekati ROP / di bawah ROP.

2. **Konteks ramalan JUJUR** (`forecast_context` / `forecast_label`): dashboard TIDAK
   melakukan pelatihan ulang/forecast live. Ia menyurfacekan **prediksi one-step
   walk-forward TERVALIDASI TERAKHIR** dari Tahap 7 (RF·gt) — beserta minggu yang
   diramal dan tanggal *as-of* data. Label ditulis eksplisit agar tidak menyesatkan
   sebagai "ramalan minggu depan" live. Ini konsisten dengan lingkup prototipe
   (tanpa integrasi POS real-time, tanpa retraining otomatis) — lihat batasan Bab I/V.
"""
from __future__ import annotations

import math

import pandas as pd

STATUS_SAFE = "aman"
STATUS_NEAR = "mendekati ROP"
STATUS_BELOW = "di bawah ROP"

# Indikator warna & emoji untuk kartu status dashboard.
STATUS_COLOR = {STATUS_SAFE: "green", STATUS_NEAR: "orange", STATUS_BELOW: "red"}
STATUS_EMOJI = {STATUS_SAFE: "🟢", STATUS_NEAR: "🟡", STATUS_BELOW: "🔴"}

NEAR_FRAC_DEFAULT = 0.15   # pita "mendekati ROP" = (ROP, ROP*(1+frac)]


def order_recommendation(current_stock: float, reorder_point: float,
                         order_up_to_level: float,
                         near_frac: float = NEAR_FRAC_DEFAULT) -> dict:
    """Rekomendasi kuantitas pesan (kebijakan periodic-review order-up-to).

    stok <= ROP            -> "di bawah ROP", pesan ceil(OUL - stok) unit
    ROP < stok <= ROP·1,15 -> "mendekati ROP", belum pesan (peringatan dini)
    stok > ROP·1,15        -> "aman", belum pesan
    """
    if current_stock <= reorder_point:
        status = STATUS_BELOW
        order_qty = int(math.ceil(max(0.0, order_up_to_level - current_stock)))
    elif current_stock <= reorder_point * (1.0 + near_frac):
        status = STATUS_NEAR
        order_qty = 0
    else:
        status = STATUS_SAFE
        order_qty = 0
    return {"order_qty": order_qty, "status": status,
            "color": STATUS_COLOR[status], "emoji": STATUS_EMOJI[status]}


def recommend_text(store: str, brand: str, current_stock: float,
                   reorder_point: float, order_up_to_level: float,
                   near_frac: float = NEAR_FRAC_DEFAULT) -> str:
    """Kalimat rekomendasi bahasa awam untuk pemilik toko."""
    rec = order_recommendation(current_stock, reorder_point, order_up_to_level, near_frac)
    stok = f"{current_stock:g}"
    if rec["order_qty"] > 0:
        return (f"Stok {brand} di {store} saat ini {stok} unit; ROP={reorder_point:.0f} "
                f"→ SARAN: pesan {rec['order_qty']} unit (isi hingga OUL="
                f"{order_up_to_level:.0f}).")
    return (f"Stok {brand} di {store} saat ini {stok} unit; ROP={reorder_point:.0f} "
            f"→ stok {rec['status']}, belum perlu memesan.")


def forecast_context(series_df: pd.DataFrame, freq_days: int = 7) -> dict:
    """Ambil prediksi walk-forward TERVALIDASI TERAKHIR untuk satu deret.

    `series_df`: baris satu gerai×merek dari prediksi Tahap 7 (kolom week_start,
    y_true, y_pred). Mengembalikan minggu yang diramal, tanggal as-of (minggu
    sebelumnya = batas data yang dipakai one-step), dan nilai ramalan/aktual.
    """
    g = series_df.sort_values("week_start")
    last = g.iloc[-1]
    forecast_week = pd.Timestamp(last["week_start"])
    as_of = forecast_week - pd.Timedelta(days=freq_days)
    return {"forecast_week": forecast_week, "as_of": as_of,
            "forecast_value": float(last["y_pred"]),
            "actual_value": float(last["y_true"]), "n_weeks": int(len(g))}


def forecast_label(ctx: dict) -> str:
    """Label jujur soal sumber angka ramalan (bukan forecast live)."""
    return (f"Ramalan untuk minggu {ctx['forecast_week']:%d %b %Y} — prediksi "
            f"walk-forward TERVALIDASI terakhir (Tahap 7), berdasar data hingga "
            f"{ctx['as_of']:%d %b %Y}. Prototipe DSS tidak melatih ulang secara "
            f"langsung (tanpa integrasi POS real-time).")


def build_dss_table(winner_preds: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    """Gabung konteks ramalan (per deret) + parameter inventori Tahap 8.

    Satu baris per gerai×merek: minggu ramalan, as-of, nilai ramalan, plus SS/ROP/OUL.
    """
    rows = []
    for (store, brand), g in winner_preds.groupby(["store", "brand"]):
        ctx = forecast_context(g)
        rows.append({"store": store, "brand": brand,
                     "forecast_week": ctx["forecast_week"], "as_of": ctx["as_of"],
                     "forecast_value": ctx["forecast_value"],
                     "actual_last": ctx["actual_value"]})
    fc = pd.DataFrame(rows)
    keep = ["store", "brand", "mean_weekly_demand", "safety_stock",
            "reorder_point", "order_up_to_level"]
    keep = [c for c in keep if c in params.columns]
    out = fc.merge(params[keep], on=["store", "brand"], how="left")
    return out.sort_values(["store", "brand"]).reset_index(drop=True)
