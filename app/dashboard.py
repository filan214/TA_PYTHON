"""Tahap 9 — Purwarupa DSS Pengadaan Stok Smartphone (Bab III §3.1.9).

Dashboard Streamlit dengan tiga subsistem DSS:
    - Data      : memuat artefak (prediksi tervalidasi, parameter inventori, riwayat).
    - Model     : ramalan per gerai×merek dari model terbaik (RF·gt, Tahap 7).
    - UI/saran  : kartu status warna + rekomendasi kuantitas pesan bahasa awam,
                  grafik aktual vs ramalan, input stok terkini manual.

Kejujuran ramalan (keputusan desain, load-only proxy): dashboard TIDAK melatih ulang
model. Angka "ramalan" adalah prediksi one-step **walk-forward tervalidasi TERAKHIR**
dari Tahap 7, diberi label minggu & tanggal as-of secara eksplisit — bukan forecast
live. Konsisten dgn lingkup prototipe (tanpa integrasi POS real-time / retraining
otomatis), lihat batasan Bab I/V.

Jalankan:  streamlit run app/dashboard.py -- --config config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.dss import recommend as R  # noqa: E402

ALGO_LABEL = {"sarimax": "SARIMAX", "rf": "RF", "lstm": "LSTM"}


def _config_path() -> str:
    """Ambil --config dari argumen setelah '--' (streamlit run app.py -- --config ...)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args, _ = ap.parse_known_args()
    return args.config


@st.cache_data(show_spinner=False)
def load_artifacts(config_path: str) -> dict:
    """Muat semua artefak Tahap 6–8 + pilih model terbaik (MAE_mean terendah, Tahap 7)."""
    cfg = load_config(config_path)
    results = cfg.paths.results

    summ = pd.read_csv(results / "metrics_summary.csv")
    summ = summ[summ["variant"].isin(["baseline", "gt"])]
    best = summ.loc[summ["MAE_mean"].idxmin()]
    algo, variant = str(best["algo"]), str(best["variant"])

    preds = pd.read_parquet(results / f"predictions_{algo}_{variant}.parquet")
    params = pd.read_csv(results / "inventory_params.csv")
    weekly = pd.read_parquet(cfg.paths.interim / "weekly_store_brand.parquet")
    cost = pd.read_csv(results / "cost_impact.csv")

    dss = R.build_dss_table(preds, params)
    return {"model_label": f"{ALGO_LABEL.get(algo, algo)}({variant})",
            "mae_mean": float(best["MAE_mean"]), "mase_mean": float(best.get("MASE_mean", float("nan"))),
            "preds": preds, "params": params, "weekly": weekly, "cost": cost, "dss": dss,
            "service_level": float(cfg["service_level"]),
            "lead_time_weeks": int(cfg["lead_time_weeks"]),
            "review_period_weeks": int(cfg["review_period_weeks"])}


def _status_card(col, brand: str, row: pd.Series, current_stock: float):
    rec = R.order_recommendation(current_stock, row["reorder_point"],
                                 row["order_up_to_level"])
    with col:
        st.markdown(f"### {rec['emoji']} {brand}")
        st.caption(f"Status: **{rec['status']}**")
        st.metric("Ramalan minggu ini (unit)", f"{row['forecast_value']:.1f}")
        c1, c2 = st.columns(2)
        c1.metric("ROP", f"{row['reorder_point']:.1f}")
        c2.metric("OUL", f"{row['order_up_to_level']:.1f}")
        if rec["order_qty"] > 0:
            st.error(f"SARAN: **pesan {rec['order_qty']} unit** "
                     f"(isi hingga OUL {row['order_up_to_level']:.0f}).")
        else:
            st.success("Stok cukup — belum perlu memesan.")


def _forecast_chart(preds: pd.DataFrame, store: str, brand: str):
    g = preds[(preds["store"] == store) & (preds["brand"] == brand)].sort_values("week_start")
    if g.empty:
        st.info("Tak ada data prediksi untuk deret ini.")
        return
    chart_df = g.set_index("week_start")[["y_true", "y_pred"]].rename(
        columns={"y_true": "Aktual", "y_pred": "Ramalan"})
    st.line_chart(chart_df, height=240)


def render(config_path: str) -> None:
    st.set_page_config(page_title="DSS Pengadaan Stok Smartphone", page_icon="📦",
                       layout="wide")
    st.title("📦 DSS Pengadaan Stok Smartphone")

    try:
        art = load_artifacts(config_path)
    except FileNotFoundError as e:
        st.error(f"Artefak belum lengkap: {e}. Jalankan `python -m src.run_all` dulu.")
        st.stop()

    dss = art["dss"]
    fc_week = pd.Timestamp(dss["forecast_week"].max())
    as_of = pd.Timestamp(dss["as_of"].max())

    st.caption(
        f"Model terbaik: **{art['model_label']}** (MAE {art['mae_mean']:.3f}, "
        f"MASE {art['mase_mean']:.3f}) · service level {art['service_level']:.0%} · "
        f"lead time {art['lead_time_weeks']} mgg · review {art['review_period_weeks']} mgg")
    st.warning(
        f"⚠️ **Ramalan untuk minggu {fc_week:%d %b %Y}** — prediksi walk-forward "
        f"TERVALIDASI terakhir (Tahap 7), berdasar data hingga **{as_of:%d %b %Y}**. "
        f"Prototipe ini **tidak** melatih ulang model secara langsung (tanpa integrasi "
        f"POS real-time / retraining otomatis).")

    stores = sorted(dss["store"].unique())
    with st.sidebar:
        st.header("Pengaturan")
        store = st.selectbox("Pilih gerai", stores, index=0)
        st.caption("Masukkan stok terkini tiap merek untuk saran real-time:")
        sub = dss[dss["store"] == store].sort_values("brand")
        stock_input = {}
        for _, row in sub.iterrows():
            default = int(round(row["mean_weekly_demand"]))     # contoh stok rendah
            stock_input[row["brand"]] = st.number_input(
                f"Stok {row['brand']}", min_value=0, value=default, step=1,
                key=f"stock_{store}_{row['brand']}")

    tab_rec, tab_model, tab_data = st.tabs(
        ["🛒 Rekomendasi Pengadaan", "📈 Model & Ramalan", "🗂️ Data & Biaya"])

    # --- Subsistem UI/rekomendasi ---
    with tab_rec:
        st.subheader(f"Rekomendasi untuk {store}")
        brands = list(sub["brand"])
        cols = st.columns(min(len(brands), 3))
        for i, (_, row) in enumerate(sub.iterrows()):
            _status_card(cols[i % len(cols)], row["brand"], row,
                         stock_input[row["brand"]])
        st.divider()
        st.caption("Ringkasan saran (bahasa awam):")
        for _, row in sub.iterrows():
            st.write("• " + R.recommend_text(store, row["brand"],
                     stock_input[row["brand"]], row["reorder_point"],
                     row["order_up_to_level"]))

    # --- Subsistem Model ---
    with tab_model:
        st.subheader(f"Aktual vs Ramalan tervalidasi — {store}")
        brand = st.selectbox("Pilih merek", list(sub["brand"]), key="model_brand")
        _forecast_chart(art["preds"], store, brand)
        brow = sub[sub["brand"] == brand].iloc[0]
        st.caption(R.forecast_label({"forecast_week": brow["forecast_week"],
                                     "as_of": brow["as_of"]}))

    # --- Subsistem Data ---
    with tab_data:
        st.subheader("Parameter inventori (Tahap 8)")
        st.dataframe(art["params"].round(2), use_container_width=True, hide_index=True)
        st.subheader("Dampak biaya antar-algoritma (simulasi order-up-to)")
        st.dataframe(art["cost"].round(3), use_container_width=True, hide_index=True)
        st.caption("Harga per unit dinormalisasi = 1; holding cost = fraksi harga/unit/minggu.")


if __name__ == "__main__":
    render(_config_path())
