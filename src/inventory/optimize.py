"""Tahap 8 — Optimasi Inventori (Bab III §3.1.8, keputusan desain D8).

Parameter pengadaan stok dihitung dari **galat peramalan one-step-ahead
walk-forward (D9)** model terbaik Tahap 7 (bukan σ permintaan historis — D8,
pendekatan Prak dkk. [23]). Rezim one-step wajib: formula safety stock di bawah
menskalakan σ galat 1-periode dengan sqrt(lead_time); galat multi-step akan
terakumulasi berbeda dan membuat perhitungan salah secara konseptual.

Formula (per deret gerai×merek):
    safety_stock (SS) = z · σ_galat · sqrt(L)            z = Φ⁻¹(service_level)
    reorder_point (ROP) = μ_minggu · L + SS               L = lead_time_weeks
    order_up_to_level (OUL) = ROP + μ_minggu · R          R = review_period_weeks
dengan μ_minggu = rata-rata permintaan aktual mingguan pada periode uji.

Analisis dampak biaya membandingkan **ketiga algoritma** (varian terbaik masing-
masing dari Tahap 7) lewat simulasi kebijakan periodic-review order-up-to (base
stock) pada 37 minggu uji: hitung total holding cost & jumlah minggu stockout.
Harga per unit dinormalisasi = 1 (holding_cost = fraksi harga/unit/minggu, asumsi
config); biaya dilaporkan dalam satuan harga-unit. Sensitivitas: variasi
service_level & holding_cost pada grid kecil.

Output:
    reports/results/inventory_params.csv   per deret: σ, μ, SS, ROP, OUL (model terbaik)
    reports/results/cost_impact.csv        per algoritma: holding cost, stockout, fill-rate
    reports/results/inventory_sensitivity.csv  grid service_level × holding_cost (model terbaik)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Grid sensitivitas (D8): variasi kecil di sekitar asumsi config.
SERVICE_LEVELS = (0.90, 0.95, 0.99)
HOLDING_COSTS = (0.01, 0.02, 0.05)

ALGO_LABEL = {"sarimax": "SARIMAX", "rf": "RF", "lstm": "LSTM"}


# --- Formula parameter inventori --------------------------------------------

def z_from_service_level(service_level: float) -> float:
    """Faktor keamanan z = Φ⁻¹(service_level) (kuantil normal baku)."""
    return float(stats.norm.ppf(service_level))


def forecast_error_sigma(y_true, y_pred) -> float:
    """σ galat peramalan one-step (D9) = simpangan baku sampel residual (ddof=1).

    0.0 bila <2 titik atau ramalan sempurna. Residual = aktual − ramalan.
    """
    resid = np.asarray(y_true, float) - np.asarray(y_pred, float)
    if resid.size < 2:
        return 0.0
    return float(np.std(resid, ddof=1))


def safety_stock(sigma: float, z: float, lead_time_weeks: float) -> float:
    """SS = z · σ_galat · sqrt(L)."""
    return float(z * sigma * np.sqrt(lead_time_weeks))


def reorder_point(mean_weekly_demand: float, lead_time_weeks: float, ss: float) -> float:
    """ROP = μ_minggu · L + SS (permintaan rata-rata sepanjang lead time + safety stock)."""
    return float(mean_weekly_demand * lead_time_weeks + ss)


def order_up_to_level(rop: float, mean_weekly_demand: float,
                      review_period_weeks: float) -> float:
    """OUL = ROP + μ_minggu · R (tambah permintaan selama periode review, sesuai [16])."""
    return float(rop + mean_weekly_demand * review_period_weeks)


def series_params(y_true, y_pred, z: float, lead_time_weeks: float,
                  review_period_weeks: float) -> dict:
    """Parameter inventori satu deret dari galat one-step + rata-rata permintaan uji."""
    sigma = forecast_error_sigma(y_true, y_pred)
    mu = float(np.mean(np.asarray(y_true, float)))
    ss = safety_stock(sigma, z, lead_time_weeks)
    rop = reorder_point(mu, lead_time_weeks, ss)
    oul = order_up_to_level(rop, mu, review_period_weeks)
    return {"sigma_error": sigma, "mean_weekly_demand": mu,
            "safety_stock": ss, "reorder_point": rop, "order_up_to_level": oul}


# --- Simulasi kebijakan periodic-review order-up-to (base stock) ------------

def simulate_base_stock(demand, S: float, lead_time_weeks: int,
                        review_period_weeks: int = 1,
                        init_inventory: float | None = None) -> dict:
    """Simulasi (R,S) lost-sales pada deret permintaan aktual.

    Tiap minggu: (1) terima pesanan yang tiba, (2) penuhi permintaan (kekurangan =
    lost sale, dicatat stockout), (3) akrual holding atas on-hand akhir minggu,
    (4) tiap R minggu pesan hingga level S berdasar inventory position (on-hand +
    pipeline), tiba setelah L minggu. Mulai dengan on-hand = S (init penuh) kecuali
    `init_inventory` diberikan.
    """
    d = np.asarray(demand, float)
    T = d.size
    L = int(lead_time_weeks)
    R = max(int(review_period_weeks), 1)
    on_hand = float(S if init_inventory is None else init_inventory)
    arrivals: dict[int, float] = {}      # minggu tiba -> kuantitas
    outstanding = 0.0                    # total pesanan dalam perjalanan (pipeline)

    stockout_weeks = 0
    shortage_units = 0.0
    holding_unit_weeks = 0.0

    for t in range(T):
        if t in arrivals:                # (1) penerimaan
            recv = arrivals.pop(t)
            on_hand += recv
            outstanding -= recv
        sales = min(on_hand, d[t])       # (2) pemenuhan permintaan (lost sales)
        shortage = d[t] - sales
        on_hand -= sales
        if shortage > 1e-9:
            stockout_weeks += 1
            shortage_units += shortage
        holding_unit_weeks += on_hand    # (3) holding on-hand akhir minggu
        if t % R == 0:                   # (4) review & pemesanan
            ip = on_hand + outstanding
            order = max(0.0, S - ip)
            if order > 1e-9:
                arrivals[t + L] = arrivals.get(t + L, 0.0) + order
                outstanding += order

    demand_total = float(d.sum())
    fill_rate = 1.0 if demand_total <= 0 else (demand_total - shortage_units) / demand_total
    return {"n_weeks": T, "stockout_weeks": stockout_weeks,
            "shortage_units": float(shortage_units), "demand_total": demand_total,
            "holding_unit_weeks": float(holding_unit_weeks),
            "fill_rate": float(fill_rate),
            "service_level_ach": float(1.0 - stockout_weeks / T) if T else float("nan")}


# --- Tabel keluaran ---------------------------------------------------------

def inventory_params_table(preds: pd.DataFrame, service_level: float,
                           lead_time_weeks: float, review_period_weeks: float) -> pd.DataFrame:
    """Parameter inventori per deret (model terbaik) — satu baris per gerai×merek."""
    z = z_from_service_level(service_level)
    rows = []
    for (store, brand), g in preds.groupby(["store", "brand"]):
        g = g.sort_values("week_start")
        p = series_params(g["y_true"].to_numpy(), g["y_pred"].to_numpy(),
                          z, lead_time_weeks, review_period_weeks)
        rows.append({"store": store, "brand": brand, "z": z, **p})
    return pd.DataFrame(rows).sort_values(["store", "brand"]).reset_index(drop=True)


def _simulate_series(g: pd.DataFrame, z: float, lead_time_weeks: float,
                     review_period_weeks: float) -> dict:
    """Params + simulasi untuk satu deret; kembalikan SS & metrik simulasi."""
    g = g.sort_values("week_start")
    yt = g["y_true"].to_numpy()
    yp = g["y_pred"].to_numpy()
    p = series_params(yt, yp, z, lead_time_weeks, review_period_weeks)
    sim = simulate_base_stock(yt, S=p["order_up_to_level"],
                              lead_time_weeks=int(lead_time_weeks),
                              review_period_weeks=int(review_period_weeks))
    return {**p, **sim}


def cost_impact_table(preds_by_algo: dict[str, pd.DataFrame], service_level: float,
                      lead_time_weeks: float, review_period_weeks: float,
                      holding_cost: float) -> pd.DataFrame:
    """Dampak biaya per algoritma: agregasi simulasi order-up-to antar-deret.

    Tiap algoritma memakai OUL dari σ galat-nya sendiri; permintaan aktual identik,
    jadi perbedaan biaya/stockout murni berasal dari akurasi ramalan (D8).
    """
    z = z_from_service_level(service_level)
    rows = []
    for algo, preds in preds_by_algo.items():
        ss_list, hold_uw, stockouts, shortage, dem_tot, n_weeks = [], 0.0, 0, 0.0, 0.0, 0
        for _, g in preds.groupby(["store", "brand"]):
            r = _simulate_series(g, z, lead_time_weeks, review_period_weeks)
            ss_list.append(r["safety_stock"])
            hold_uw += r["holding_unit_weeks"]
            stockouts += r["stockout_weeks"]
            shortage += r["shortage_units"]
            dem_tot += r["demand_total"]
            n_weeks += r["n_weeks"]
        fill = 1.0 if dem_tot <= 0 else (dem_tot - shortage) / dem_tot
        rows.append({
            "algo": algo,
            "safety_stock_mean": float(np.mean(ss_list)),
            "holding_unit_weeks": float(hold_uw),
            "holding_cost_total": float(holding_cost * hold_uw),
            "stockout_weeks": int(stockouts),
            "shortage_units": float(shortage),
            "fill_rate": float(fill),
            "service_level_ach": float(1.0 - stockouts / n_weeks) if n_weeks else float("nan"),
        })
    return pd.DataFrame(rows)


def sensitivity_table(best_preds: pd.DataFrame, lead_time_weeks: float,
                      review_period_weeks: float, holding_cost_base: float,
                      service_levels=SERVICE_LEVELS,
                      holding_costs=HOLDING_COSTS) -> pd.DataFrame:
    """Sensitivitas SS & biaya model terbaik terhadap grid service_level × holding_cost."""
    rows = []
    for sl in service_levels:
        params = inventory_params_table(best_preds, sl, lead_time_weeks, review_period_weeks)
        # simulasi memakai OUL pada service_level ini (holding_cost hanya penskala biaya)
        z = z_from_service_level(sl)
        hold_uw, stockouts, shortage, dem_tot, n_weeks = 0.0, 0, 0.0, 0.0, 0
        for _, g in best_preds.groupby(["store", "brand"]):
            r = _simulate_series(g, z, lead_time_weeks, review_period_weeks)
            hold_uw += r["holding_unit_weeks"]; stockouts += r["stockout_weeks"]
            shortage += r["shortage_units"]; dem_tot += r["demand_total"]; n_weeks += r["n_weeks"]
        fill = 1.0 if dem_tot <= 0 else (dem_tot - shortage) / dem_tot
        for hc in holding_costs:
            rows.append({
                "service_level": sl, "holding_cost": hc,
                "safety_stock_mean": float(params["safety_stock"].mean()),
                "order_up_to_level_mean": float(params["order_up_to_level"].mean()),
                "holding_cost_total": float(hc * hold_uw),
                "stockout_weeks": int(stockouts), "fill_rate": float(fill),
            })
    return pd.DataFrame(rows)


# --- Orkestrasi -------------------------------------------------------------

def _load_predictions(results_dir: Path) -> dict[tuple[str, str], pd.DataFrame]:
    from src.evaluation.metrics import load_predictions
    return load_predictions(results_dir)


def _best_variant_per_algo(results_dir: Path) -> dict[str, str]:
    """Varian terbaik per algoritma dari metrics_summary.csv (Tahap 7). Fallback 'gt'."""
    path = results_dir / "metrics_summary.csv"
    if not path.exists():
        return {a: "gt" for a in ALGO_LABEL}
    summ = pd.read_csv(path)
    summ = summ[summ["variant"].isin(["baseline", "gt"])]
    best = {}
    for algo in ALGO_LABEL:
        sub = summ[summ["algo"] == algo]
        if len(sub):
            best[algo] = str(sub.loc[sub["MAE_mean"].idxmin(), "variant"])
    return best


def _final_winner_key(results_dir: Path, best_per_algo: dict[str, str]) -> tuple[str, str]:
    """(algo, varian) pemenang FINAL: algoritma dgn MAE_mean terendah (D7/D11)."""
    path = results_dir / "metrics_summary.csv"
    if not path.exists():
        return ("rf", best_per_algo.get("rf", "gt"))
    summ = pd.read_csv(path)
    summ = summ[summ["variant"].isin(["baseline", "gt"])]
    row = summ.loc[summ["MAE_mean"].idxmin()]
    return (str(row["algo"]), str(row["variant"]))


def run_inventory(cfg, save: bool = True) -> dict:
    """Hitung parameter inventori (model terbaik) + dampak biaya 3 algoritma + sensitivitas."""
    results_dir = cfg.paths.results
    preds = _load_predictions(results_dir)
    if not preds:
        raise FileNotFoundError(
            f"Tak ada file prediksi di {results_dir}. Jalankan Tahap 6–7 dulu.")

    L = float(cfg["lead_time_weeks"])
    R = float(cfg["review_period_weeks"])
    service_level = float(cfg["service_level"])
    holding_cost = float(cfg["holding_cost"])

    best_per_algo = _best_variant_per_algo(results_dir)
    winner_algo, winner_variant = _final_winner_key(results_dir, best_per_algo)
    best_preds = preds[(winner_algo, winner_variant)]

    params = inventory_params_table(best_preds, service_level, L, R)
    params.insert(0, "model", f"{ALGO_LABEL.get(winner_algo, winner_algo)}({winner_variant})")

    preds_by_algo = {
        ALGO_LABEL[algo]: preds[(algo, best_per_algo.get(algo, "gt"))]
        for algo in ALGO_LABEL if (algo, best_per_algo.get(algo, "gt")) in preds
    }
    cost = cost_impact_table(preds_by_algo, service_level, L, R, holding_cost)
    cost.insert(1, "variant", [best_per_algo.get(a, "gt")
                               for a in [k.lower() for k in cost["algo"]]])

    sens = sensitivity_table(best_preds, L, R, holding_cost)

    if save:
        results_dir.mkdir(parents=True, exist_ok=True)
        params.to_csv(results_dir / "inventory_params.csv", index=False)
        cost.to_csv(results_dir / "cost_impact.csv", index=False)
        sens.to_csv(results_dir / "inventory_sensitivity.csv", index=False)
        logger.info("Inventori tersimpan: inventory_params.csv (%d deret), "
                    "cost_impact.csv (%d algo), inventory_sensitivity.csv",
                    len(params), len(cost))

    return {"winner": (winner_algo, winner_variant), "params": params,
            "cost_impact": cost, "sensitivity": sens}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Tahap 8 — optimasi inventori (D8)")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)

    from src.config import load_config

    cfg = load_config(args.config)
    cfg.paths.ensure()
    res = run_inventory(cfg)

    algo, variant = res["winner"]
    print(f"\nTahap 8 — Optimasi Inventori (D8). Model terbaik = "
          f"{ALGO_LABEL.get(algo, algo)}({variant})")
    print(f"service_level={cfg['service_level']}  lead_time={cfg['lead_time_weeks']}mgg  "
          f"review={cfg['review_period_weeks']}mgg  holding_cost={cfg['holding_cost']}")
    p = res["params"]
    print(f"\nParameter inventori ({len(p)} deret) — ringkasan:")
    print(f"  SS  rata2 {p['safety_stock'].mean():.2f}  (min {p['safety_stock'].min():.2f}, "
          f"maks {p['safety_stock'].max():.2f})")
    print(f"  ROP rata2 {p['reorder_point'].mean():.2f}   OUL rata2 {p['order_up_to_level'].mean():.2f}")
    print("\nDampak biaya antar-algoritma (simulasi order-up-to pada minggu uji):")
    cols = ["algo", "variant", "safety_stock_mean", "holding_cost_total",
            "stockout_weeks", "fill_rate"]
    print(res["cost_impact"][cols].to_string(index=False,
          formatters={"safety_stock_mean": "{:.3f}".format,
                      "holding_cost_total": "{:.3f}".format,
                      "fill_rate": "{:.3f}".format}))


if __name__ == "__main__":
    main()
