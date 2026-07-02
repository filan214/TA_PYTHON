"""Tahap 7 — Metrik evaluasi & agregasi (Bab III §3.1.7, D7/D9).

Semua metrik dihitung dari prediksi **one-step-ahead walk-forward (D9)** yang
tersimpan di `reports/results/predictions_<algo>_<variant>.parquet`
(kolom: store, brand, week_start, y_true, y_pred), 20 deret × 37 minggu uji.

Penanganan minggu nol (§9): deret gerai×merek ~26% minggu bernilai nol, sehingga
**MAPE murni meledak** (pembagian nol). Keputusan terdokumentasi:
    - MAPE dihitung HANYA pada minggu dengan aktual != 0 (di-mask), dalam persen.
      Jika sebuah deret tak punya minggu non-nol, MAPE = NaN dan dikecualikan dari
      agregasi. Jumlah minggu non-nol dilaporkan (`n_nonzero`).
    - sMAPE dilaporkan sebagai pendamping yang aman: 2|p-a|/(|a|+|p|), dengan
      konvensi 0/0 -> 0 (kedua nilai nol = tepat). Dalam persen, rentang [0,200].
    - RMSE & MAE selalu terdefinisi; jadi metrik utama pembanding pada data ber-nol.

Agregasi per (algoritma, varian): rata-rata antar-deret (`_mean`) dan rata-rata
berbobot volume (`_wmean`, bobot = total unit aktual uji per deret).

Orkestrator `run_evaluation()` menghasilkan (lih. Tahap 7 DoD):
    reports/results/metrics_summary.csv        (algo,varian) × metrik mean+weighted
    reports/results/metrics_per_series.csv      3 algo × 2 varian × 20 deret
    reports/results/dm_tests.csv                uji Diebold-Mariano (2 kelompok)
    reports/results/gt_ablation_comparison.csv  ablation GT per deret + ringkasan
    reports/figures/actual_vs_pred_<store>_<brand>_<variant>.png (deret representatif)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

ALGOS = ["sarimax", "rf", "lstm"]
VARIANTS = ["baseline", "gt"]
ALGO_LABEL = {"sarimax": "SARIMAX", "rf": "RF", "lstm": "LSTM"}


# --- Metrik dasar (array masuk) --------------------------------------------

def mae(y_true, y_pred) -> float:
    a, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(a - p)))


def rmse(y_true, y_pred) -> float:
    a, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(y_true, y_pred) -> float:
    """MAPE (%) dihitung HANYA pada aktual != 0. NaN bila tak ada minggu non-nol."""
    a, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = a != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100.0)


def smape(y_true, y_pred) -> float:
    """sMAPE (%) aman pada nol: konvensi 0/0 -> 0. Rentang [0,200]."""
    a, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(a) + np.abs(p)
    num = 2.0 * np.abs(p - a)
    term = np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)
    return float(np.mean(term) * 100.0)


# --- Agregasi ---------------------------------------------------------------

def _mean(values) -> float:
    v = np.asarray(values, float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def _wmean(values, weights) -> float:
    """Rata-rata berbobot; abaikan NaN & bobot non-positif."""
    v, w = np.asarray(values, float), np.asarray(weights, float)
    m = ~np.isnan(v) & (w > 0)
    if not m.any():
        return float("nan")
    return float(np.sum(v[m] * w[m]) / np.sum(w[m]))


def series_metrics(y_true, y_pred) -> dict:
    a = np.asarray(y_true, float)
    return {
        "MAE": mae(a, y_pred), "RMSE": rmse(a, y_pred),
        "MAPE": mape(a, y_pred), "sMAPE": smape(a, y_pred),
        "n": int(a.size), "n_nonzero": int((a != 0).sum()),
        "volume": float(a.sum()),
    }


# --- Muat prediksi ----------------------------------------------------------

def load_predictions(results_dir: str | Path) -> dict[tuple[str, str], pd.DataFrame]:
    """Muat 6 file prediksi -> {(algo, variant): df}."""
    results_dir = Path(results_dir)
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for algo in ALGOS:
        for variant in VARIANTS:
            path = results_dir / f"predictions_{algo}_{variant}.parquet"
            if path.exists():
                out[(algo, variant)] = pd.read_parquet(path)
            else:
                logger.warning("Prediksi hilang: %s", path)
    return out


def per_series_metrics(preds: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Tabel panjang: satu baris per (algo, varian, store, brand)."""
    rows = []
    for (algo, variant), df in preds.items():
        for (store, brand), g in df.groupby(["store", "brand"]):
            g = g.sort_values("week_start")
            m = series_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
            rows.append({"algo": algo, "variant": variant,
                         "store": store, "brand": brand, **m})
    return pd.DataFrame(rows)


def summary_from_per_series(per_series: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan (algo, varian): metrik mean & weighted-by-volume antar-deret."""
    rows = []
    for (algo, variant), g in per_series.groupby(["algo", "variant"]):
        w = g["volume"].to_numpy()
        rows.append({
            "algo": algo, "variant": variant, "n_series": int(len(g)),
            "MAE_mean": _mean(g["MAE"]), "RMSE_mean": _mean(g["RMSE"]),
            "MAPE_mean": _mean(g["MAPE"]), "sMAPE_mean": _mean(g["sMAPE"]),
            "MAE_wmean": _wmean(g["MAE"], w), "RMSE_wmean": _wmean(g["RMSE"], w),
            "MAPE_wmean": _wmean(g["MAPE"], w), "sMAPE_wmean": _wmean(g["sMAPE"], w),
        })
    summary = pd.DataFrame(rows)
    # urutkan konsisten (algo lalu varian) & tandai model terbaik (MAE_mean terendah)
    summary["algo"] = pd.Categorical(summary["algo"], ALGOS, ordered=True)
    summary["variant"] = pd.Categorical(summary["variant"], VARIANTS, ordered=True)
    summary = summary.sort_values(["algo", "variant"]).reset_index(drop=True)
    summary["algo"] = summary["algo"].astype(str)
    summary["variant"] = summary["variant"].astype(str)
    summary["is_best"] = summary["MAE_mean"] == summary["MAE_mean"].min()
    return summary


def best_variant_by_mae(summary: pd.DataFrame) -> dict[str, str]:
    """Varian terbaik per algoritma (MAE_mean terendah) — untuk perbandingan antar-algo."""
    best = {}
    for algo in ALGOS:
        sub = summary[summary["algo"] == algo]
        if len(sub):
            best[algo] = str(sub.loc[sub["MAE_mean"].idxmin(), "variant"])
    return best


# --- Figur aktual vs prediksi ----------------------------------------------

def representative_series(preds: dict[tuple[str, str], pd.DataFrame], n: int = 6) -> list[tuple[str, str]]:
    """N deret volume-tertinggi (paling informatif untuk plot). Deterministik."""
    df = next(iter(preds.values()))
    vol = (df.groupby(["store", "brand"])["y_true"].sum()
           .sort_values(ascending=False))
    return [tuple(k) for k in vol.head(n).index]


def plot_actual_vs_pred(preds, series_keys, variant, figures_dir: str | Path) -> list[Path]:
    """Overlay aktual + prediksi ketiga algoritma (varian tsb) per deret representatif."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for store, brand in series_keys:
        fig, ax = plt.subplots(figsize=(10, 4))
        drawn_actual = False
        for algo in ALGOS:
            df = preds.get((algo, variant))
            if df is None:
                continue
            g = df[(df["store"] == store) & (df["brand"] == brand)].sort_values("week_start")
            if not drawn_actual:
                ax.plot(g["week_start"], g["y_true"], color="black", lw=2,
                        marker="o", ms=3, label="Aktual")
                drawn_actual = True
            ax.plot(g["week_start"], g["y_pred"], lw=1.4, alpha=0.85,
                    label=f"{ALGO_LABEL[algo]}")
        ax.set_title(f"Aktual vs Prediksi — {store} × {brand} ({variant}, one-step D9)")
        ax.set_xlabel("Minggu")
        ax.set_ylabel("Unit")
        ax.legend(loc="upper right", fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        out = figures_dir / f"actual_vs_pred_{store}_{brand}_{variant}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        saved.append(out)
    return saved


# --- Orkestrasi -------------------------------------------------------------

def run_evaluation(cfg, save: bool = True) -> dict:
    """Jalankan seluruh Tahap 7: metrik, DM (2 kelompok), ablation GT, figur."""
    from src.evaluation.diebold_mariano import build_gt_ablation, dm_from_frames, run_comparisons

    preds = load_predictions(cfg.paths.results)
    if not preds:
        raise FileNotFoundError(
            f"Tak ada file prediksi di {cfg.paths.results}. Jalankan Tahap 6 dulu.")

    per_series = per_series_metrics(preds)
    summary = summary_from_per_series(per_series)
    best_var = best_variant_by_mae(summary)
    dm_tests = run_comparisons(preds, best_var)
    ablation = build_gt_ablation(preds)

    if save:
        rdir = cfg.paths.results
        rdir.mkdir(parents=True, exist_ok=True)
        per_series.to_csv(rdir / "metrics_per_series.csv", index=False)
        summary.to_csv(rdir / "metrics_summary.csv", index=False)
        dm_tests.to_csv(rdir / "dm_tests.csv", index=False)
        ablation.to_csv(rdir / "gt_ablation_comparison.csv", index=False)
        reps = representative_series(preds)
        for variant in VARIANTS:
            plot_actual_vs_pred(preds, reps, variant, cfg.paths.figures)

    # Model terbaik + apakah signifikan vs runner-up (pada varian terbaik masing-masing)
    best_row = summary[summary["is_best"]].iloc[0]
    best_key = (str(best_row["algo"]), str(best_row["variant"]))
    ranked = summary.sort_values("MAE_mean").reset_index(drop=True)
    runner = None
    if len(ranked) > 1:
        r = ranked.iloc[1]
        runner_key = (str(r["algo"]), str(r["variant"]))
        dm = dm_from_frames(preds[best_key], preds[runner_key])
        runner = {"model": f"{ALGO_LABEL[runner_key[0]]}({runner_key[1]})",
                  "dm_stat": dm["dm_stat"], "p_value": dm["p_value"],
                  "best_sig_better": bool(dm["p_value"] < 0.05 and dm["better"] == 1)}

    return {"per_series": per_series, "summary": summary, "dm_tests": dm_tests,
            "ablation": ablation, "best_model": best_key, "best_variant": best_var,
            "runner_up": runner}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Tahap 7 — Evaluasi + Diebold-Mariano + ablation GT")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)

    from src.config import load_config

    cfg = load_config(args.config)
    cfg.paths.ensure()
    res = run_evaluation(cfg)

    algo, variant = res["best_model"]
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("\n=== metrics_summary (mean antar-deret) ===")
    print(res["summary"][["algo", "variant", "MAE_mean", "RMSE_mean",
                          "MAPE_mean", "sMAPE_mean", "is_best"]].to_string(index=False))
    print("\n=== Diebold-Mariano ===")
    print(res["dm_tests"].to_string(index=False))
    print(f"\nModel terbaik (MAE_mean): {ALGO_LABEL[algo]}({variant})")
    if res["runner_up"]:
        ru = res["runner_up"]
        verdict = "signifikan lebih baik" if ru["best_sig_better"] else "TIDAK signifikan berbeda"
        print(f"  vs runner-up {ru['model']}: DM={ru['dm_stat']:.3f} p={ru['p_value']:.4f} "
              f"-> {verdict} (alpha=0.05)")


if __name__ == "__main__":
    main()
