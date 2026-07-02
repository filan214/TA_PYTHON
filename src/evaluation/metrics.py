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

**MASE (D10, backfill 9b):** MASE = MAE_model / MAE_naive, dihitung **per deret**
dengan denominator = MAE baseline naif one-step (Tahap 6d, rezim walk-forward D9 yang
sama). MASE<1 berarti model mengungguli asumsi sederhana pemilik toko ("minggu ini =
minggu lalu"); skala-bebas & tak meledak di nilai nol (tidak seperti MAPE/sMAPE). Naif
sendiri -> MASE=1 secara konstruksi. **Catatan interpretasi (D7/D10):** sMAPE ~84% aktual
mendekati lantai teoretis ~65–70% untuk data hitung diskrit λ≈1,9 — perbaikan yang dikejar
adalah MAE/MASE, BUKAN menurunkan sMAPE ke <15% (mustahil secara struktural).

Orkestrator `run_evaluation()` menghasilkan (lih. Tahap 7 DoD, direvisi D10/D11):
    reports/results/metrics_summary.csv        (algo,varian) × metrik + MASE, + baris naif
    reports/results/metrics_per_series.csv      3 algo × 2 varian × 20 deret (+ MASE)
    reports/results/dm_tests.csv                uji DM (4 kelompok: antar-algo, ablation GT,
                                                vs-naif D10, kandidat-akurasi D11)
    reports/results/gt_ablation_comparison.csv  ablation GT per deret + ringkasan
    reports/results/smape_theoretical_floor.csv lantai teoretis sMAPE (D10) — konteks Bab IV
    reports/results/accuracy_improvement_verdict.csv  verdict kandidat 6e (D11)
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
NAIVE_METHODS = ["naive", "snaive"]

# Kandidat perbaikan akurasi Tahap 6e (D11) — label & nama file prediksi.
CANDIDATE_SPECS = [
    ("RF-Poisson(baseline)", "predictions_rf_poisson_baseline.parquet"),
    ("RF-Poisson(gt)", "predictions_rf_poisson_gt.parquet"),
    ("HGB-Poisson(baseline)", "predictions_hgb_poisson_baseline.parquet"),
    ("HGB-Poisson(gt)", "predictions_hgb_poisson_gt.parquet"),
    ("Croston", "predictions_croston.parquet"),
    ("TSB", "predictions_tsb.parquet"),
    ("Ensemble-mean(baseline)", "predictions_ensemble_mean_baseline.parquet"),
    ("Ensemble-mean(gt)", "predictions_ensemble_mean_gt.parquet"),
    ("Ensemble-invMAE(baseline)", "predictions_ensemble_invmae_baseline.parquet"),
    ("Ensemble-invMAE(gt)", "predictions_ensemble_invmae_gt.parquet"),
]


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


def mase(y_true, y_pred, naive_mae: float | None) -> float:
    """MASE = MAE model / MAE naive (denominator per deret, D10). NaN bila denom tak valid.

    MASE<1 -> model unggul atas baseline naif one-step (walk-forward D9 yang sama).
    """
    if naive_mae is None or not np.isfinite(naive_mae) or naive_mae <= 0:
        return float("nan")
    return float(mae(y_true, y_pred) / naive_mae)


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


def series_metrics(y_true, y_pred, naive_mae: float | None = None) -> dict:
    a = np.asarray(y_true, float)
    return {
        "MAE": mae(a, y_pred), "RMSE": rmse(a, y_pred),
        "MAPE": mape(a, y_pred), "sMAPE": smape(a, y_pred),
        "MASE": mase(a, y_pred, naive_mae),
        "n": int(a.size), "n_nonzero": int((a != 0).sum()),
        "volume": float(a.sum()),
    }


# --- Muat prediksi ----------------------------------------------------------

def load_predictions(results_dir: str | Path) -> dict[tuple[str, str], pd.DataFrame]:
    """Muat 6 file prediksi model utama -> {(algo, variant): df}."""
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


def load_naive_predictions(results_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Muat baseline naif (D10) -> {method: df}. Kosong bila belum ada."""
    results_dir = Path(results_dir)
    out: dict[str, pd.DataFrame] = {}
    for method in NAIVE_METHODS:
        path = results_dir / f"predictions_{method}.parquet"
        if path.exists():
            out[method] = pd.read_parquet(path)
    return out


def load_candidate_predictions(results_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Muat kandidat perbaikan akurasi Tahap 6e (D11) -> {label: df}. Kosong bila belum ada."""
    results_dir = Path(results_dir)
    out: dict[str, pd.DataFrame] = {}
    for label, fname in CANDIDATE_SPECS:
        path = results_dir / fname
        if path.exists():
            out[label] = pd.read_parquet(path)
    return out


def naive_mae_lookup(naive_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """MAE naif per deret -> denominator MASE (D10)."""
    out: dict[tuple[str, str], float] = {}
    for (store, brand), g in naive_df.groupby(["store", "brand"]):
        out[(store, brand)] = mae(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
    return out


def per_series_metrics(preds: dict[tuple[str, str], pd.DataFrame],
                       naive_mae: dict[tuple[str, str], float] | None = None) -> pd.DataFrame:
    """Tabel panjang: satu baris per (algo, varian, store, brand). MASE bila naive_mae ada."""
    rows = []
    for (algo, variant), df in preds.items():
        for (store, brand), g in df.groupby(["store", "brand"]):
            g = g.sort_values("week_start")
            nm = None if naive_mae is None else naive_mae.get((store, brand))
            m = series_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy(), naive_mae=nm)
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
            "MASE_mean": _mean(g["MASE"]) if "MASE" in g else float("nan"),
            "MAE_wmean": _wmean(g["MAE"], w), "RMSE_wmean": _wmean(g["RMSE"], w),
            "MAPE_wmean": _wmean(g["MAPE"], w), "sMAPE_wmean": _wmean(g["sMAPE"], w),
            "MASE_wmean": _wmean(g["MASE"], w) if "MASE" in g else float("nan"),
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


def baseline_summary(baseline_preds: dict[str, pd.DataFrame],
                     naive_mae: dict[tuple[str, str], float]) -> pd.DataFrame:
    """Baris ringkasan untuk baseline naif (D10) — format kolom sama dgn summary utama.

    Naif sendiri -> MASE_mean=1 secara konstruksi (denominator = MAE naif). is_best=False:
    baseline BUKAN kandidat model terbaik, hanya pembanding (D10).
    """
    rows = []
    for name, df in baseline_preds.items():
        per = []
        for (store, brand), g in df.groupby(["store", "brand"]):
            g = g.sort_values("week_start")
            per.append(series_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy(),
                                      naive_mae=naive_mae.get((store, brand))))
        pdf = pd.DataFrame(per)
        w = pdf["volume"].to_numpy()
        rows.append({
            "algo": name, "variant": "-", "n_series": int(len(pdf)),
            "MAE_mean": _mean(pdf["MAE"]), "RMSE_mean": _mean(pdf["RMSE"]),
            "MAPE_mean": _mean(pdf["MAPE"]), "sMAPE_mean": _mean(pdf["sMAPE"]),
            "MASE_mean": _mean(pdf["MASE"]),
            "MAE_wmean": _wmean(pdf["MAE"], w), "RMSE_wmean": _wmean(pdf["RMSE"], w),
            "MAPE_wmean": _wmean(pdf["MAPE"], w), "sMAPE_wmean": _wmean(pdf["sMAPE"], w),
            "MASE_wmean": _wmean(pdf["MASE"], w), "is_best": False,
        })
    return pd.DataFrame(rows)


def smape_theoretical_floor(preds: dict[tuple[str, str], pd.DataFrame], best_key: tuple[str, str],
                            seed: int = 42, n_sim: int = 50000) -> pd.DataFrame:
    """Lantai teoretis sMAPE per deret (D10, konteks Bab IV — BUKAN kriteria lulus).

    Untuk tiap deret: λ = rata-rata aktual uji. Simulasikan peramal *oracle* yang tahu
    persis λ (ramal konstan λ) terhadap aktual ~ Poisson(λ). sMAPE hasil simulasi = lantai
    struktural yang tak terhindari untuk data hitung diskrit bervolume rendah, dibandingkan
    dengan sMAPE aktual model terbaik. Menunjukkan sMAPE ~84% ≈ mendekati lantai, bukan
    indikasi model buruk.
    """
    rng = np.random.default_rng(seed)
    best_df = preds[best_key]
    rows = []
    for (store, brand), g in best_df.groupby(["store", "brand"]):
        g = g.sort_values("week_start")
        a = g["y_true"].to_numpy()
        lam = float(a.mean())
        sim = rng.poisson(lam, size=n_sim).astype(float) if lam > 0 else np.zeros(n_sim)
        floor = smape(sim, np.full(n_sim, lam))
        rows.append({
            "store": store, "brand": brand, "lambda_test": round(lam, 4),
            "smape_floor_oracle": round(floor, 2),
            "smape_actual_best": round(smape(a, g["y_pred"].to_numpy()), 2),
        })
    df = pd.DataFrame(rows)
    df["gap_vs_floor"] = (df["smape_actual_best"] - df["smape_floor_oracle"]).round(2)
    return df


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
    """Jalankan Tahap 7 penuh + backfill D10/D11: metrik+MASE, DM (4 kelompok),
    ablation GT, lantai sMAPE, verdict kandidat akurasi, figur."""
    from src.evaluation.diebold_mariano import (build_gt_ablation, compare_candidates,
                                                dm_from_frames, run_comparisons)

    preds = load_predictions(cfg.paths.results)
    if not preds:
        raise FileNotFoundError(
            f"Tak ada file prediksi di {cfg.paths.results}. Jalankan Tahap 6 dulu.")

    # D10: baseline naif -> denominator MASE (pakai naive random-walk sbg denominator standar)
    naive_preds = load_naive_predictions(cfg.paths.results)
    naive_mae = naive_mae_lookup(naive_preds["naive"]) if "naive" in naive_preds else None

    per_series = per_series_metrics(preds, naive_mae)
    summary = summary_from_per_series(per_series)
    best_var = best_variant_by_mae(summary)
    dm_tests = run_comparisons(preds, best_var, naive_preds=naive_preds or None)
    ablation = build_gt_ablation(preds)

    best_row = summary[summary["is_best"]].iloc[0]
    best_key = (str(best_row["algo"]), str(best_row["variant"]))

    # D10: lantai teoretis sMAPE + ringkasan baris naif utk metrics_summary
    floor = smape_theoretical_floor(preds, best_key) if naive_mae is not None else None
    base_summary = baseline_summary(naive_preds, naive_mae) if naive_mae is not None else None

    # D11: verdict kandidat perbaikan akurasi (kelompok DM ke-4)
    candidate_preds = load_candidate_predictions(cfg.paths.results)
    verdict, final_winner = None, best_key
    if candidate_preds:
        dm4, verdict = compare_candidates(preds, best_key, candidate_preds, naive_mae)
        dm_tests = pd.concat([dm_tests, dm4], ignore_index=True)
        adopted = verdict[verdict["signif_better_than_old"]]
        if len(adopted):  # pemenang final berganti hanya bila ada kandidat signifikan lebih baik
            final_winner = adopted.sort_values("MAE_mean").iloc[0]["candidate"]

    if save:
        rdir = cfg.paths.results
        rdir.mkdir(parents=True, exist_ok=True)
        per_series.to_csv(rdir / "metrics_per_series.csv", index=False)
        summary_out = (pd.concat([summary, base_summary], ignore_index=True)
                       if base_summary is not None else summary)
        summary_out.to_csv(rdir / "metrics_summary.csv", index=False)
        dm_tests.to_csv(rdir / "dm_tests.csv", index=False)
        ablation.to_csv(rdir / "gt_ablation_comparison.csv", index=False)
        if floor is not None:
            floor.to_csv(rdir / "smape_theoretical_floor.csv", index=False)
        if verdict is not None:
            verdict.to_csv(rdir / "accuracy_improvement_verdict.csv", index=False)
        reps = representative_series(preds)
        for variant in VARIANTS:
            plot_actual_vs_pred(preds, reps, variant, cfg.paths.figures)

    # Model terbaik vs runner-up + vs naif (kriteria D7/D10: MASE<1 & signifikan vs naif)
    ranked = summary.sort_values("MAE_mean").reset_index(drop=True)
    runner = None
    if len(ranked) > 1:
        r = ranked.iloc[1]
        runner_key = (str(r["algo"]), str(r["variant"]))
        dm = dm_from_frames(preds[best_key], preds[runner_key])
        runner = {"model": f"{ALGO_LABEL[runner_key[0]]}({runner_key[1]})",
                  "dm_stat": dm["dm_stat"], "p_value": dm["p_value"],
                  "best_sig_better": bool(dm["p_value"] < 0.05 and dm["better"] == 1)}

    best_mase = float(best_row["MASE_mean"]) if "MASE_mean" in best_row else float("nan")
    vs_naive = None
    if naive_mae is not None and "naive" in naive_preds:
        dmn = dm_from_frames(preds[best_key], naive_preds["naive"])
        vs_naive = {"MASE_mean": best_mase, "dm_stat": dmn["dm_stat"], "p_value": dmn["p_value"],
                    "sig_better": bool(dmn["p_value"] < 0.05 and dmn["better"] == 1),
                    "meets_criteria": bool(best_mase < 1 and dmn["p_value"] < 0.05 and dmn["better"] == 1)}

    return {"per_series": per_series, "summary": summary, "dm_tests": dm_tests,
            "ablation": ablation, "best_model": best_key, "best_variant": best_var,
            "runner_up": runner, "smape_floor": floor, "verdict": verdict,
            "final_winner": final_winner, "vs_naive": vs_naive}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Tahap 7 — Evaluasi + DM + ablation GT + MASE (D10) + verdict (D11)")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)

    from src.config import load_config

    cfg = load_config(args.config)
    cfg.paths.ensure()
    res = run_evaluation(cfg)

    algo, variant = res["best_model"]
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    cols = ["algo", "variant", "MAE_mean", "RMSE_mean", "sMAPE_mean", "MASE_mean", "is_best"]
    print("\n=== metrics_summary (mean antar-deret) ===")
    print(res["summary"][cols].to_string(index=False))
    print("\n=== Diebold-Mariano ===")
    print(res["dm_tests"].to_string(index=False))
    print(f"\nModel terbaik (MAE_mean): {ALGO_LABEL[algo]}({variant})")
    if res["runner_up"]:
        ru = res["runner_up"]
        verdict = "signifikan lebih baik" if ru["best_sig_better"] else "TIDAK signifikan berbeda"
        print(f"  vs runner-up {ru['model']}: DM={ru['dm_stat']:.3f} p={ru['p_value']:.4f} "
              f"-> {verdict} (alpha=0.05)")
    if res["vs_naive"]:
        vn = res["vs_naive"]
        ok = "MEMENUHI" if vn["meets_criteria"] else "TIDAK memenuhi"
        print(f"  Kriteria D7/D10 (MASE<1 & signifikan vs naive): {ok} "
              f"(MASE={vn['MASE_mean']:.3f}, vs-naive DM={vn['dm_stat']:.3f} p={vn['p_value']:.4f})")
    if res["verdict"] is not None:
        print("\n=== Verdict kandidat perbaikan akurasi (D11) ===")
        print(res["verdict"].to_string(index=False))
        print(f"\nPemenang FINAL (berganti hanya bila kandidat signifikan lebih baik): {res['final_winner']}")


if __name__ == "__main__":
    main()
