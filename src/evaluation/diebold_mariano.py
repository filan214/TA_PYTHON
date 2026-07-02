"""Tahap 7 — Uji Diebold-Mariano (Bab III §3.1.7, D7).

Uji apakah dua model punya akurasi peramalan yang berbeda secara statistik.
Loss default = squared error, one-step (h=1). Statistik memakai koreksi sampel
kecil **Harvey-Leybourne-Newbold (1997)** dan p-value dari distribusi t (df=T-1),
dua sisi — lebih tepat untuk sampel pendek (T=37 per deret) dibanding normal.

PRASYARAT VALIDITAS (D9): loss kedua model harus berasal dari rezim evaluasi yang
sama (one-step-ahead walk-forward) dan forecast origin yang **sepadan**. Karena itu
`align_frames` memverifikasi kedua frame prediksi berbagi (store, brand, week_start,
y_true) yang identik sebelum menghitung loss differential; ketaksesuaian -> error,
bukan diam-diam salah membandingkan rezim berbeda.

Dua kelompok perbandingan (lih. Tahap 7):
    1. Antar-algoritma (pada varian terbaik masing-masing): SARIMAX vs RF,
       SARIMAX vs LSTM, RF vs LSTM.
    2. Ablation GT per algoritma (baseline vs gt): menjawab langsung
       "apakah Google Trends terbukti membantu di data toko riil ini".
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

ALGOS = ["sarimax", "rf", "lstm"]
ALGO_LABEL = {"sarimax": "SARIMAX", "rf": "RF", "lstm": "LSTM"}
_KEYS = ["store", "brand", "week_start"]


def _loss(err: np.ndarray, kind: str) -> np.ndarray:
    if kind == "squared":
        return err ** 2
    if kind == "absolute":
        return np.abs(err)
    raise ValueError(f"loss tak dikenal: {kind}")


def diebold_mariano(y_true, pred1, pred2, h: int = 1, loss: str = "squared") -> dict:
    """Uji DM dua sisi dgn koreksi HLN. `better`=1 jika model1 loss lebih rendah.

    Mengembalikan dm_stat (terkoreksi HLN), p_value (t, df=T-1), mean_diff (d̄,
    negatif -> model1 unggul), n, dan `better` ∈ {1,2,0}. Kasus loss identik
    (varians 0) -> dm_stat=0, p_value=1 (tak ada beda terdeteksi).
    """
    a = np.asarray(y_true, float)
    e1 = a - np.asarray(pred1, float)
    e2 = a - np.asarray(pred2, float)
    d = _loss(e1, loss) - _loss(e2, loss)
    T = d.size
    dbar = float(d.mean())

    # Varians jangka-panjang d̄ dengan Newey-West lag = h-1 (h=1 -> hanya γ0)
    d0 = d - dbar
    gamma0 = float(np.mean(d0 * d0))
    lrv = gamma0
    for k in range(1, h):
        gamma_k = float(np.mean(d0[k:] * d0[:-k]))
        lrv += 2.0 * gamma_k
    var_dbar = lrv / T

    better = 1 if dbar < 0 else (2 if dbar > 0 else 0)
    if var_dbar <= 0 or T < 2:            # differensial loss tanpa variasi
        if dbar == 0 or T < 2:            # prediksi identik / tak bisa diuji -> seri
            return {"dm_stat": 0.0, "p_value": 1.0, "mean_diff": dbar,
                    "n": int(T), "better": 0}
        # differensial konstan tak-nol: satu model unggul deterministik di tiap titik
        return {"dm_stat": float(np.sign(dbar) * np.inf), "p_value": 0.0,
                "mean_diff": dbar, "n": int(T), "better": better}

    dm = dbar / np.sqrt(var_dbar)
    # Koreksi sampel kecil Harvey-Leybourne-Newbold (1997)
    corr = (T + 1 - 2 * h + h * (h - 1) / T) / T
    dm_star = dm * np.sqrt(max(corr, 0.0))
    p_value = float(2.0 * stats.t.cdf(-abs(dm_star), df=T - 1))
    return {"dm_stat": float(dm_star), "p_value": p_value, "mean_diff": dbar,
            "n": int(T), "better": better}


def align_frames(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Gabung dua frame prediksi pada origin sepadan; verifikasi y_true identik (D9)."""
    m = df1[_KEYS + ["y_true", "y_pred"]].merge(
        df2[_KEYS + ["y_true", "y_pred"]], on=_KEYS, suffixes=("_1", "_2"))
    if len(m) != len(df1) or len(m) != len(df2):
        raise ValueError(
            "Origin forecast tak sepadan antar-model (pelanggaran D9): "
            f"{len(df1)} & {len(df2)} baris -> {len(m)} cocok.")
    if not np.allclose(m["y_true_1"].to_numpy(), m["y_true_2"].to_numpy()):
        raise ValueError("y_true berbeda pada origin yang sama (pelanggaran D9).")
    return m.sort_values(_KEYS).reset_index(drop=True)


def dm_from_frames(df1: pd.DataFrame, df2: pd.DataFrame, h: int = 1,
                   loss: str = "squared") -> dict:
    """Uji DM pada loss ter-pool di seluruh deret×minggu, origin tervalidasi (D9)."""
    m = align_frames(df1, df2)
    return diebold_mariano(m["y_true_1"], m["y_pred_1"], m["y_pred_2"], h=h, loss=loss)


# --- Dua kelompok perbandingan ---------------------------------------------

def run_comparisons(preds: dict[tuple[str, str], pd.DataFrame],
                    best_variant: dict[str, str], loss: str = "squared",
                    naive_preds: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """dm_tests.csv: antar-algoritma + ablation GT + (opsional) vs-naif (D10).

    Bila `naive_preds` diberikan, tambah Kelompok 3: tiap (algoritma, varian terbaik)
    vs tiap baseline naif — membuktikan apakah kompleksitas model bermanfaat dibanding
    asumsi sederhana pemilik toko (D10). Tanpa `naive_preds` -> hanya 2 kelompok.
    """
    rows = []

    # Kelompok 1 — antar-algoritma pada varian terbaik masing-masing
    for a1, a2 in [("sarimax", "rf"), ("sarimax", "lstm"), ("rf", "lstm")]:
        v1, v2 = best_variant.get(a1), best_variant.get(a2)
        if (a1, v1) not in preds or (a2, v2) not in preds:
            continue
        r = dm_from_frames(preds[(a1, v1)], preds[(a2, v2)], loss=loss)
        winner = {1: f"{ALGO_LABEL[a1]}({v1})", 2: f"{ALGO_LABEL[a2]}({v2})", 0: "seri"}[r["better"]]
        rows.append({
            "group": "antar_algoritma",
            "model1": f"{ALGO_LABEL[a1]}({v1})", "model2": f"{ALGO_LABEL[a2]}({v2})",
            "dm_stat": r["dm_stat"], "p_value": r["p_value"], "n": r["n"],
            "significant": bool(r["p_value"] < 0.05),
            "better": winner if r["p_value"] < 0.05 else "tak signifikan",
        })

    # Kelompok 2 — ablation GT (baseline vs gt) per algoritma
    for algo in ALGOS:
        if (algo, "baseline") not in preds or (algo, "gt") not in preds:
            continue
        r = dm_from_frames(preds[(algo, "baseline")], preds[(algo, "gt")], loss=loss)
        winner = {1: f"{ALGO_LABEL[algo]}(baseline)", 2: f"{ALGO_LABEL[algo]}(gt)",
                  0: "seri"}[r["better"]]
        rows.append({
            "group": "ablation_gt",
            "model1": f"{ALGO_LABEL[algo]}(baseline)", "model2": f"{ALGO_LABEL[algo]}(gt)",
            "dm_stat": r["dm_stat"], "p_value": r["p_value"], "n": r["n"],
            "significant": bool(r["p_value"] < 0.05),
            "better": winner if r["p_value"] < 0.05 else "tak signifikan",
        })

    # Kelompok 3 — vs baseline naif (D10): tiap (algo, varian terbaik) vs tiap naif
    if naive_preds:
        for algo in ALGOS:
            v = best_variant.get(algo)
            if (algo, v) not in preds:
                continue
            model_label = f"{ALGO_LABEL[algo]}({v})"
            for method, ndf in naive_preds.items():
                r = dm_from_frames(preds[(algo, v)], ndf, loss=loss)
                winner = {1: model_label, 2: method, 0: "seri"}[r["better"]]
                rows.append({
                    "group": "vs_naive",
                    "model1": model_label, "model2": method,
                    "dm_stat": r["dm_stat"], "p_value": r["p_value"], "n": r["n"],
                    "significant": bool(r["p_value"] < 0.05),
                    "better": winner if r["p_value"] < 0.05 else "tak signifikan",
                })

    return pd.DataFrame(rows)


def _agg_mae_mase(df: pd.DataFrame,
                  naive_mae: dict[tuple[str, str], float] | None) -> tuple[float, float]:
    """Rata-rata antar-deret MAE & MASE untuk satu set prediksi (kolom kompatibel Tahap 7)."""
    from src.evaluation.metrics import mae as _mae, mase as _mase

    maes, mases = [], []
    for (store, brand), g in df.groupby(["store", "brand"]):
        g = g.sort_values("week_start")
        a, p = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
        maes.append(_mae(a, p))
        nm = None if naive_mae is None else naive_mae.get((store, brand))
        mases.append(_mase(a, p, nm))
    mase_arr = np.asarray(mases, float)
    mase_mean = float(np.nanmean(mase_arr)) if np.isfinite(mase_arr).any() else float("nan")
    return float(np.nanmean(maes)), mase_mean


def compare_candidates(preds: dict[tuple[str, str], pd.DataFrame], best_key: tuple[str, str],
                       candidate_preds: dict[str, pd.DataFrame],
                       naive_mae: dict[tuple[str, str], float] | None = None,
                       loss: str = "squared") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kelompok 4 (D11): pemenang lama vs tiap kandidat 6e -> (dm_rows, verdict).

    Pemenang final berganti HANYA jika kandidat signifikan lebih baik (p<0.05 & loss
    lebih rendah). Baris pertama verdict = pemenang lama sebagai referensi.
    """
    old = preds[best_key]
    old_label = f"{ALGO_LABEL[best_key[0]]}({best_key[1]})"
    old_mae, old_mase = _agg_mae_mase(old, naive_mae)

    dm_rows = []
    verdict_rows = [{
        "candidate": f"{old_label} [pemenang lama]", "MAE_mean": round(old_mae, 4),
        "MASE_mean": round(old_mase, 4), "dm_stat": np.nan,
        "p_value_vs_old_winner": np.nan, "signif_better_than_old": False,
        "decision": "REFERENSI (pemenang Tahap 7)",
    }]

    for label, cand in candidate_preds.items():
        r = dm_from_frames(old, cand, loss=loss)   # better=1 -> lama unggul, 2 -> kandidat unggul
        cand_mae, cand_mase = _agg_mae_mase(cand, naive_mae)
        sig = r["p_value"] < 0.05
        adopt = bool(sig and r["better"] == 2)
        if adopt:
            decision = "ADOPSI (signifikan lebih baik)"
        elif sig and r["better"] == 1:
            decision = "tolak (signifikan LEBIH BURUK dari pemenang lama)"
        else:
            decision = "tolak (tak berbeda signifikan)"
        dm_rows.append({
            "group": "kandidat_akurasi", "model1": old_label, "model2": label,
            "dm_stat": r["dm_stat"], "p_value": r["p_value"], "n": r["n"],
            "significant": bool(sig),
            "better": (label if adopt else old_label if (sig and r["better"] == 1) else "tak signifikan"),
        })
        verdict_rows.append({
            "candidate": label, "MAE_mean": round(cand_mae, 4), "MASE_mean": round(cand_mase, 4),
            "dm_stat": round(r["dm_stat"], 4), "p_value_vs_old_winner": round(r["p_value"], 4),
            "signif_better_than_old": adopt, "decision": decision,
        })
    return pd.DataFrame(dm_rows), pd.DataFrame(verdict_rows)


def build_gt_ablation(preds: dict[tuple[str, str], pd.DataFrame],
                      loss: str = "squared") -> pd.DataFrame:
    """gt_ablation_comparison.csv: per (algo, deret) baseline vs gt + baris ringkasan.

    Verdikt per deret dari uji DM (37 galat one-step): membaik/memburuk (signifikan
    α=0.05) atau tak signifikan. Baris ringkasan (row_type='ringkasan') mencacah
    jumlah deret membaik vs memburuk vs tak-signifikan per algoritma.
    """
    from src.evaluation.metrics import mae, mape

    rows, summary_rows = [], []
    for algo in ALGOS:
        if (algo, "baseline") not in preds or (algo, "gt") not in preds:
            continue
        b = preds[(algo, "baseline")]
        g = preds[(algo, "gt")]
        m = align_frames(b, g)
        n_better = n_worse = n_ns = 0
        for (store, brand), grp in m.groupby(["store", "brand"]):
            grp = grp.sort_values("week_start")
            a = grp["y_true_1"].to_numpy()
            pb, pg = grp["y_pred_1"].to_numpy(), grp["y_pred_2"].to_numpy()
            r = diebold_mariano(a, pb, pg, loss=loss)
            sig = r["p_value"] < 0.05
            if sig and r["better"] == 2:
                verdict, n_better = "membaik (signifikan)", n_better + 1
            elif sig and r["better"] == 1:
                verdict, n_worse = "memburuk (signifikan)", n_worse + 1
            else:
                verdict, n_ns = "tak signifikan", n_ns + 1
            mape_b, mape_g = mape(a, pb), mape(a, pg)
            rows.append({
                "row_type": "deret", "algo": ALGO_LABEL[algo],
                "store": store, "brand": brand,
                "MAPE_baseline": mape_b, "MAPE_gt": mape_g,
                "delta_MAPE_pct": mape_g - mape_b,
                "MAE_baseline": mae(a, pb), "MAE_gt": mae(a, pg),
                "delta_MAE": mae(a, pg) - mae(a, pb),
                "dm_stat": r["dm_stat"], "dm_p": r["p_value"], "verdict": verdict,
                "n_membaik": np.nan, "n_memburuk": np.nan, "n_tak_signifikan": np.nan,
            })
        summary_rows.append({
            "row_type": "ringkasan", "algo": ALGO_LABEL[algo],
            "store": "SEMUA", "brand": "SEMUA",
            "MAPE_baseline": np.nan, "MAPE_gt": np.nan, "delta_MAPE_pct": np.nan,
            "MAE_baseline": np.nan, "MAE_gt": np.nan, "delta_MAE": np.nan,
            "dm_stat": np.nan, "dm_p": np.nan,
            "verdict": f"membaik={n_better}; memburuk={n_worse}; tak_signifikan={n_ns}",
            "n_membaik": n_better, "n_memburuk": n_worse, "n_tak_signifikan": n_ns,
        })
    return pd.DataFrame(rows + summary_rows)
