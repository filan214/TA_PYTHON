"""Tahap 6e-3 — Ensemble RF + LSTM (D11, prioritas 3).

Mengombinasikan prediksi RF & LSTM yang **sudah ada** di `predictions_*.parquet` —
TIDAK ada training baru sama sekali; murni kombinasi pasca-prediksi per deret per varian.
Dua skema:
    - **mean**   : rata-rata sederhana (RF + LSTM)/2. Bebas-leakage sepenuhnya.
    - **invMAE** : rata-rata berbobot invers-MAE secara *prequential/online* — pada minggu
      uji ke-i, bobot ∝ 1/MAE tiap model dihitung dari galat minggu uji [0..i-1] yang SUDAH
      teramati (minggu pertama -> rata-rata sederhana). Ini bebas-leakage karena hanya
      memakai galat masa lalu, konsisten dgn premis walk-forward D9 (aktual s/d t-1 diketahui).

**Rezim evaluasi (D9):** origin sepadan dgn komponennya (RF & LSTM sudah one-step D9).

Output (kelompok DM ke-4 Tahap 7, D11):
    reports/results/predictions_ensemble_mean_<variant>.parquet
    reports/results/predictions_ensemble_invmae_<variant>.parquet
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VARIANTS = ("baseline", "gt")
_KEYS = ["store", "brand", "week_start"]
_EPS = 1e-6


def _load_pair(results_dir, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rf = pd.read_parquet(results_dir / f"predictions_rf_{variant}.parquet")
    lstm = pd.read_parquet(results_dir / f"predictions_lstm_{variant}.parquet")
    return rf, lstm


def _merge_pair(rf: pd.DataFrame, lstm: pd.DataFrame) -> pd.DataFrame:
    """Gabung RF & LSTM pada origin sepadan; verifikasi y_true identik (prasyarat D9)."""
    m = rf[_KEYS + ["y_true", "y_pred"]].merge(
        lstm[_KEYS + ["y_true", "y_pred"]], on=_KEYS, suffixes=("_rf", "_lstm"))
    if len(m) != len(rf) or len(m) != len(lstm):
        raise ValueError("Origin RF & LSTM tak sepadan (pelanggaran D9).")
    if not np.allclose(m["y_true_rf"].to_numpy(), m["y_true_lstm"].to_numpy()):
        raise ValueError("y_true RF & LSTM berbeda pada origin yang sama (pelanggaran D9).")
    return m


def ensemble_mean(m: pd.DataFrame) -> np.ndarray:
    return np.clip(0.5 * m["y_pred_rf"].to_numpy() + 0.5 * m["y_pred_lstm"].to_numpy(), 0.0, None)


def ensemble_invmae_series(y_true, p_rf, p_lstm) -> np.ndarray:
    """Bobot invers-MAE prequential (online) untuk satu deret — bebas-leakage."""
    y_true, p_rf, p_lstm = map(lambda a: np.asarray(a, float), (y_true, p_rf, p_lstm))
    n = len(y_true)
    out = np.zeros(n)
    for i in range(n):
        if i == 0:
            w_rf = w_lstm = 0.5
        else:
            mae_rf = np.mean(np.abs(y_true[:i] - p_rf[:i]))
            mae_lstm = np.mean(np.abs(y_true[:i] - p_lstm[:i]))
            inv_rf, inv_lstm = 1.0 / (mae_rf + _EPS), 1.0 / (mae_lstm + _EPS)
            w_rf, w_lstm = inv_rf / (inv_rf + inv_lstm), inv_lstm / (inv_rf + inv_lstm)
        out[i] = w_rf * p_rf[i] + w_lstm * p_lstm[i]
    return np.clip(out, 0.0, None)


def build_variant(results_dir, variant: str) -> dict[str, pd.DataFrame]:
    """Bangun kedua ensemble untuk satu varian -> {scheme: df prediksi}."""
    rf, lstm = _load_pair(results_dir, variant)
    out = {"mean": [], "invmae": []}
    for (store, brand), g in _merge_pair(rf, lstm).groupby(["store", "brand"]):
        g = g.sort_values("week_start")
        base = {"store": store, "brand": brand, "week_start": g["week_start"].to_numpy(),
                "y_true": g["y_true_rf"].to_numpy()}
        out["mean"].append(pd.DataFrame({**base, "y_pred": ensemble_mean(g)}))
        out["invmae"].append(pd.DataFrame({**base, "y_pred": ensemble_invmae_series(
            g["y_true_rf"], g["y_pred_rf"], g["y_pred_lstm"])}))
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def train_all(cfg, save: bool = True) -> dict[str, pd.DataFrame]:
    """Hasilkan 4 file ensemble (2 skema × 2 varian) dari prediksi RF+LSTM existing."""
    rdir = cfg.paths.results
    out = {}
    for variant in VARIANTS:
        built = build_variant(rdir, variant)
        for scheme, df in built.items():
            name = f"ensemble_{scheme}_{variant}"
            out[name] = df
            if save:
                path = rdir / f"predictions_{name}.parquet"
                df.to_parquet(path, index=False)
                logger.info("Tersimpan ensemble %s: %s (%d baris)", name, path.name, len(df))
    return out


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Tahap 6e-3 — Ensemble RF+LSTM (D11)")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)

    from src.config import load_config

    cfg = load_config(args.config)
    cfg.paths.ensure()
    out = train_all(cfg)
    print("Stage 6e-3 (ensemble):", {k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
