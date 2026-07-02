"""Tahap 6c — LSTM per deret & per varian (Bab III §3.1.6, D5/D6/D9).

Sekuens sliding-window (window=L) untuk meramal permintaan minggu berikutnya.
Kanal per-timestep = `units` + eksogen (kalender + Fourier + year_trend), dan
`gt_index` sebagai kanal tambahan HANYA di varian `gt` (ablation D5). Arsitektur
identik antar-varian agar selisih performa murni dari ada/tidaknya GT.

**Rezim evaluasi (D9):** one-step-ahead — tiap jendela uji memakai permintaan
aktual hingga t-1 (bukan rekursif). Origin uji = 37 minggu terakhir, sepadan
dengan SARIMAX walk-forward & RF -> valid untuk uji DM (Tahap 7).

Arsitektur: LSTM(units) -> Dropout -> Dense(1); loss=MSE, opt=Adam,
EarlyStopping(patience, restore_best_weights), validation_split temporal (tanpa shuffle).
Skala MinMax di-fit HANYA pada minggu latih (anti-leakage). Ramalan di-inverse & clip >= 0.

Output:
    reports/results/predictions_lstm_<variant>.parquet  [store,brand,week_start,y_true,y_pred]
    models/lstm/<variant>/<store>_<brand>.keras + scaler_<store>_<brand>.pkl

DoD: 20 artefak per varian (40 total); prediksi 37 minggu per deret×varian.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

# Harus di-set sebelum TensorFlow diimpor (impor keras dilakukan lazy di bawah):
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # senyapkan log INFO/WARNING TF
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")     # reproducibility (matikan oneDNN)

import numpy as np
import pandas as pd

from src.models.base import Forecaster
from src.models.sarimax import sarimax_exog_columns
from src.utils.splits import temporal_masks, train_cutoff_week

logger = logging.getLogger(__name__)


# --- Konstruksi kanal & sekuens --------------------------------------------

def lstm_channels(all_cols, cfg, variant: str) -> list[str]:
    """Kanal LSTM = units + eksogen (kalender+Fourier+year_trend[+gt_index])."""
    return ["units"] + sarimax_exog_columns(all_cols, cfg, variant)


def make_sequences(channels_scaled: np.ndarray, y_scaled: np.ndarray,
                   week_start: pd.DatetimeIndex, L: int):
    """Jendela geser: prediksi minggu t dari kanal minggu [t-L .. t-1] (one-step)."""
    X, y, idx = [], [], []
    for t in range(L, len(y_scaled)):
        X.append(channels_scaled[t - L:t])
        y.append(y_scaled[t])
        idx.append(week_start[t])
    return np.asarray(X, dtype="float32"), np.asarray(y, dtype="float32"), pd.DatetimeIndex(idx)


def build_series_sequences(features_df: pd.DataFrame, cfg, variant: str, cutoff: pd.Timestamp):
    """Bangun sekuens train/test satu deret (skala MinMax fit train saja, D3)."""
    from sklearn.preprocessing import MinMaxScaler

    df = features_df.sort_values("week_start").reset_index(drop=True)
    channels = lstm_channels(df.columns, cfg, variant)
    mat = df[channels].astype(float).to_numpy()
    weeks = pd.DatetimeIndex(df["week_start"].values)
    L = int(cfg["lags"])

    is_train_week = weeks < cutoff
    scaler = MinMaxScaler().fit(mat[is_train_week])   # fit pada minggu latih saja
    mat_s = scaler.transform(mat)
    umin, umax = scaler.data_min_[0], scaler.data_max_[0]     # 'units' = kolom 0
    urange = (umax - umin) or 1.0
    y_scaled = (mat[:, 0] - umin) / urange

    X, y, idx = make_sequences(mat_s, y_scaled, weeks, L)
    mask_train, mask_test = temporal_masks(pd.Series(idx), cutoff=cutoff)
    tr, te = mask_train.values, mask_test.values
    y_true = df.loc[df["week_start"].isin(idx[te]), "units"].astype(float).to_numpy()
    return (X[tr], y[tr], X[te], pd.DatetimeIndex(idx[te]), y_true,
            float(umin), float(urange))


# --- Forecaster -------------------------------------------------------------

class LSTMForecaster(Forecaster):
    """LSTM(units)->Dropout->Dense(1), EarlyStopping, validation temporal."""

    def __init__(self, units: int = 64, dropout: float = 0.2, epochs: int = 200,
                 batch_size: int = 16, patience: int = 10, val_frac: float = 0.2,
                 seed: int = 42):
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_frac = val_frac
        self.seed = seed
        self.model_ = None

    def _build(self, n_steps: int, n_channels: int):
        import keras
        from keras import layers

        keras.utils.set_random_seed(self.seed)   # python/np/tf seed
        model = keras.Sequential([
            keras.Input((n_steps, n_channels)),
            layers.LSTM(self.units),
            layers.Dropout(self.dropout),
            layers.Dense(1),
        ])
        model.compile(loss="mse", optimizer="adam")
        return model

    def fit(self, X, y=None) -> "LSTMForecaster":
        import keras

        self.model_ = self._build(X.shape[1], X.shape[2])
        es = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )
        self.model_.fit(
            X, y, epochs=self.epochs, batch_size=self.batch_size,
            validation_split=self.val_frac, shuffle=False,   # validasi = ekor temporal
            callbacks=[es], verbose=0,
        )
        return self

    def predict(self, X, exog_future=None) -> np.ndarray:
        return self.model_.predict(X, verbose=0).ravel()

    def name(self) -> str:
        return "LSTM"


# --- Orkestrasi -------------------------------------------------------------

def train_all(cfg, variant: str, series: list[str] | None = None,
              save: bool = True, **lstm_kwargs):
    """Latih LSTM 20 deret satu varian (one-step D9); kembalikan predictions_df."""
    from src.models.sarimax import _load_series_features

    feats = _load_series_features(cfg)
    if series is not None:
        feats = {k: v for k, v in feats.items() if k in series}
    grid = pd.DatetimeIndex(sorted(next(iter(feats.values()))["week_start"].unique()))
    cutoff = train_cutoff_week(grid, float(cfg["train_ratio"]))

    lstm_cfg = dict(cfg.get("lstm", {}))
    params = {
        "units": lstm_cfg.get("units", 64), "dropout": lstm_cfg.get("dropout", 0.2),
        "epochs": lstm_cfg.get("epochs", 200), "batch_size": lstm_cfg.get("batch_size", 16),
        "patience": lstm_cfg.get("patience", 10),
    }
    params.update(lstm_kwargs)

    pred_rows = []
    for key, df in feats.items():
        store, brand = key.split("|")
        X_tr, y_tr, X_te, test_index, y_true, umin, urange = build_series_sequences(
            df, cfg, variant, cutoff
        )
        fc = LSTMForecaster(seed=int(cfg.seed), **params).fit(X_tr, y_tr)
        y_pred = np.clip(fc.predict(X_te) * urange + umin, 0.0, None)  # inverse skala + clip

        pred_rows.append(pd.DataFrame({
            "store": store, "brand": brand, "week_start": test_index,
            "y_true": y_true, "y_pred": y_pred,
        }))
        if save:
            import joblib

            mdir = cfg.paths.models / "lstm" / variant
            mdir.mkdir(parents=True, exist_ok=True)
            fc.model_.save(mdir / f"{store}_{brand}.keras")
            joblib.dump({"units_min": umin, "units_range": urange},
                        mdir / f"scaler_{store}_{brand}.pkl")
        logger.info("LSTM[%s] %s: pred[:3]=%s", variant, key, np.round(y_pred[:3], 2))

    predictions = pd.concat(pred_rows, ignore_index=True)
    if save:
        cfg.paths.results.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(cfg.paths.results / f"predictions_lstm_{variant}.parquet", index=False)
        logger.info("Tersimpan prediksi LSTM (one-step D9): predictions_lstm_%s.parquet (%d baris)",
                    variant, len(predictions))
    return predictions


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="Tahap 6c — LSTM (baseline & gt)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--variant", choices=["baseline", "gt", "both"], default="both")
    args = ap.parse_args(argv)

    from src.config import load_config
    from src.utils.io import set_seed

    cfg = load_config(args.config)
    cfg.paths.ensure()
    set_seed(cfg.seed)
    variants = ["baseline", "gt"] if args.variant == "both" else [args.variant]

    n = 0
    for variant in variants:
        preds = train_all(cfg, variant)
        n += preds.groupby(["store", "brand"]).ngroups
    print("Stage 6c:", {"variants": variants, "n_series_variant": n})


if __name__ == "__main__":
    main()
