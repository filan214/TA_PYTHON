"""Tahap 6a — SARIMA / SARIMAX per deret & per varian (Bab III §3.1.6, D5/D6/D9).

Dua varian per deret (ablation GT, D5):
    - `baseline` = SARIMA, exog = kalender + Fourier (TANPA gt_index)
    - `gt`       = SARIMAX, exog = kalender + Fourier + gt_index
Keduanya punya exog; bedanya murni ada/tidaknya gt_index -> isolasi efek GT.

Orde (p,d,q)(P,D,Q)_52 dipilih per deret via `pmdarima.auto_arima` (AIC, stepwise),
dipandu d/D awal dari `eda_summary.json`. **Orde musiman TIDAK dipaksa non-nol** (D6).
Fallback non-seasonal bila auto_arima seasonal gagal konvergen.

**Rezim evaluasi (D9) — one-step-ahead walk-forward:** prediksi 37 minggu uji BUKAN
fixed-origin 37-langkah. Orde tetap dari pemilihan pada data latih; state model
diperbarui satu minggu demi satu minggu dengan data aktual via `statsmodels`
`append(refit=False)` (identitas model tetap, tak re-fit tiap langkah). Ini menyamakan
rezim dengan RF/LSTM (alami one-step) dan menjadi PRASYARAT validitas uji DM (Tahap 7)
serta formula safety stock one-step (Tahap 8).

Dua jalur:
    - `fit()`       : auto_arima (pencarian orde) — untuk run penuh dari nol.
    - `fit_fixed()` : bangun SARIMAX statsmodels dari orde yang SUDAH dipilih
                      (mis. dari sarima_orders.csv) tanpa pencarian ulang — murah,
                      untuk regenerasi prediksi walk-forward.
Keduanya berbagi `walk_forward()`.

Catatan: eksogen SARIMAX = kolom varian KECUALI lag_*/roll_* (ARIMA memodelkan
autokorelasi target sendiri). Eksogen distandardisasi (fit train). Ramalan clip >= 0.

Output:
    reports/results/predictions_sarimax_<variant>.parquet  [store,brand,week_start,y_true,y_pred]
    reports/results/sarima_orders.csv  (kedua varian; kolom has_seasonal_terms)
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build import resolve_feature_columns
from src.models.base import Forecaster
from src.utils.splits import temporal_masks, train_cutoff_week

logger = logging.getLogger(__name__)

_LAG_ROLL_PREFIXES = ("lag_", "roll_")


# --- Pemilihan eksogen & split ---------------------------------------------

def sarimax_exog_columns(all_cols, cfg, variant: str) -> list[str]:
    """Kolom eksogen SARIMAX = kolom varian tanpa lag_*/roll_* (lih. catatan modul)."""
    cols = resolve_feature_columns(all_cols, cfg, variant)
    return [c for c in cols if not c.startswith(_LAG_ROLL_PREFIXES)]


def split_endog_exog(features_df: pd.DataFrame, cfg, variant: str, cutoff: pd.Timestamp):
    """Bagi deret penuh (184 mgg) jadi endog/exog latih (147) & uji (37) via cutoff D3."""
    df = features_df.sort_values("week_start").reset_index(drop=True)
    exog_cols = sarimax_exog_columns(df.columns, cfg, variant)
    mask_train, mask_test = temporal_masks(df["week_start"], cutoff=cutoff)
    y = df["units"].astype(float)
    y_train = y[mask_train.values].to_numpy()
    y_test = y[mask_test.values].to_numpy()
    X_train = df.loc[mask_train.values, exog_cols].astype(float)
    X_test = df.loc[mask_test.values, exog_cols].astype(float)
    test_index = pd.DatetimeIndex(df.loc[mask_test.values, "week_start"].values)
    return y_train, y_test, X_train, X_test, test_index, exog_cols


# --- Forecaster -------------------------------------------------------------

class SarimaxForecaster(Forecaster):
    """SARIMA/SARIMAX dengan pencarian orde (auto_arima) atau orde tetap, + walk-forward."""

    def __init__(self, seasonal: bool = True, m: int = 52, d=None, D=None,
                 max_p: int = 3, max_q: int = 3, max_P: int = 1, max_Q: int = 1,
                 standardize_exog: bool = True):
        self.seasonal = seasonal
        self.m = m
        self.d = d
        self.D = D
        self.max_p, self.max_q, self.max_P, self.max_Q = max_p, max_q, max_P, max_Q
        self.standardize_exog = standardize_exog
        self.model_ = None          # model pmdarima (jalur auto_arima)
        self._res_fixed = None      # SARIMAXResults statsmodels (jalur fit_fixed)
        self.exog_scaler_ = None
        self.fell_back_ = False
        self._order = None
        self._sorder = None
        self._aic = float("nan")

    def _prep_exog(self, exog, *, fit: bool):
        if exog is None:
            return None
        X = np.asarray(exog, dtype=float)
        if not self.standardize_exog:
            return X
        from sklearn.preprocessing import StandardScaler

        if fit:
            self.exog_scaler_ = StandardScaler().fit(X)
        return self.exog_scaler_.transform(X)

    def fit(self, y, exog=None) -> "SarimaxForecaster":
        """Jalur auto_arima: cari orde via AIC, fallback non-seasonal bila gagal."""
        from pmdarima import auto_arima

        X = self._prep_exog(exog, fit=True)
        common = dict(
            X=X, d=self.d, max_p=self.max_p, max_q=self.max_q,
            stepwise=True, error_action="ignore", suppress_warnings=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.model_ = auto_arima(
                    y, seasonal=self.seasonal, m=self.m, D=self.D,
                    max_P=self.max_P, max_Q=self.max_Q, **common,
                )
            except Exception as exc:  # noqa: BLE001 — auto_arima gagal di semua kandidat
                logger.warning("auto_arima seasonal gagal (%s); fallback non-seasonal.", exc)
                self.fell_back_ = True
                self.model_ = auto_arima(y, seasonal=False, **common)
        self._order = tuple(self.model_.order)
        self._sorder = tuple(self.model_.seasonal_order)
        try:
            self._aic = float(self.model_.aic())
        except Exception:  # noqa: BLE001
            self._aic = float("nan")
        return self

    def fit_fixed(self, y, exog, order, seasonal_order, trend=None) -> "SarimaxForecaster":
        """Jalur orde tetap: bangun SARIMAX statsmodels dari orde yang sudah dipilih."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        X = self._prep_exog(exog, fit=True)
        order = tuple(int(v) for v in order)
        seasonal_order = tuple(int(v) for v in seasonal_order)
        if trend is None:
            trend = "c" if order[1] == 0 else "n"  # intercept saat tak ada differencing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = SARIMAX(np.asarray(y, dtype=float), exog=X, order=order,
                          seasonal_order=seasonal_order, trend=trend,
                          enforce_stationarity=False, enforce_invertibility=False)
            self._res_fixed = mod.fit(disp=False, maxiter=200)
        self._order = order
        self._sorder = seasonal_order
        self._aic = float(self._res_fixed.aic)
        return self

    def _statsmodels_result(self):
        """Hasil statsmodels untuk walk-forward (dari fit_fixed atau arima_res_ pmdarima)."""
        if self._res_fixed is not None:
            return self._res_fixed
        return self.model_.arima_res_

    def walk_forward(self, y_test, exog_test=None) -> np.ndarray:
        """D9: one-step-ahead walk-forward. Ramal t, tambah aktual t, maju (refit=False)."""
        res = self._statsmodels_result()
        Xt = self._prep_exog(exog_test, fit=False) if exog_test is not None else None
        y_test = np.asarray(y_test, dtype=float)
        preds, cur = [], res
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for t in range(len(y_test)):
                xf = None if Xt is None else Xt[t:t + 1]
                yhat = float(np.asarray(cur.forecast(steps=1, exog=xf))[0])
                preds.append(yhat)
                cur = cur.append(endog=y_test[t:t + 1], exog=xf, refit=False)
        return np.clip(np.asarray(preds, dtype=float), 0.0, None)

    def predict(self, horizon: int, exog_future=None) -> np.ndarray:
        """Multi-step fixed-origin (pmdarima). Disediakan untuk referensi; evaluasi pakai walk_forward."""
        X = self._prep_exog(exog_future, fit=False)
        yhat = np.asarray(self.model_.predict(n_periods=horizon, X=X), dtype=float)
        return np.clip(yhat, 0.0, None)

    def name(self) -> str:
        return "SARIMAX"

    @property
    def order(self) -> tuple:
        return self._order

    @property
    def seasonal_order(self) -> tuple:
        return self._sorder

    @property
    def has_seasonal_terms(self) -> bool:
        P, D, Q, _ = self._sorder
        return bool((P + D + Q) > 0)

    @property
    def aic(self) -> float:
        return self._aic


# --- Orkestrasi -------------------------------------------------------------

def _eda_orders(cfg) -> dict[str, tuple]:
    """Peta series_key -> (recommend_d, recommend_D) dari eda_summary.json (Tahap 4)."""
    path = cfg.paths.results / "eda_summary.json"
    if not path.exists():
        return {}
    adf = json.loads(path.read_text(encoding="utf-8")).get("adf", {})
    return {k: (v.get("recommend_d"), v.get("recommend_D")) for k, v in adf.items()}


def _load_series_features(cfg) -> dict[str, pd.DataFrame]:
    """Muat fitur per deret dari data/processed (bangun bila belum ada)."""
    proc = cfg.paths.processed
    files = sorted(proc.glob("features_*.parquet"))
    if not files:
        from src.features.build import _load_inputs, build_all_features

        weekly, gt = _load_inputs(cfg)
        return build_all_features(weekly, gt, cfg, processed_dir=proc)
    out = {}
    for f in files:
        df = pd.read_parquet(f)
        out[f"{df['store'].iloc[0]}|{df['brand'].iloc[0]}"] = df
    return out


def _orders_lookup(orders_df: pd.DataFrame, variant: str) -> dict[str, dict]:
    sub = orders_df[orders_df["variant"] == variant]
    return {f"{r.store}|{r.brand}": r for r in sub.itertuples(index=False)}


def fit_one(features_df, cfg, variant, cutoff, d0=None, D0=None,
            order_row=None, **search_kwargs):
    """Fit (auto_arima ATAU orde tetap dari order_row) + prediksi walk-forward (D9)."""
    y_train, y_test, X_train, X_test, test_index, exog_cols = split_endog_exog(
        features_df, cfg, variant, cutoff
    )
    fc = SarimaxForecaster(m=int(cfg["seasonal_period"]), d=d0, D=D0, **search_kwargs)
    if order_row is not None:
        fc.fit_fixed(y_train, X_train,
                     order=(order_row.p, order_row.d, order_row.q),
                     seasonal_order=(order_row.P, order_row.D, order_row.Q, order_row.m))
    else:
        fc.fit(y_train, X_train)
    y_pred = fc.walk_forward(y_test, X_test)          # D9 one-step-ahead
    return fc, y_test, y_pred, test_index, exog_cols


def train_all(cfg, variant: str, series: list[str] | None = None,
              save: bool = True, orders_df: pd.DataFrame | None = None, **search_kwargs):
    """Latih 20 deret untuk satu varian (prediksi walk-forward D9); (predictions, orders)."""
    feats = _load_series_features(cfg)
    if series is not None:
        feats = {k: v for k, v in feats.items() if k in series}
    grid = pd.DatetimeIndex(sorted(next(iter(feats.values()))["week_start"].unique()))
    cutoff = train_cutoff_week(grid, float(cfg["train_ratio"]))
    eda = _eda_orders(cfg)
    lookup = _orders_lookup(orders_df, variant) if orders_df is not None else {}

    pred_rows, order_rows = [], []
    for key, df in feats.items():
        store, brand = key.split("|")
        d0, D0 = eda.get(key, (None, None))
        fc, y_test, y_pred, test_index, _ = fit_one(
            df, cfg, variant, cutoff, d0=d0, D0=D0,
            order_row=lookup.get(key), **search_kwargs
        )
        pred_rows.append(pd.DataFrame({
            "store": store, "brand": brand, "week_start": test_index,
            "y_true": y_test, "y_pred": y_pred,
        }))
        P, D, Q, m = fc.seasonal_order
        p, d, q = fc.order
        order_rows.append({
            "store": store, "brand": brand, "variant": variant,
            "p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q, "m": m,
            "order": f"({p},{d},{q})", "seasonal_order": f"({P},{D},{Q})[{m}]",
            "has_seasonal_terms": fc.has_seasonal_terms,
            "fell_back": fc.fell_back_, "aic": round(fc.aic, 2),
        })
        logger.info("SARIMAX[%s] %s: (%d,%d,%d)(%d,%d,%d)[%d] seasonal=%s walk-forward",
                    variant, key, p, d, q, P, D, Q, m, fc.has_seasonal_terms)

    predictions = pd.concat(pred_rows, ignore_index=True)
    orders = pd.DataFrame(order_rows)
    if save:
        cfg.paths.results.mkdir(parents=True, exist_ok=True)
        out_pred = cfg.paths.results / f"predictions_sarimax_{variant}.parquet"
        predictions.to_parquet(out_pred, index=False)
        logger.info("Tersimpan prediksi (walk-forward D9): %s (%d baris)", out_pred, len(predictions))
    return predictions, orders


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="Tahap 6a — SARIMA/SARIMAX (baseline & gt), walk-forward D9")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--variant", choices=["baseline", "gt", "both"], default="both")
    ap.add_argument("--from-orders", action="store_true",
                    help="regenerasi prediksi walk-forward dari sarima_orders.csv (tanpa auto_arima)")
    args = ap.parse_args(argv)

    from src.config import load_config

    cfg = load_config(args.config)
    cfg.paths.ensure()
    variants = ["baseline", "gt"] if args.variant == "both" else [args.variant]

    orders_df = None
    orders_path = cfg.paths.results / "sarima_orders.csv"
    if args.from_orders:
        orders_df = pd.read_csv(orders_path)
        logger.info("Regenerasi prediksi walk-forward dari orde tersimpan: %s", orders_path)

    all_orders = []
    for variant in variants:
        _, orders = train_all(cfg, variant, orders_df=orders_df)
        all_orders.append(orders)

    if not args.from_orders:  # jalur pencarian: tulis ulang orders
        orders_out = pd.concat(all_orders, ignore_index=True)
        orders_out.to_csv(orders_path, index=False)

    final = pd.read_csv(orders_path)
    print("Stage 6a:", {
        "variants": variants,
        "mode": "from-orders (walk-forward regen)" if args.from_orders else "auto_arima search",
        "n_models": len(final),
        "n_with_seasonal_terms": int(final["has_seasonal_terms"].sum()),
        "orders_csv": str(orders_path),
    })


if __name__ == "__main__":
    main()
