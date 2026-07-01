"""Tahap 3 DoD — Google Trends eksogen (§5, §7).

Semua uji OFFLINE & deterministik: `fetch_trends` di-monkeypatch, tak ada
panggilan network. Memverifikasi penyelarasan grid, indeks [0,100], cakupan,
idempotensi cache, dan jalur fallback (--no-trends & fetch gagal).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.data.google_trends as gt

KW = {"Xiaomi": ["Xiaomi HP"], "OPPO": ["OPPO HP"]}


def _grid(n: int, start: str = "2022-01-03") -> pd.DatetimeIndex:
    # n minggu Senin berturut-turut (grid ala Tahap 2).
    return pd.date_range(start, periods=n, freq="W-MON")


def _valid_frame(keywords_by_brand, week_grid) -> pd.DataFrame:
    # Frame GT "nyata" cakupan penuh, nilai [0,100], untuk mem-patch fetch_trends.
    frames = []
    for i, brand in enumerate(keywords_by_brand):
        vals = np.linspace(0, 100, len(week_grid)) if i == 0 else np.full(len(week_grid), 42.0)
        frames.append(pd.DataFrame({"week_start": week_grid, "brand": brand, "gt_index": vals}))
    return pd.concat(frames, ignore_index=True)[gt.COLUMNS]


# --- align_to_grid ----------------------------------------------------------

def test_align_sunday_points_to_monday_grid():
    grid = _grid(6)
    sundays = grid - pd.Timedelta(days=1)  # anchor Minggu ala Google
    s = pd.Series([0, 20, 40, 60, 80, 100], index=sundays)
    aligned = gt.align_to_grid(s, grid)
    assert aligned.notna().all()                       # cakupan penuh via nearest
    assert list(aligned.values) == [0, 20, 40, 60, 80, 100]


def test_align_ffills_single_gap_only():
    grid = _grid(6)
    sundays = grid - pd.Timedelta(days=1)
    # Hilangkan titik minggu ke-3 -> Senin ke-3 di luar toleransi (nearest 8/6 hari).
    s = pd.Series([0, 20, 60, 80, 100], index=sundays.delete(2))
    aligned = gt.align_to_grid(s, grid)
    assert aligned.iloc[2] == 20                        # di-ffill dari minggu ke-2
    assert aligned.notna().all()


def test_align_two_consecutive_gaps_leave_nan():
    grid = _grid(6)
    sundays = grid - pd.Timedelta(days=1)
    # Hilangkan minggu ke-3 & ke-4: ffill(limit=1) hanya menutup satu.
    s = pd.Series([0, 20, 80, 100], index=sundays.delete([2, 3]))
    aligned = gt.align_to_grid(s, grid)
    assert aligned.iloc[2] == 20                        # tertutup ffill
    assert np.isnan(aligned.iloc[3])                    # tetap NaN (limit=1)


# --- _rescale_0_100 ---------------------------------------------------------

def test_rescale_peak_to_100_preserves_zero():
    s = pd.Series([0.0, 25.0, 50.0])
    out = gt._rescale_0_100(s)
    assert list(out.values) == [0.0, 50.0, 100.0]


def test_rescale_all_zero_unchanged():
    s = pd.Series([0.0, 0.0, 0.0])
    out = gt._rescale_0_100(s)
    assert (out == 0).all()


# --- fallback_trends --------------------------------------------------------

def test_fallback_schema_all_nan():
    grid = _grid(5)
    fb = gt.fallback_trends(KW, grid)
    assert list(fb.columns) == gt.COLUMNS
    assert len(fb) == 5 * len(KW)
    assert fb["gt_index"].isna().all()
    assert set(fb["brand"]) == set(KW)


# --- validate_trends --------------------------------------------------------

def test_validate_detects_low_coverage():
    grid = _grid(20)
    frame = _valid_frame(KW, grid).copy()
    # Kosongkan 3 dari 20 minggu untuk OPPO -> cakupan 0.85 < 0.95.
    mask = (frame["brand"] == "OPPO") & (frame["week_start"].isin(grid[:3]))
    frame.loc[mask, "gt_index"] = np.nan
    stats = gt.validate_trends(frame, grid)
    assert stats["min_coverage_ok"] is False
    assert stats["coverage_by_brand"]["OPPO"] == pytest.approx(0.85)


def test_validate_flags_out_of_range():
    grid = _grid(10)
    frame = _valid_frame(KW, grid).copy()
    frame.loc[0, "gt_index"] = 150.0
    stats = gt.validate_trends(frame, grid)
    assert stats["in_range"] is False


# --- build_google_trends: fetched / cache / fallback ------------------------

def test_build_fetched_writes_valid_cache(monkeypatch, tmp_path):
    grid = _grid(184)
    monkeypatch.setattr(gt, "fetch_trends", lambda *a, **k: _valid_frame(KW, grid))
    out = tmp_path / "google_trends.csv"
    df = gt.build_google_trends(KW, grid, "tf", "ID", out_path=out)
    assert df.attrs["stats"]["source"] == "fetched"
    assert df.attrs["stats"]["in_range"] and df.attrs["stats"]["min_coverage_ok"]
    assert out.exists()
    vals = df["gt_index"].dropna()
    assert (vals >= 0).all() and (vals <= 100).all()
    assert list(df.columns) == gt.COLUMNS


def test_build_cache_is_idempotent_no_network(monkeypatch, tmp_path):
    grid = _grid(184)
    out = tmp_path / "google_trends.csv"
    _valid_frame(KW, grid).to_csv(out, index=False)  # cache sudah ada

    def _boom(*a, **k):
        raise AssertionError("fetch_trends dipanggil padahal cache ada")

    monkeypatch.setattr(gt, "fetch_trends", _boom)
    df = gt.build_google_trends(KW, grid, "tf", "ID", out_path=out)  # tak boleh fetch
    assert df.attrs["stats"]["source"] == "cache"
    assert pd.api.types.is_datetime64_any_dtype(df["week_start"])


def test_build_force_refetches_over_cache(monkeypatch, tmp_path):
    grid = _grid(184)
    out = tmp_path / "google_trends.csv"
    gt.fallback_trends(KW, grid).to_csv(out, index=False)  # cache fallback lama
    called = {"n": 0}

    def _fake(*a, **k):
        called["n"] += 1
        return _valid_frame(KW, grid)

    monkeypatch.setattr(gt, "fetch_trends", _fake)
    df = gt.build_google_trends(KW, grid, "tf", "ID", out_path=out, force=True)
    assert called["n"] == 1
    assert df.attrs["stats"]["source"] == "fetched"


def test_build_no_trends_flag_yields_nan(tmp_path):
    grid = _grid(184)
    out = tmp_path / "google_trends.csv"
    df = gt.build_google_trends(KW, grid, "tf", "ID", out_path=out, no_trends=True)
    assert df.attrs["stats"]["source"] == "fallback"
    assert df["gt_index"].isna().all()
    assert out.exists()  # file fallback ditulis -> pipeline tetap punya artefak


def test_build_fetch_failure_falls_back(monkeypatch, tmp_path):
    grid = _grid(184)

    def _fail(*a, **k):
        raise RuntimeError("rate limited (429)")

    monkeypatch.setattr(gt, "fetch_trends", _fail)
    out = tmp_path / "google_trends.csv"
    df = gt.build_google_trends(KW, grid, "tf", "ID", out_path=out)  # tak melempar
    assert df.attrs["stats"]["source"] == "fallback"
    assert df["gt_index"].isna().all()
    assert out.exists()
