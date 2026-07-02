"""Tahap 9 DoD — dashboard Streamlit boot tanpa error (§9).

Memakai `streamlit.testing.v1.AppTest` (headless) untuk menjalankan skrip apa adanya
dan memastikan tak ada exception, plus elemen kunci muncul: judul, caveat kejujuran
ramalan, dan minimal satu saran pengadaan. Dilewati bila streamlit/artefak tak ada.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "dashboard.py"
REQUIRED = [ROOT / "reports" / "results" / "inventory_params.csv",
            ROOT / "reports" / "results" / "metrics_summary.csv"]


@pytest.mark.skipif(not all(p.exists() for p in REQUIRED),
                    reason="artefak Tahap 7/8 belum ada")
def test_dashboard_runs_without_exception():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception, f"dashboard error: {at.exception}"
    # Judul & caveat kejujuran ramalan (load-only proxy) hadir.
    assert any("DSS Pengadaan Stok" in t.value for t in at.title)
    warn_text = " ".join(w.value for w in at.warning)
    assert "TERVALIDASI" in warn_text and "tidak" in warn_text.lower()


@pytest.mark.skipif(not all(p.exists() for p in REQUIRED),
                    reason="artefak Tahap 7/8 belum ada")
def test_dashboard_shows_order_recommendation():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    # Ada input stok di sidebar & minimal satu ringkasan saran ("ROP=").
    assert len(at.sidebar.number_input) >= 1
    body = " ".join(m.value for m in at.markdown)
    assert "ROP=" in body
