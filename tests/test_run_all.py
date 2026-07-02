"""§6 — orkestrator pipeline. Uji logika pemilihan tahap & deteksi-lewati (skip)
tanpa mengeksekusi pipeline berat (tak ada fit model di sini)."""
from pathlib import Path

from src import run_all as RA


def test_build_steps_ordered_and_covers_stage_1_to_8():
    steps = RA.build_steps()
    stages = [s.stage for s in steps]
    assert stages == sorted(stages)              # urut menaik
    assert min(stages) == 1 and max(stages) == 8
    names = [s.name for s in steps]
    assert names[0] == "clean" and names[-1] == "inventory"


def test_select_steps_filters_by_stage_range():
    steps = RA.build_steps()
    sel = RA.select_steps(steps, from_stage=6, to_stage=7, no_trends=False)
    assert sel and all(6 <= s.stage <= 7 for s in sel)
    assert not any(s.stage < 6 or s.stage > 7 for s in sel)


def test_select_steps_drops_trends_when_no_trends():
    steps = RA.build_steps()
    sel = RA.select_steps(steps, from_stage=1, to_stage=8, no_trends=True)
    assert not any(s.name == "google_trends" for s in sel)
    assert any(s.name == "clean" for s in sel)   # tahap lain tetap ada


def test_should_skip_respects_artifacts_and_force(tmp_path):
    exists = tmp_path / "done.parquet"
    exists.write_text("x")
    missing = tmp_path / "missing.parquet"
    done = RA.Step(1, "done", run=lambda p: None, outputs=lambda c: [exists])
    todo = RA.Step(1, "todo", run=lambda p: None, outputs=lambda c: [missing])

    assert RA.should_skip(done, cfg=None, force=False) is True     # artefak ada -> lewati
    assert RA.should_skip(todo, cfg=None, force=False) is False    # artefak hilang -> jalan
    assert RA.should_skip(done, cfg=None, force=True) is False     # --force -> selalu jalan
