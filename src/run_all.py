"""Orkestrator pipeline end-to-end (Bab III §6, §9 reproducibility).

Menjalankan Tahap 1→8 berurutan, melewati tahap yang artefaknya sudah ada (kecuali
`--force`). Seed global difiksasi (seed=42) sebelum tahap apa pun. Tiap tahap
memanggil `main(["--config", ...])` modulnya sendiri (loop varian/kind sudah di sana).

    python -m src.run_all --config config.yaml [--no-trends] [--from-stage N] [--to-stage M] [--force]

Catatan: Tahap 9 (DSS dashboard) BUKAN bagian batch pipeline — dijalankan terpisah
via `streamlit run app/dashboard.py` (interaktif), sesuai rencana.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class Step:
    stage: int                              # nomor Tahap (1..8) untuk filter --from/--to
    name: str
    run: Callable[[str], None]              # eksekusi tahap (argumen: path config)
    outputs: Callable[[object], list[Path]]  # artefak penanda untuk deteksi-lewati


def _module_runner(module_path: str, func: str = "main") -> Callable[[str], None]:
    """Runner yang mengimpor modul secara lazy lalu memanggil main(["--config", path])."""
    def run(config_path: str) -> None:
        getattr(importlib.import_module(module_path), func)(["--config", config_path])
    return run


def build_steps() -> list[Step]:
    """Daftar tahap terurut 1→8 dengan artefak penanda masing-masing."""
    def res(c) -> Path:
        return c.paths.results

    return [
        Step(1, "clean", _module_runner("src.data.clean"),
             lambda c: [c.paths.interim / "transactions_clean.parquet"]),
        Step(2, "aggregate", _module_runner("src.data.aggregate"),
             lambda c: [c.paths.interim / "weekly_store_brand.parquet"]),
        Step(3, "google_trends", _module_runner("src.data.google_trends"),
             lambda c: [c.paths.interim / "google_trends.csv"]),
        Step(4, "eda", _module_runner("src.eda.explore"),
             lambda c: [res(c) / "eda_summary.json"]),
        Step(5, "features", _module_runner("src.features.build"),
             lambda c: [c.paths.processed / "features_Toko_A_Xiaomi.parquet"]),
        Step(6, "sarimax", _module_runner("src.models.sarimax"),
             lambda c: [res(c) / "predictions_sarimax_gt.parquet"]),
        Step(6, "random_forest", _module_runner("src.models.random_forest"),
             lambda c: [res(c) / "predictions_rf_gt.parquet"]),
        Step(6, "lstm", _module_runner("src.models.lstm"),
             lambda c: [res(c) / "predictions_lstm_gt.parquet"]),
        Step(6, "naive", _module_runner("src.models.naive"),
             lambda c: [res(c) / "predictions_naive.parquet"]),
        Step(6, "rf_poisson", _module_runner("src.models.rf_poisson"),
             lambda c: [res(c) / "predictions_rf_poisson_gt.parquet"]),
        Step(6, "croston", _module_runner("src.models.croston"),
             lambda c: [res(c) / "predictions_croston.parquet"]),
        Step(6, "ensemble", _module_runner("src.models.ensemble"),
             lambda c: [res(c) / "predictions_ensemble_mean_gt.parquet"]),
        Step(7, "evaluation", _module_runner("src.evaluation.metrics"),
             lambda c: [res(c) / "metrics_summary.csv"]),
        Step(8, "inventory", _module_runner("src.inventory.optimize"),
             lambda c: [res(c) / "inventory_params.csv"]),
    ]


def select_steps(steps: list[Step], from_stage: int, to_stage: int,
                 no_trends: bool) -> list[Step]:
    """Saring tahap pada rentang [from_stage, to_stage]; buang trends bila --no-trends."""
    out = [s for s in steps if from_stage <= s.stage <= to_stage]
    if no_trends:
        out = [s for s in out if s.name != "google_trends"]
    return out


def should_skip(step: Step, cfg, force: bool) -> bool:
    """Lewati bila SEMUA artefak penanda sudah ada (kecuali --force)."""
    if force:
        return False
    return all(Path(p).exists() for p in step.outputs(cfg))


def run_pipeline(config_path: str, from_stage: int = 1, to_stage: int = 8,
                 no_trends: bool = False, force: bool = False) -> list[str]:
    """Jalankan tahap terpilih; kembalikan daftar nama tahap yang benar-benar dieksekusi."""
    from src.config import load_config
    from src.utils.io import set_seed

    cfg = load_config(config_path)
    cfg.paths.ensure()
    set_seed(cfg.seed)

    steps = select_steps(build_steps(), from_stage, to_stage, no_trends)
    executed = []
    for step in steps:
        if should_skip(step, cfg, force):
            logger.info("Tahap %d [%s]: DILEWATI (artefak sudah ada).", step.stage, step.name)
            continue
        logger.info("Tahap %d [%s]: MULAI.", step.stage, step.name)
        t0 = time.perf_counter()
        step.run(config_path)
        logger.info("Tahap %d [%s]: SELESAI (%.1fs).", step.stage, step.name,
                    time.perf_counter() - t0)
        executed.append(step.name)
    logger.info("Pipeline selesai. Dieksekusi: %s", executed or "(semua dilewati)")
    return executed


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Orkestrator pipeline Bab III (Tahap 1→8)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--from-stage", type=int, default=1)
    ap.add_argument("--to-stage", type=int, default=8)
    ap.add_argument("--no-trends", action="store_true",
                    help="lewati fetch Google Trends (pakai cache bila ada)")
    ap.add_argument("--force", action="store_true",
                    help="jalankan ulang walau artefak sudah ada")
    args = ap.parse_args(argv)

    executed = run_pipeline(args.config, args.from_stage, args.to_stage,
                            args.no_trends, args.force)
    print("Dieksekusi:", executed or "(semua tahap dilewati — artefak lengkap)")


if __name__ == "__main__":
    main()
