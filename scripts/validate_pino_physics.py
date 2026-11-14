"""CLI script to execute PINO physics validation, synthetic benchmarks, and UKEA validation diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics_validation.pino_physics_validator import run_all_diagnostics  # noqa: E402
from physics_validation.svg_plots import (  # noqa: E402
    plot_loss_and_gradient_dominance,
    plot_softplus_negative_paradox,
    plot_synthetic_benchmarks,
    plot_validation_mass_balance,
    plot_validation_spatial_maps,
)

DEFAULT_CACHE_DIR = Path(
    r"C:\Users\SHREEYANSH\.gemini\antigravity\brain\41ac7867-f88e-4dfd-bdf4-2d5ee850285d\scratch\ukea_val_cache"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PINO physics validation and diagnostic suite.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "physics_diagnostics",
        help="Directory for diagnostics JSON and SVG plots.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="External cache directory for UKEA validation arrays (outside Git).",
    )
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = run_all_diagnostics(repo_root=REPO_ROOT, cache_dir=args.cache_dir)
    arrays = report.pop("_arrays", None)

    json_path = out_dir / "diagnostics_summary.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    plot_synthetic_benchmarks(report["synthetic_benchmarks"], out_dir / "synthetic_benchmark_residuals.svg")
    plot_loss_and_gradient_dominance(report["loss_and_gradient_analysis"], out_dir / "loss_scale_and_gradient_dominance.svg")
    plot_softplus_negative_paradox(
        report["synthetic_benchmarks"],
        report["loss_and_gradient_analysis"],
        out_dir / "softplus_negative_depth_paradox.svg",
    )
    if report["validation_diagnostics"].get("events"):
        plot_validation_mass_balance(
            report["validation_diagnostics"],
            out_dir / "validation_event_mass_balance_diagnostics.svg",
        )
    if arrays is not None:
        plot_validation_spatial_maps(arrays, out_dir / "validation_spatial_residual_maps.svg")

    print(f"Wrote diagnostics JSON: {json_path}")
    print(f"Plots written to: {out_dir}")


if __name__ == "__main__":
    main()
