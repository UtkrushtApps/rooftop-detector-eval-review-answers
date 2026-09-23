"""Command-line entry point for local evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io import load_split
from .reporting import build_summary, format_summary
from .thresholds import select_operating_point, sweep_thresholds


def _format_sweep(sweep) -> str:
    lines = ["validation threshold sweep (selection evidence):"]
    for _, row in sweep.iterrows():
        lines.append(
            f"  t={row['threshold']:.2f} "
            f"macroF1={row['headline']:.4f} "
            f"macroP={row['macro_precision']:.4f} "
            f"macroR={row['macro_recall']:.4f} "
            f"safetyR={row['safety_rail_gap_recall']:.4f}"
        )
    return "\n".join(lines)


def run(data_root: str | Path = "data") -> str:
    data_root = Path(data_root)
    val = load_split(data_root / "validation")
    test = load_split(data_root / "test")

    # Select the threshold only on validation data.  The test split is used once,
    # after the operating point is fixed, to estimate unseen performance.
    validation_sweep = sweep_thresholds(val["annotations"], val["predictions"])
    operating_point = select_operating_point(validation_sweep)

    validation_summary = build_summary(
        val["annotations"],
        val["predictions"],
        score_threshold=operating_point,
        split_name="validation at selected threshold",
    )
    test_summary = build_summary(
        test["annotations"],
        test["predictions"],
        score_threshold=operating_point,
        split_name="test at validation-selected threshold",
    )

    parts = [
        _format_sweep(validation_sweep),
        "",
        f"selected operating score threshold from validation: {operating_point}",
        "",
        format_summary(validation_summary),
        "",
        format_summary(test_summary),
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rooftop detector outputs.")
    parser.add_argument("--data-root", default="data", help="Path to the data directory.")
    args = parser.parse_args()
    print(run(args.data_root))


if __name__ == "__main__":
    main()
