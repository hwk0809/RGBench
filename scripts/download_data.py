#!/usr/bin/env python3
"""Download the RGBench Cloth Sim-to-Real dataset from Hugging Face.

The dataset bundles segmented cloth point clouds, bimanual robot
trajectory CSVs, camera calibration, garment meshes, and reference
simulator metrics. By default the full set (~6.6 GB) lands in
``./data/sample/``, matching ``configs/main.yaml``'s default
``dataset_path`` so the benchmark works without setting env vars.

Use ``--sample-only`` to grab just the smoke-test capture (~100 MB) —
enough to run the quickstart but not to reproduce paper-scale numbers.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Default matches configs/main.yaml ``dataset_path`` fallback so the
# benchmark works out-of-the-box after a fresh download.
DEFAULT_TARGET = REPO_ROOT / "data" / "sample"

DEFAULT_REPO_ID = os.environ.get(
    "RGBENCH_HF_DATASET",
    "RGBench/RGBench-Cloth-Sim2Real-v1",
)

# One capture + its meshes + its reference result — the bare minimum to
# run the smoke test. Keeps the sample download under ~100 MB.
SAMPLE_PATTERNS = [
    "green_tshirt/green_tshirt_grasp_2025-07-19-19-14-58/**",
    "meshes/Green_Tshirt/**",
    "meshes/Green_Tshirt_Compare/**",
    "reference_results/green_tshirt/grasp/**",
    "README.md",
    "LICENSE",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repo (default: %(default)s).",
    )
    parser.add_argument(
        "--target",
        default=str(DEFAULT_TARGET),
        help="Local directory to populate (default: %(default)s).",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Download only the smoke-test sample subset.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset revision/branch (default: main).",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub not installed. Run:\n"
            "    pip install huggingface_hub\n"
            "or install RGBench with `pip install -e .` from the repo root.",
            file=sys.stderr,
        )
        return 1

    target = Path(args.target).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    allow_patterns = SAMPLE_PATTERNS if args.sample_only else None

    print(f"Repo:     {args.repo_id} @ {args.revision}")
    print(f"Target:   {target}")
    print(f"Subset:   {'sample only' if args.sample_only else 'full dataset'}")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
