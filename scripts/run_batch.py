#!/usr/bin/env python3
"""Run a batch of RGBench cells defined in configs/experiment_library.yaml.

By default runs every cell in the experiment library. Use the filters
to scope down to one garment, one action, or one specific sample.

Examples:

    # Run paper full set on PyBullet (7 garments x 3 actions x 3 samples)
    python scripts/run_batch.py --sim pybullet

    # Run all green_tshirt cells across actions and samples
    python scripts/run_batch.py --sim pybullet --garment green_tshirt

    # Run all samples of green_tshirt grasp
    python scripts/run_batch.py --sim pybullet --garment green_tshirt --action grasp

    # Run one specific sample (equivalent to: make benchmark SAMPLE=green_tshirt/grasp/02)
    python scripts/run_batch.py --sim pybullet --garment green_tshirt \\
                                --action grasp --sample 02

    # Dry-run: print the plan without executing
    python scripts/run_batch.py --sim pybullet --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = REPO_ROOT / "configs" / "experiment_library.yaml"

VALID_SIMS = ("pybullet", "isaacsim", "mujoco", "garment_dynamics")
VALID_ACTIONS = ("fling", "fold", "grasp")
VALID_MODES = ("fixed_point", "robot")


def iter_cells(library: dict,
               garment: str | None,
               action: str | None,
               robot: str | None,
               sample: str | None) -> Iterable[tuple[str, str, str, str]]:
    """Yield (garment, action, robot, sample_id) tuples matching the filters."""
    for g, by_action in (library.get("experiments") or {}).items():
        if garment and g != garment:
            continue
        for a, by_robot in (by_action or {}).items():
            if action and a != action:
                continue
            for r, by_sample in (by_robot or {}).items():
                if robot and r != robot:
                    continue
                for s in (by_sample or {}):
                    if sample and str(s) != str(sample):
                        continue
                    yield g, a, r, str(s)


def cell_command(python: str, garment: str, action: str, robot: str,
                 sample: str, sim: str, mode: str) -> list[str]:
    """Hydra override list that selects exactly one cell."""
    return [
        python,
        str(REPO_ROOT / "scripts" / "run_benchmark.py"),
        f"params.cloth_name={garment}",
        f"params.action_type={action}",
        f"params.robot={robot}",
        f"params.sample_index={sample}",
        f"params.sim_environment={sim}",
        f"params.sim_mode={mode}",
        f"env={sim}",
        f"cloth_params={garment}",
        "active_run.visualization.vis_sim=false",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sim", required=True, choices=VALID_SIMS,
                        help="Simulator environment to evaluate")
    parser.add_argument("--mode", default="fixed_point", choices=VALID_MODES,
                        help="fixed_point (pseudo, pinned grippers) or robot (full bimanual). "
                             "Default: fixed_point.")
    parser.add_argument("--garment", default=None,
                        help="Restrict to one garment, e.g. green_tshirt")
    parser.add_argument("--action", default=None, choices=VALID_ACTIONS,
                        help="Restrict to one action")
    parser.add_argument("--robot", default=None,
                        help="Restrict to one robot (currently only piper)")
    parser.add_argument("--sample", default=None,
                        help="Restrict to one sample id, e.g. 02")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter to invoke (default: current)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the matched cells and the commands without executing")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Keep running remaining cells if one fails (default: stop)")
    args = parser.parse_args()

    if not LIB_PATH.is_file():
        print(f"ERROR: {LIB_PATH} not found.", file=sys.stderr)
        return 2

    library = yaml.safe_load(LIB_PATH.read_text()) or {}
    cells = list(iter_cells(library, args.garment, args.action, args.robot, args.sample))
    if not cells:
        print("No cells matched. Check --garment/--action/--robot/--sample against "
              "configs/experiment_library.yaml.", file=sys.stderr)
        return 1

    print(f"Matched {len(cells)} cells. Sim: {args.sim}, mode: {args.mode}.")
    if args.dry_run:
        for g, a, r, s in cells:
            print(f"  {g:<22} {a:<6} {r:<6} sample_{s}")
        return 0

    t_start = time.time()
    failed = []
    for i, (g, a, r, s) in enumerate(cells, 1):
        cmd = cell_command(args.python, g, a, r, s, args.sim, args.mode)
        print()
        print(f"=" * 70)
        print(f"[{i:3d}/{len(cells)}]  {g} / {a} / {r} / sample_{s}  ({args.sim} {args.mode})")
        print(f"=" * 70)
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
        if rc != 0:
            print(f"  cell FAILED with exit {rc}")
            failed.append((g, a, r, s, rc))
            if not args.continue_on_error:
                break

    dur = time.time() - t_start
    print()
    print(f"Done in {dur/60:.1f} min. {len(cells) - len(failed)}/{len(cells)} succeeded.")
    if failed:
        print("Failed cells:")
        for g, a, r, s, rc in failed:
            print(f"  {g}/{a}/{r}/sample_{s} (exit {rc})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
