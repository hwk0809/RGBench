#!/usr/bin/env python3
"""Stage the filtered RGBench dataset into a directory ready for HF upload.

The full RGBench source data (Piper_Data + Style3dCloth) is ~260 GB on the
author's disk, but the benchmark only needs three subdirectories per
capture (calibration/, joints/, segment_pcds/) plus the cloth meshes that
``configs/cloth_params/*.yaml`` actually references. This script copies
just those into a clean staging directory so the upload to Hugging Face
is exactly what the benchmark consumes.

Typical use:

    # 1. Preview what would be staged (no disk writes)
    python scripts/prepare_hf_dataset.py --dry-run

    # 2. Actually stage (uses hardlinks when source and staging share a fs)
    python scripts/prepare_hf_dataset.py

    # 3. Upload to Hugging Face
    huggingface-cli upload-large-folder \\
        hwk0809/RGBench-Cloth-Sim2Real-v1 \\
        /tmp/rgbench_hf_upload --repo-type=dataset

The staged layout mirrors the dataset layout the benchmark expects:

    <staging>/
    ├── README.md                       # dataset card
    ├── LICENSE                          # CC-BY 4.0 for the data
    ├── <garment>/                       # captures, one subdir per garment
    │   └── <garment>_<action>_<ts>/
    │       ├── calibration/             # world_to_camera_transform.json + ...
    │       ├── joints/                  # left/right arm csv
    │       └── segment_pcds/            # *.pcd
    └── meshes/                          # cloth meshes
        └── <Garment>/
            ├── *.obj
            └── *.usda

Reference baseline numbers live in the GitHub repo at results/
(paper_baselines.csv), not in the HF dataset.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_PIPER = Path.home() / "DataSets" / "Piper_Data" / "Official"
DEFAULT_MESHES = Path.home() / "DataSets" / "Style3dCloth"
DEFAULT_STAGING = Path("/tmp/rgbench_hf_upload")

CAPTURE_SUBDIRS = ("calibration", "joints", "segment_pcds")


def collect_capture_subfolders(experiment_library: Path) -> list[str]:
    """Return unique ``data_subfolder`` values used by experiment_library.yaml."""
    with experiment_library.open() as f:
        data = yaml.safe_load(f)
    subfolders: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            if "data_subfolder" in node and isinstance(node["data_subfolder"], str):
                subfolders.add(node["data_subfolder"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    return sorted(subfolders)


def collect_mesh_paths(cloth_params_dir: Path) -> list[str]:
    """Return unique mesh file paths referenced from cloth_params/*.yaml."""
    mesh_paths: set[str] = set()
    for yaml_file in sorted(cloth_params_dir.glob("*.yaml")):
        with yaml_file.open() as f:
            params = yaml.safe_load(f) or {}
        for key in ("cloth_model_file_name", "cloth_model_usda"):
            val = params.get(key)
            if isinstance(val, str) and val.strip():
                mesh_paths.add(val.strip())
    return sorted(mesh_paths)


def human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                pass
    return total


def link_or_copy(src: Path, dst: Path, link_mode: str) -> None:
    """Hardlink (fall back to copy) a single file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link_mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    if link_mode == "symlink":
        os.symlink(src, dst)
        return
    shutil.copy2(src, dst)


def link_or_copy_tree(src: Path, dst: Path, link_mode: str) -> int:
    """Mirror src -> dst using hardlinks when possible. Returns bytes."""
    bytes_total = 0
    for entry in src.rglob("*"):
        if entry.is_file():
            rel = entry.relative_to(src)
            target = dst / rel
            link_or_copy(entry, target, link_mode)
            try:
                bytes_total += entry.stat().st_size
            except OSError:
                pass
    return bytes_total


def stage_captures(
    capture_subfolders: list[str],
    source_root: Path,
    staging: Path,
    link_mode: str,
    dry_run: bool,
) -> int:
    total_bytes = 0
    missing: list[str] = []
    for sub in capture_subfolders:
        src_capture = source_root / sub
        if not src_capture.is_dir():
            missing.append(sub)
            continue
        for sd in CAPTURE_SUBDIRS:
            src = src_capture / sd
            if not src.is_dir():
                continue
            dst = staging / sub / sd
            if dry_run:
                total_bytes += dir_size(src)
            else:
                total_bytes += link_or_copy_tree(src, dst, link_mode)
    if missing:
        print(f"  WARN: {len(missing)} referenced capture(s) missing on disk:", file=sys.stderr)
        for m in missing[:10]:
            print(f"    - {m}", file=sys.stderr)
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more", file=sys.stderr)
    return total_bytes


def stage_meshes(
    mesh_paths: list[str],
    source_root: Path,
    staging: Path,
    link_mode: str,
    dry_run: bool,
) -> int:
    """Mesh paths may be ``<Garment>/file.obj`` or just ``file.obj``."""
    total_bytes = 0
    meshes_dst = staging / "meshes"
    missing: list[str] = []
    for rel in mesh_paths:
        src = source_root / rel
        if not src.is_file():
            missing.append(rel)
            continue
        dst = meshes_dst / rel
        if dry_run:
            total_bytes += src.stat().st_size
        else:
            link_or_copy(src, dst, link_mode)
            total_bytes += src.stat().st_size
    if missing:
        print(f"  WARN: {len(missing)} mesh file(s) missing on disk:", file=sys.stderr)
        for m in missing:
            print(f"    - {m}", file=sys.stderr)
    return total_bytes


DATASET_CARD = """\
---
license: cc-by-4.0
task_categories:
  - robotics
tags:
  - cloth-simulation
  - sim-to-real
  - point-cloud
  - benchmark
size_categories:
  - 1K<n<10K
---

# RGBench Cloth Sim-to-Real (v1)

🌐 Project page: <https://rgbench.github.io/>  ·  📦 Code: <https://github.com/hwk0809/RGBench>

Nine carefully captured garments — three bimanual manipulation actions
each (fling / fold / grasp) — with **real-world ground truth point
clouds** for evaluating any cloth simulator's sim-to-real gap. Released
as the evaluation half of the AAAI 2026 paper *[Real Garment Benchmark
(RGBench)](https://rgbench.github.io/)*.

The larger 6 000+ garment-mesh asset library and the GarmentDynamics
simulator from the paper are on their way to open-sourcing in follow-up
releases. This dataset is the piece you need to **benchmark any cloth
simulator against real captured dynamics today**.

## Contents

| Path | What's inside |
| --- | --- |
| `<garment>/<garment>_<action>_<ts>/calibration/` | Camera extrinsics, initial object pose |
| `<garment>/<garment>_<action>_<ts>/joints/` | Bimanual robot joint + end-effector CSVs |
| `<garment>/<garment>_<action>_<ts>/segment_pcds/` | Segmented cloth point clouds (cloth-only, world frame after extrinsics) |
| `meshes/<Garment>/*.obj`, `*.usda` | Cloth garment meshes used by the simulators |

9 garments × {grasp, fold, fling} actions, captured with a Piper bimanual
gripper and a RealSense D455. ~100 evaluation samples in total. Two
garments (`grey_sunwear`, `khaki_blazer`) have non-manifold meshes and
are **not** part of the paper's published baselines — see the
[results/](https://github.com/hwk0809/RGBench/tree/main/results) folder
in the GitHub repo for the published baseline numbers and the
methodology note.

Garment meshes ship in four resolutions for `green_tshirt` (5k / 10k /
20k / 40k triangles) under `meshes/Green_Tshirt_Compare/` so researchers
can study how cloth mesh resolution affects sim-to-real fidelity.

## Quickstart

```bash
git clone https://github.com/hwk0809/RGBench
cd RGBench
bash setup.sh                       # installs deps + downloads this dataset
make benchmark sim=pybullet         # runs the smoke-test sample
```

If you only want to fetch the data:

```bash
pip install huggingface_hub
python -m huggingface_hub.commands.huggingface_cli download \\
    hwk0809/RGBench-Cloth-Sim2Real-v1 --repo-type dataset \\
    --local-dir ./data/sample
```

## Licensing

- **Data**: CC-BY 4.0 — attribution required, commercial use permitted.
- **Code (benchmark)**: MIT — see the [RGBench repo](https://github.com/hwk0809/RGBench).

## Citation

If you use this dataset, please cite the AAAI 2026 paper:

```bibtex
@inproceedings{hu2026rgbench,
  title     = {Real Garment Benchmark ({RGBench}): A Comprehensive Benchmark for Robotic Garment Manipulation featuring a High-Fidelity Scalable Simulator},
  author    = {Hu, Wenkang and Tang, Xincheng and E, Yanzhi and Li, Yitong and Shu, Zhengjie and Li, Wei and Wang, Huamin and Yang, Ruigang},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
  url       = {https://rgbench.github.io/}
}
```
"""


LICENSE_CC_BY_4 = """\
Creative Commons Attribution 4.0 International (CC BY 4.0)

You are free to:
  Share — copy and redistribute the material in any medium or format
  Adapt — remix, transform, and build upon the material for any purpose,
          even commercially.

Under the following terms:
  Attribution — You must give appropriate credit, provide a link to
                the license, and indicate if changes were made. You may
                do so in any reasonable manner, but not in any way that
                suggests the licensor endorses you or your use.

No additional restrictions — You may not apply legal terms or
                              technological measures that legally restrict
                              others from doing anything the license permits.

Full legal text: https://creativecommons.org/licenses/by/4.0/legalcode
"""


def write_dataset_card(staging: Path, dry_run: bool) -> int:
    """Write README.md + LICENSE into the staging dir."""
    files = {
        staging / "README.md": DATASET_CARD,
        staging / "LICENSE": LICENSE_CC_BY_4,
    }
    bytes_total = 0
    for path, body in files.items():
        bytes_total += len(body.encode("utf-8"))
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
    return bytes_total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source-piper",
        default=str(DEFAULT_PIPER),
        help=f"Real-capture root directory (default: {DEFAULT_PIPER}).",
    )
    parser.add_argument(
        "--source-meshes",
        default=str(DEFAULT_MESHES),
        help=f"Cloth-mesh root directory (default: {DEFAULT_MESHES}).",
    )
    parser.add_argument(
        "--staging",
        default=str(DEFAULT_STAGING),
        help=f"Where to stage the filtered dataset (default: {DEFAULT_STAGING}).",
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How to materialize files into the staging dir (default: hardlink, falls back to copy).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and total size, but do not write anything.",
    )
    args = parser.parse_args()

    piper_root = Path(args.source_piper).expanduser().resolve()
    mesh_root = Path(args.source_meshes).expanduser().resolve()
    staging = Path(args.staging).expanduser().resolve()

    if not piper_root.is_dir():
        print(f"ERROR: --source-piper does not exist: {piper_root}", file=sys.stderr)
        return 2
    if not mesh_root.is_dir():
        print(f"ERROR: --source-meshes does not exist: {mesh_root}", file=sys.stderr)
        return 2

    capture_subfolders = collect_capture_subfolders(CONFIGS_DIR / "experiment_library.yaml")
    mesh_paths = collect_mesh_paths(CONFIGS_DIR / "cloth_params")

    print("=" * 70)
    print(f"Source (captures): {piper_root}")
    print(f"Source (meshes):   {mesh_root}")
    print(f"Staging:           {staging}")
    print(f"Link mode:         {args.link_mode}")
    print(f"Dry-run:           {args.dry_run}")
    print("=" * 70)
    print(f"Captures referenced by experiment_library.yaml: {len(capture_subfolders)}")
    print(f"Mesh files referenced by cloth_params/*.yaml:   {len(mesh_paths)}")
    print()

    if not args.dry_run:
        staging.mkdir(parents=True, exist_ok=True)

    print("[1/3] Staging captures (calibration/, joints/, segment_pcds/) ...")
    cap_bytes = stage_captures(capture_subfolders, piper_root, staging, args.link_mode, args.dry_run)
    print(f"      {human_size(cap_bytes)}")

    print("[2/3] Staging cloth meshes ...")
    mesh_bytes = stage_meshes(mesh_paths, mesh_root, staging, args.link_mode, args.dry_run)
    print(f"      {human_size(mesh_bytes)}")

    print("[3/3] Writing dataset card and LICENSE ...")
    doc_bytes = write_dataset_card(staging, args.dry_run)
    print(f"      {human_size(doc_bytes)}")

    total = cap_bytes + mesh_bytes + doc_bytes
    print("=" * 70)
    print(f"TOTAL: {human_size(total)}")
    if args.dry_run:
        print("(dry run — no files written)")
    else:
        print(f"Staged at: {staging}")
        print()
        print("Next:")
        print(f"  huggingface-cli upload-large-folder \\")
        print(f"      hwk0809/RGBench-Cloth-Sim2Real-v1 \\")
        print(f"      {staging} --repo-type=dataset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
