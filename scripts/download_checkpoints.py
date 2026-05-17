#!/usr/bin/env python3
"""Download the pre-trained checkpoints RGBench needs for point-cloud segmentation.

Two models, both released by their original authors:

  - Segment Anything (SAM, ViT-H) — released by Meta AI under Apache 2.0.
  - GroundingDINO (Swin-T, OGC) — released by IDEA-Research.

Default download target: third_party/grounded_sam/checkpoints/ inside the
repo. Override with $RGBENCH_CHECKPOINT_DIR.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "third_party" / "grounded_sam" / "checkpoints"

# (filename, url, expected sha256, human-readable size)
CHECKPOINTS = [
    (
        "sam_vit_h_4b8939.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
        "2.4 GB",
    ),
    (
        "groundingdino_swint_ogc.pth",
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "bb5b3215f797a8ebec5cee1ba9e16f6f9aff32f2f33fbb96e8e3a4f51b5e7f4a",
        "662 MB",
    ),
    (
        "GroundingDINO_SwinT_OGC.py",
        "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        None,
        "1 KB",
    ),
]


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.rename(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        default=os.environ.get("RGBENCH_CHECKPOINT_DIR", str(DEFAULT_DIR)),
        help="Target directory (default: %(default)s).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if files exist."
    )
    args = parser.parse_args()

    target = Path(args.dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint dir: {target}")

    failed = []
    for name, url, expected_sha, size in CHECKPOINTS:
        dest = target / name
        if dest.exists() and not args.force:
            print(f"  [skip] {name} ({size}) — already present")
            continue
        print(f"  [get ] {name} ({size}) from {url}")
        try:
            download(url, dest)
        except Exception as exc:
            print(f"  [fail] {name}: {exc}")
            failed.append(name)
            continue
        if expected_sha:
            got = sha256_of(dest)
            if got != expected_sha:
                print(f"  [warn] {name}: sha256 mismatch (got {got[:12]}…, "
                      f"expected {expected_sha[:12]}…). File kept; re-run with --force to retry.")

    if failed:
        print(f"\n{len(failed)} checkpoint(s) failed: {failed}", file=sys.stderr)
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
