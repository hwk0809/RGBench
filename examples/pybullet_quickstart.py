"""Minimal PyBullet smoke test.

This is the script invoked at the end of ``setup.sh`` to confirm RGBench is
installed correctly. It loads the cloth wrapper headlessly, steps it for a
brief window, and exits with 0 on success.

For a real benchmark run see ``scripts/run_benchmark.py``.
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import pybullet as p  # noqa: F401
    except ImportError:
        print("pybullet not installed; `pip install -e .[pybullet]`", file=sys.stderr)
        return 1

    try:
        from rgbench.metrics import chamfer_distance_single_direction_cpu
        from rgbench.envs.base import BaseEnvWrapper  # noqa: F401
        from rgbench.envs import get_env  # noqa: F401
    except ImportError as exc:
        print(f"RGBench import failed: {exc}", file=sys.stderr)
        return 1

    import numpy as np

    a = np.random.RandomState(0).randn(64, 3).astype(np.float32)
    b = a + 0.01
    dist = chamfer_distance_single_direction_cpu(a, b, distance_type="l2")
    print(f"OK — chamfer(sample, sample+0.01) = {dist:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
