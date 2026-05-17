"""Smoke test: load a MuJoCo XML model and step it in an interactive viewer.

Pass an XML path with --model; defaults to the bundled FlexCloth flex demo.
"""

import argparse
import os
import time

import mujoco
import mujoco.viewer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_MODEL = os.path.join(
    REPO_ROOT, "assets", "mujoco_model", "FlexCloth_Fixed_Point.xml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to MuJoCo XML.")
    parser.add_argument("--duration", type=float, default=10000.0, help="Wall-seconds.")
    args = parser.parse_args()

    m = mujoco.MjModel.from_xml_path(args.model)
    d = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < args.duration:
            step_start = time.time()
            mujoco.mj_step(m, d)
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
            viewer.sync()
            wait = m.opt.timestep - (time.time() - step_start)
            if wait > 0:
                time.sleep(wait)


if __name__ == "__main__":
    main()
