# Installation

RGBench supports three install paths. The one-shot script is the
recommended default; the manual conda path is for users who want fine
control over each dependency; the Docker path is a future option.

## Path 1 — one-shot `setup.sh` (recommended)

```bash
git clone https://github.com/hwk0809/RGBench
cd RGBench
bash setup.sh
```

What it does:
1. Confirms Python ≥ 3.10.
2. Creates a virtualenv at `.venv/` if one isn't already active.
3. Editable install with the `[all]` extras (PyBullet, MuJoCo, the
   segmentation pipeline, dev tools).
4. Downloads SAM + GroundingDINO checkpoints from the original authors
   into `third_party/grounded_sam/checkpoints/`.
5. Downloads the smoke-test sample from Hugging Face into `data/sample/`.
6. Runs `examples/pybullet_quickstart.py` and exits non-zero if it fails.

## Path 2 — manual conda environment

```bash
git clone https://github.com/hwk0809/RGBench
cd RGBench
conda create -n rgbench python=3.11 -y
conda activate rgbench
pip install -e ".[pybullet,mujoco,segment,dev]"
python scripts/download_checkpoints.py
python scripts/download_data.py --sample-only
python examples/pybullet_quickstart.py
```

Optional extras groups:
| Group | Adds |
| --- | --- |
| `pybullet` | `pybullet` |
| `mujoco`  | `mujoco`  |
| `segment` | `groundingdino-py`, `segment-anything`, `pykeops` (GPU chamfer) |
| `dev` | `pytest`, `ruff`, `build` |
| `all` | everything above |

## Path 3 — Isaac Sim

Isaac Sim cannot be installed via pip; install it from NVIDIA's Omniverse
Launcher first, then point at its bundled Python:

```bash
export ISAACSIM_PYTHON=~/isaacsim/python.sh
make install-isaacsim
# or, equivalently:
"$ISAACSIM_PYTHON" -m pip install -r requirements-isaacsim.txt
```

Then run the benchmark with `make benchmark sim=isaacsim` — `run_batch.py`
detects `isaacsim` as the backend and dispatches the child process under
the Isaac interpreter.

## Path 4 — Docker (planned for v0.2)

A Dockerfile and pre-built image will be published alongside the v0.2
release. For now use Path 1 or Path 2.

## Troubleshooting

**`OSError: libGL.so.1: cannot open shared object file`** — install
system OpenGL: `sudo apt install libgl1-mesa-glx libegl1`. `setup.sh`
exports `MUJOCO_GL=egl` + `LIBGL_ALWAYS_SOFTWARE=1` to avoid the most
common GL issues on headless servers.

**`huggingface_hub` 401 / private dataset** — the dataset is public; if
you see auth errors, log in with `huggingface-cli login` or point at a
mirror by setting `RGBENCH_HF_DATASET=<your-org>/<repo>`.

**Chamfer distance is slow** — install `pykeops` (in the `segment`
extras) for the GPU code path. CPU fallback via SciPy is correct but
~50× slower for large point clouds.

**Isaac Sim crashes on launch** — verify your driver and Isaac Sim
versions match the matrix at
https://docs.isaacsim.omniverse.nvidia.com/.
