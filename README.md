# RGBench

**Real Garment Benchmark — quantifying the sim-to-real gap for cloth simulators.**

🌐 **[rgbench.github.io](https://rgbench.github.io/)** &nbsp;·&nbsp; 📄 **AAAI 2026 paper** ([camera-ready PDF](https://rgbench.github.io/))

Robotic manipulation of garments is hard because cloth lives in an
effectively infinite-dimensional state space with thin-shell dynamics,
heavy self-contact, and material diversity. Most current cloth
simulators use approximations like Position-Based Dynamics that trade
fidelity for speed, leaving a substantial sim-to-real gap on
manipulation tasks. RGBench is the first benchmark that **systematically
quantifies that gap** with carefully measured real garment dynamics.

## What this repository releases

This release ships the **evaluation half** of the AAAI 2026 paper:

- ✅ **9 garments with real-world ground truth** —
  [`hwk0809/RGBench-Cloth-Sim2Real-v1`](https://huggingface.co/datasets/hwk0809/RGBench-Cloth-Sim2Real-v1)
  (6.7 GB on Hugging Face, CC-BY 4.0). Each garment is captured under
  three bimanual manipulation actions (fling / fold / grasp), with
  segmented point clouds, robot joint + end-effector CSVs, camera
  calibration, and the matching cloth meshes at multiple resolutions.
- ✅ **An open-source evaluation harness** with two reference simulator
  wrappers ([PyBullet](rgbench/envs/pybullet/),
  [Isaac Sim](rgbench/envs/isaacsim/)) and a clean
  [`BaseEnvWrapper`](rgbench/envs/base.py) interface for plugging in
  any new cloth simulator with four methods.
- ✅ **Published AAAI 2026 baseline numbers** in
  [`results/paper_baselines.csv`](results/paper_baselines.csv), plus a
  [`scripts/compare_to_paper.py`](scripts/compare_to_paper.py) helper
  that ranks a new simulator against them in one command.

---

## Architecture

```
Real-world capture                         Simulator
─────────────────                          ─────────
RealSense D455                             Any BaseEnvWrapper:
  │                                          • PyBullet
  │ RGB-D                                    • Isaac Sim
  ▼                                          • GarmentDynamics (open-sourcing soon; baseline numbers shipped)
GroundingDINO + SAM
  │
  │ segmented cloth pcd
  ▼                                          │
Real point cloud  ◄──── time sync ────►  Sim mesh vertices
                                             │
                                             ▼
              ┌──── Metrics ────┐
              │  Chamfer L1 / L2 │
              │  one-sided Hausdorff
              │  stability score │
              │  z-axis error    │
              └─────────────────┘
                       │
                       ▼
                  metrics.csv
                       │
                       ▼
              compare_to_paper.py
                       │
                       ▼
              "you rank 2/4 against paper baselines"
```

The same robot-trajectory CSV feeds both sides, and the benchmark
queries each simulator only through four methods (`step_to_time`,
`get_sim_vertices`, `get_master_start_time`, `get_current_sim_time`),
so plugging in a new simulator means implementing one class.

## Quickstart

```bash
git clone https://github.com/hwk0809/RGBench
cd RGBench
bash setup.sh                                # install + download sample data
make benchmark                               # PyBullet on green_tshirt grasp sample 02
python scripts/compare_to_paper.py outputs/  # compare your run to paper baselines
```

`setup.sh` creates a venv, installs RGBench with all extras, downloads
the SAM and GroundingDINO checkpoints, pulls the sample subset from
Hugging Face into `data/sample/`, and runs the PyBullet smoke test.

For Isaac Sim, install once into its bundled Python:

```bash
export ISAACSIM_PYTHON=~/isaacsim/python.sh
make install-isaacsim
make benchmark SIM=isaacsim
```

See [`docs/INSTALL.md`](docs/INSTALL.md) for the conda / Docker paths and
[`docs/BENCHMARK.md`](docs/BENCHMARK.md) for the evaluation protocol.

## Running the benchmark

RGBench supports four scopes. All use the same per-cell evaluation
protocol; they differ only in *which* cells of the experiment library
get run. `SIM` selects the simulator; `MODE` selects pseudo (pinned
grippers) vs full robot kinematics.

```bash
# 1) One specific (garment, action, sample) cell — fastest iteration unit
make benchmark SIM=pybullet SAMPLE=green_tshirt/grasp/02

# 2) All samples of one (garment, action) — typically 3–4 samples
make benchmark-action SIM=pybullet GARMENT=green_tshirt ACTION=grasp

# 3) All actions and samples of one garment — about 10 cells
make benchmark-garment SIM=pybullet GARMENT=green_tshirt

# 4) Every cell in configs/experiment_library.yaml — ~98 cells
make benchmark-all SIM=pybullet
```

All four targets accept `MODE={fixed_point,robot}` (default `fixed_point`)
and `ROBOT=piper`. Outputs land in
`outputs/<garment>/<action>/<sim>/<mode>/<robot>/sample_<NN>/<timestamp>/`.

For more control (filter by sample id, dry-run a batch first, pick a
non-default Python interpreter) the batch runner is callable directly:

```bash
python scripts/run_batch.py --sim pybullet --garment green_tshirt --dry-run
python scripts/run_batch.py --sim pybullet --garment green_tshirt --action grasp --sample 02
python scripts/run_batch.py --help
```

After any run, `scripts/compare_to_paper.py outputs/` ranks each cell
against the published PyBullet / IsaacSim / GarmentDynamics numbers in
[`results/paper_baselines.csv`](results/paper_baselines.csv).

## Evaluation modes (`fixed_point` vs `robot`)

Every cell can be run in two modes. They use the same captures and the
same metrics but model the gripper–cloth interaction differently. The
paper reports both: Table 4 is `fixed_point`, Table 5 is `robot`.

| | `fixed_point` (pseudo mode) | `robot` (full bimanual kinematics) |
| --- | --- | --- |
| Grippers | Massless mocap bodies driven by the recorded end-effector pose. The cloth is pinned at fixed vertex indices (`shoulder_index` in `cloth_params/<garment>.yaml`). | Two Piper arms loaded from URDF, PD-tracking the recorded joint trajectories. Gripper geometry + cloth contacts are simulated. |
| Robot tracking error | Excluded (the mocap follows the recorded pose exactly). | Included (PD controller has finite gains, gripper geometry has tolerances). |
| What it isolates | Cloth-solver fidelity — how well the simulator reproduces real cloth dynamics when the actuation is perfect. | End-to-end sim-to-real fit, including how well your simulator's robot tracking matches the real robot. |
| Cost | Fast — no robot dynamics. | Slower — robot + cloth + contact. |
| When to use | Comparing two cloth simulators on equal footing; isolating cloth physics from robot tracking; quick iteration. | Evaluating a full simulation stack including the robot; closest match to real-world experience. |

Pick the mode with `MODE=fixed_point` (default) or `MODE=robot`:

```bash
# Same cell in both modes
make benchmark SIM=pybullet SAMPLE=green_tshirt/grasp/02 MODE=fixed_point
make benchmark SIM=pybullet SAMPLE=green_tshirt/grasp/02 MODE=robot

# Full set, robot mode
make benchmark-all SIM=pybullet MODE=robot
```

For `robot` mode the eval set is smaller than `fixed_point` (some fling
cells were skipped in the paper because the bimanual robot couldn't
release reliably). `paper_baselines.csv` carries `n_samples=0` rows for
those — `compare_to_paper.py` just skips them.

## Metrics

Per evaluated frame, the benchmark computes seven metrics comparing the
simulated cloth mesh against the real-world segmented point cloud
(both in the world frame after applying the camera extrinsics). The
summary row written at the end of every `metrics.csv` is the mean over
the cell's evaluation window.

| Column | Formula | What it captures |
| --- | --- | --- |
| `chamfer_l1_real_to_sim`  ★ | mean over real points of distance to nearest sim point | Coverage — does the simulator reproduce every part of the real cloth? |
| `chamfer_l1_sim_to_real` | mean over sim points of distance to nearest real point | Excess — does the simulator stretch / drift outside the real cloth? |
| `chamfer_l2_sim_to_real` | same direction as above but L2 norm | Heavier penalty on large local errors |
| `one_sided_hausdorff_real_to_sim` | max over real points of distance to nearest sim point | Worst-case coverage miss |
| `one_sided_hausdorff_sim_to_real` | max over sim points of distance to nearest real point | Worst-case excess |
| `sim_stability_score` | vertex-wise local-mean deviation $\\frac{1}{N}\\sum \\big\\|\\frac{v_{t-1}+v_t+v_{t+1}}{3} - v_t\\big\\|$ | High = jitter / numerical instability in the simulator |
| `z_mean_error` | mean of $\\|z_{sim} - z_{real}\\|$ along gravity | How well gravity / settling is captured |

★ **Primary paper metric.** Section 4 of the AAAI paper highlights
real→sim Chamfer as the most informative number — for each real cloth
point, find the nearest simulated point. This penalizes a simulator
that fails to reproduce a part of the cloth (the real point won't have
a close sim neighbour). Numbers in the paper's Tables 4 and 5 are
real→sim Chamfer first; sim→real and the Hausdorff variants are
provided alongside.

Implementation:
[`rgbench.metrics.chamfer_distance_single_direction_{cpu,gpu}`](rgbench/metrics.py).
GPU path uses PyKeOps when available, otherwise falls back to SciPy on
CPU; results agree to ~6 decimals.

The `compare_to_paper.py` helper defaults to `--metric cd_l1_r2s`
(real→sim Chamfer L1). Other columns are selectable:

```bash
python scripts/compare_to_paper.py outputs/ --metric cd_l1_s2r
python scripts/compare_to_paper.py outputs/ --metric hd_r2s
```

## Visualization

By default a benchmark run writes only `metrics.csv` (and a `run.log`)
so batch sweeps stay fast and small on disk. When you want to see
*what* the simulator is doing — not just the chamfer number — turn on
the visualization artifacts via Hydra overrides:

```bash
make benchmark SAMPLE=green_tshirt/grasp/02 \
    active_run.visualization.save_gifs=true \
    active_run.visualization.save_sim_pcd=true \
    active_run.visualization.save_target_pcd=true
```

The cell's output directory then contains:

```
outputs/<cell>/<timestamp>/
├── metrics.csv
├── simulation_comparison.gif     # animated side-by-side sim vs real
├── sim_pcd_frames/*.pcd          # per-frame simulator output (world frame)
└── target_pcd_frames/*.pcd       # per-frame real point cloud (world frame)
```

Cost is ~5–15 MB per cell, almost all of it the per-frame PCDs;
`save_gifs=true` alone (animation only) is closer to ~0.5 MB.

### Interactive inspection of saved frames

After a run, step through the PCD frames in an Open3D viewer to see
where sim and real diverge:

```bash
# Step through real-frame point clouds for a cell
python tools/visualize_pcds_in_directory.py \
       outputs/<cell>/<timestamp>/target_pcd_frames

# Sim vs. real (loads from multiple sim wrappers, lets you toggle each)
python tools/visualize_sim_pcd.py
```

`visualize_sim_pcd.py` is configured at the top of the file — point it
at one or more `sim_pcd_frames/` directories and the matching
`target_pcd_frames/` (or HF-dataset segment_pcds) to A/B simulators on
the same capture.

### Live viewing during a benchmark run

For debugging a single cell, enable the live Open3D window via
override:

```bash
make benchmark SAMPLE=green_tshirt/grasp/02 \
    active_run.visualization.vis_sim=true \
    active_run.visualization.visualize_every_n_frames=1
```

This pops up a viewer at each evaluation frame; press `Q` to advance.

## Customising

Most users will want to evaluate their own cloth simulator against the
paper baselines, or add a new garment. The configs are arranged in
four layers; see [`docs/CONFIG.md`](docs/CONFIG.md) for what each layer
controls, what's safe to edit, and walkthroughs for **adding a new
garment** and **adding a new simulator**.

## Roadmap

This release covers the **evaluation half** of the AAAI 2026 paper.
Coming next:

- 🚧 **GarmentDynamics simulator** — the high-fidelity cloth simulator
  from Section 3 of the paper is being prepared for open release as a
  separate project. Its published baseline numbers already ship in
  [`results/paper_baselines.csv`](results/paper_baselines.csv) so any
  new simulator can be ranked against it today.
- 🚧 **6 000+ garment-mesh asset library** — the broader 3D garment
  dataset described in Section 2 will follow in a subsequent release.
- 💡 **More capture platforms** — current data is from a Piper bimanual
  gripper; the codebase has a K1 humanoid wrapper but no captures yet.

Contributions for new simulators, new garments, and new captures are
welcome via pull request; see [`docs/CONFIG.md`](docs/CONFIG.md) and
[`docs/ADDING_A_SIMULATOR.md`](docs/ADDING_A_SIMULATOR.md).

## Dataset

The Hugging Face dataset
[`hwk0809/RGBench-Cloth-Sim2Real-v1`](https://huggingface.co/datasets/hwk0809/RGBench-Cloth-Sim2Real-v1)
(6.7 GB, CC-BY 4.0) bundles **everything needed to run the benchmark**:

- **Real captures** — 98 capture sessions across 9 garments × 3 actions
  (fling / fold / grasp). Each capture has `calibration/`, `joints/`,
  and `segment_pcds/`.
- **Cloth meshes** under `meshes/<Garment>/` — these are the simulator
  input meshes, not in the git repo; they're versioned alongside the
  captures so a given dataset revision pins both data and meshes.
- **Multi-resolution `green_tshirt`** at 5 k / 10 k / 20 k / 40 k
  triangles under `meshes/Green_Tshirt_Compare/`, with matching
  `configs/cloth_params/green_tshirt_{5k,10k,20k,40k}.yaml`, for
  studying how cloth mesh resolution trades off sim-to-real fidelity
  against simulation cost.

```bash
python scripts/download_data.py                # full dataset (~6.6 GB)
python scripts/download_data.py --sample-only  # one capture for smoke test (~100 MB)
```

Of the 9 garments, **7** are evaluated in the paper's baselines. Two
garments (`grey_sunwear`, `khaki_blazer`) have non-manifold meshes
(self-intersecting / non-watertight) that PyBullet, Isaac Sim, and the
MuJoCo flex body all fail to load. Their raw real-world data and meshes
ship with the dataset so researchers using particle-based or otherwise
non-manifold-tolerant simulators can include them.

Note on `assets/` in this repo: it holds **simulator-specific scaffolding**
— MuJoCo XMLs, robot URDFs, Isaac Sim USDs — not the cloth meshes.
Cloth meshes always come from the HF dataset.

Format details and per-capture structure are documented in
[`docs/DATASET.md`](docs/DATASET.md).

## Paper baselines and comparing your simulator

[`results/paper_baselines.csv`](results/paper_baselines.csv) (105 rows,
9.5 KB) holds the published baseline numbers from the AAAI 2026 paper —
7 garments × 3 actions × {PyBullet, Isaac Sim, GarmentDynamics} ×
{fixed-point pseudo mode, robot mode}.

After running your simulator across the paper cells, compare with one
command:

```bash
python scripts/compare_to_paper.py outputs/ --metric cd_l1_r2s
```

Example output:

```
     garment action        mode your_sim you.cd_l1_r2s paper.pybullet paper.isaacsim paper.garmentdyn rank
green_tshirt  grasp fixed_point     mine        0.0241         0.0348         0.0341           0.0226  2/4

[mine]  n=21, mean rank = 1.86/4
    rank 1: 8
    rank 2: 11
    rank 3: 2
    rank 4: 0
    better than paper PyBullet: 19/21
    better than paper IsaacSim: 17/21
    better than paper GarmentDynamics: 3/21
```

See [`results/README.md`](results/README.md) for the schema, the
aggregation methodology, and the non-manifold-garment note.

## Adding a simulator

Implement [`BaseEnvWrapper`](rgbench/envs/base.py) (four methods) and
register it in [`rgbench/envs/__init__.py`](rgbench/envs/__init__.py).
A worked example lives in
[`docs/ADDING_A_SIMULATOR.md`](docs/ADDING_A_SIMULATOR.md).

## Documentation

| Doc | What's inside |
| --- | --- |
| [`docs/INSTALL.md`](docs/INSTALL.md) | venv / conda / Docker install paths; Isaac Sim's bundled Python |
| [`docs/BENCHMARK.md`](docs/BENCHMARK.md) | One-cell evaluation protocol — pseudocode for how a single `metrics.csv` is produced |
| [`docs/DATASET.md`](docs/DATASET.md) | HF dataset layout; per-capture schema; **calibration workflow** (landmarks → `world_to_camera`, optional hand-eye refinement); **time synchronization** (per-sample `camera_delay`, auto-correction) |
| [`docs/CONFIG.md`](docs/CONFIG.md) | Four-layer Hydra config (`main` / `env` / `cloth_params` / `experiment_library`) — what each layer controls, what's safe to edit, recipe for **adding a new garment** |
| [`docs/ADDING_A_SIMULATOR.md`](docs/ADDING_A_SIMULATOR.md) | `BaseEnvWrapper` interface walkthrough + factory registration for a new simulator |
| [`results/README.md`](results/README.md) | Paper baseline schema, aggregation methodology (eval-window mean → 3-sample mean), non-manifold-garment note |
| [`docs/UPLOAD_DATASET.md`](docs/UPLOAD_DATASET.md) | Maintainer-only: how to publish a new dataset revision to Hugging Face |

## Repository layout

| Path | Description |
| --- | --- |
| [`rgbench/`](rgbench/) | Python package — metrics, point-cloud processing, simulator wrappers |
| [`configs/`](configs/) | Hydra configs (experiments, per-garment cloth params, segmentation) — see [`docs/CONFIG.md`](docs/CONFIG.md) |
| [`scripts/`](scripts/) | `run_benchmark.py`, `run_batch.py`, `compare_to_paper.py`, dataset / checkpoint downloaders |
| [`tools/`](tools/) | Data-prep pipeline (rosbag → RGB-D → segmented PCD, calibration UI) |
| [`results/`](results/) | Paper baseline numbers (`paper_baselines.csv`) and methodology |
| [`tests/`](tests/) | Smoke tests for each simulator wrapper |
| [`examples/`](examples/) | Minimal end-to-end demo |
| [`third_party/`](third_party/) | Slim wrappers around GroundingDINO, SAM, and the RealSense SDK |
| [`assets/`](assets/) | MuJoCo XMLs, URDFs, Isaac Sim USDs (simulator scaffolding; cloth meshes live in the HF dataset) |
| [`docs/`](docs/) | Per-topic documentation — see the table above |

## License

Code is released under the [MIT License](LICENSE). The Hugging Face
dataset is released under [CC-BY 4.0](DATA_LICENSE). Robot URDFs
redistributed in `assets/Urdf/` carry their upstream licenses; see the
per-directory LICENSE / NOTICE files where applicable.

## Citation

If you use RGBench in your research, please cite the AAAI 2026 paper:

```bibtex
@inproceedings{hu2026rgbench,
  title     = {Real Garment Benchmark ({RGBench}): A Comprehensive Benchmark for Robotic Garment Manipulation featuring a High-Fidelity Scalable Simulator},
  author    = {Hu, Wenkang and Tang, Xincheng and E, Yanzhi and Li, Yitong and Shu, Zhengjie and Li, Wei and Wang, Huamin and Yang, Ruigang},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
  url       = {https://rgbench.github.io/}
}
```

See also [`CITATION.cff`](CITATION.cff).
