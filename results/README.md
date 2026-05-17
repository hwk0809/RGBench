# RGBench paper baselines

This folder holds the **published benchmark numbers** from the RGBench
AAAI camera-ready paper, plus the script that compares a new simulator
against them.

```
results/
├── README.md            this file
└── paper_baselines.csv  paper Tables 4 + 5 in long format (105 rows)
```

## `paper_baselines.csv`

One row per (garment, action, simulator, mode) cell. Values are the
mean over 3 sample captures of the eval-window summary metric
(`metrics.csv.iloc[-2]` produced by [`scripts/run_benchmark.py`](../scripts/run_benchmark.py)).

| column         | meaning                                                     |
| -------------- | ----------------------------------------------------------- |
| `garment`      | one of 7 paper garments (see *Excluded garments* below)     |
| `action`       | `fling`, `fold`, `grasp`                                    |
| `simulator`    | `pybullet`, `isaacsim`, `mujoco_style3d` (the latter is paper's "Ours" column, ≈ GarmentDynamics) |
| `mode`         | `fixed_point` (pseudo mode, pinned grippers) or `robot` (full bimanual robot kinematics) |
| `n_samples`    | how many capture samples were averaged (typically 3)        |
| `cd_l1_s2r`    | Chamfer L1 sim → real                                       |
| `cd_l2_s2r`    | Chamfer L2 sim → real                                       |
| `cd_l1_r2s`    | Chamfer L1 real → sim    *(the primary fidelity metric)*    |
| `hd_s2r`       | one-sided Hausdorff sim → real                              |
| `hd_r2s`       | one-sided Hausdorff real → sim                              |
| `sim_stab`     | cloth simulation stability score                            |
| `z_err`        | mean z-axis error                                           |

Cells with `n_samples == 0` were not run (e.g. fling tasks in robot
mode for some garments — the bimanual robot couldn't release reliably).
Those rows are not present in the CSV; the table covers 105/126 cells.

## Comparing your own simulator

Once you've run `scripts/run_benchmark.py` with your simulator wrapper
across some or all of the 21 paper cells:

```bash
python scripts/compare_to_paper.py outputs/ --metric cd_l1_r2s
```

Output: a per-cell table showing where your simulator ranks among
{PyBullet, IsaacSim, GarmentDynamics, you}, plus a summary like:

```
[my_simulator]  n=21, mean rank = 1.86/4
    rank 1: 8
    rank 2: 11
    rank 3: 2
    rank 4: 0
    better than paper PyBullet: 19/21
    better than paper IsaacSim: 17/21
    better than paper GarmentDynamics:  3/21
```

## Excluded garments

The HF dataset ships real-world data for **9 garments** but the paper
baselines only cover **7**. Two garments were excluded from baselines
because their cloth meshes are **non-manifold** (self-intersecting,
non-watertight) — PyBullet, IsaacSim, and MuJoCo all fail to load them:

- `grey_sunwear`
- `khaki_blazer`

Their raw real-world data (point clouds, joint trajectories, calibration)
and meshes are still distributed in the HF dataset so that researchers
with simulators that *do* handle non-manifold meshes (e.g. particle-based
PBD on arbitrary point sets) can benchmark on them.

## Reproducibility note

Re-running PyBullet or IsaacSim against this baseline typically agrees
within 2–5 % per cell. PyBullet has non-deterministic collision-solver
order; IsaacSim has non-deterministic GPU physics. GarmentDynamics
results are **not reproducible** from this repo because that simulator
is closed-source — the column is provided for comparison only.

## Where these numbers came from

`paper_baselines.csv` was generated from the per-frame `metrics.csv`
files produced during the paper's experiments, aggregated with:

1. For each (garment, action, simulator, sample, mode), take
   `metrics.csv.iloc[-2]` (the eval-window summary mean written by
   [`OutputManager`](../rgbench/output.py)).
2. Mean across the 3 samples within each cell.
3. Round to 4 decimal places.

The paper's LaTeX tables in supplementary materials are reconstructable
from this CSV (and the analysis script in
`scripts/run_benchmark.py`'s upstream pipeline).
