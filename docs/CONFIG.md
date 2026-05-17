# Config layout and customization

RGBench uses [Hydra](https://hydra.cc) to compose configs from four
files. Knowing what each layer controls makes it clear which file to
edit when you want to (a) evaluate your own simulator, (b) add a new
cloth garment, or (c) tweak per-simulator defaults.

```
configs/
├── main.yaml                       ← top-level: which sim, which cloth, which sample
├── env/<sim>.yaml                  ← simulator defaults (timestep, default stiffness, ...)
├── cloth_params/<garment>.yaml     ← per-garment mesh paths + physical parameters
└── experiment_library.yaml         ← per-(garment, action, sample) evaluation windows
```

## The four layers

### 1. `configs/main.yaml` — selector
Picks which cell to evaluate. Almost always overridden on the command
line rather than edited:

```bash
python scripts/run_benchmark.py \
    params.cloth_name=green_tshirt \
    params.action_type=grasp \
    params.sample_index=02 \
    params.sim_environment=pybullet \
    params.sim_mode=fixed_point \
    env=pybullet \
    cloth_params=green_tshirt
```

The `Makefile` targets (`make benchmark` / `make benchmark-action` /
`make benchmark-garment` / `make benchmark-all`) wrap these overrides.

### 2. `configs/env/<sim>.yaml` — simulator defaults
One file per simulator wrapper (`pybullet.yaml`, `isaacsim.yaml`,
`mujoco.yaml`). Holds the simulator-wide defaults — physics timestep,
default cloth stiffness when no per-garment value overrides it, ground
friction, etc.

**Edit when** you add a new simulator wrapper and need a place to hold
its defaults. See
[`docs/ADDING_A_SIMULATOR.md`](ADDING_A_SIMULATOR.md).

### 3. `configs/cloth_params/<garment>.yaml` — per-garment ✏️
Where the **mesh paths** and **measured physical parameters** for one
garment live. This is the file you edit to add a new cloth. Example
(`green_tshirt.yaml`):

```yaml
name: "green_tshirt"
cloth_model_file_name: "Green_Tshirt/green_tshirt_10k.obj"
cloth_model_usda:      "Green_Tshirt/green_tshirt_10k.usda"

# Physical parameters (measured on a fabric tester; see paper §2)
stretch: 230000
bending: 1000
density: 220
friction: 0.6
mass: 0.255       # kg
poisson: 0.35

# Indices on the mesh that fixed-point grippers should pin to
shoulder_index: [185, 76]
```

Paths are resolved against the HF dataset's `meshes/` directory, i.e.
`Green_Tshirt/green_tshirt_10k.obj` means
`<RGBENCH_CLOTH_MESH_ROOT>/Green_Tshirt/green_tshirt_10k.obj`.

`green_tshirt` ships with four resolution variants
(`green_tshirt_5k.yaml`, `green_tshirt_10k.yaml`, `green_tshirt_20k.yaml`,
`green_tshirt_40k.yaml`) pointing at the corresponding mesh densities so
you can study how mesh resolution trades off simulator fidelity against
runtime.

### 4. `configs/experiment_library.yaml` — paper-defined ⚠️
One row per published cell. Each row carries `data_subfolder` (which
capture session feeds the cell), `camera_delay` (RealSense timestamp
offset), and the `evaluate.{start_record_time,end_record_time,
start_calculate_time,end_calculate_time}` window over which the metric
summary is computed.

**Do not edit existing rows** — the paper's published numbers in
[`results/paper_baselines.csv`](../results/paper_baselines.csv) are
keyed on these specific windows and timestamps. Editing one would
invalidate the comparison.

Append rows to evaluate on a new capture (see "Adding a new capture"
below).

## Adding a new garment

Goal: evaluate the benchmark on a new cloth you've captured.

1. **Mesh files** — drop your `.obj` and (optional) `.usda` into the HF
   dataset under `meshes/<New_Garment>/`. If you're not publishing the
   garment, just put them locally at
   `$RGBENCH_CLOTH_MESH_ROOT/<New_Garment>/`.

2. **`configs/cloth_params/<new_garment>.yaml`** — copy the closest
   existing garment yaml and update:
   - `name`
   - `cloth_model_file_name` and `cloth_model_usda` to your new mesh
   - measured physical parameters
   - `shoulder_index` (vertex indices to pin in `fixed_point` mode)

3. **Captures** — drop your captures under
   `$RGBENCH_DATA_ROOT/<new_garment>/<new_garment>_<action>_<timestamp>/`
   with the same layout the other garments use
   (`calibration/world_to_camera_transform.json`, `joints/*.csv`,
   `segment_pcds/*.pcd`). See
   [`docs/DATASET.md`](DATASET.md) for the per-capture schema and
   `tools/` for the rosbag → segmented-PCD pipeline.

4. **`configs/experiment_library.yaml`** — append one row per capture
   you want to evaluate, keyed
   `experiments.<garment>.<action>.<robot>.<sample_id>`. Copy the shape
   of an existing entry; pick the eval window carefully (the action's
   "interesting" interval after grasp closes, before the cloth rests).

Once those four are in place:

```bash
make benchmark SIM=pybullet SAMPLE=new_garment/grasp/01
```

If the cell runs, you can also do
`make benchmark-garment SIM=pybullet GARMENT=new_garment` to sweep all
samples you added.

## Adding a new simulator

[`BaseEnvWrapper`](../rgbench/envs/base.py) defines four methods. Write
one subclass, add a factory branch in
[`rgbench/envs/__init__.py`](../rgbench/envs/__init__.py), and drop a
defaults yaml at `configs/env/<your_sim>.yaml`. Full walkthrough:
[`docs/ADDING_A_SIMULATOR.md`](ADDING_A_SIMULATOR.md).

## Hydra override grammar (cheat sheet)

```bash
# Override a simple field
python scripts/run_benchmark.py params.cloth_name=green_tshirt

# Pick a different config group file
python scripts/run_benchmark.py env=isaacsim cloth_params=blue_dress

# Override a nested simulator default (rarely needed)
python scripts/run_benchmark.py env.physics_timestep=0.001

# Print the fully resolved config without running
python scripts/run_benchmark.py --cfg job
```

Anything you can do on the command line, you can also do permanently by
editing the relevant yaml.
