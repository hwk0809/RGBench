# Dataset

The RGBench Cloth Sim-to-Real dataset lives on Hugging Face:
[`RGBench/RGBench-Cloth-Sim2Real-v1`](https://huggingface.co/datasets/RGBench/RGBench-Cloth-Sim2Real-v1).
The git repo only carries the benchmark code; the actual real-world
captures and the cloth meshes are downloaded from HF.

## Download

```bash
python scripts/download_data.py                 # full dataset (~6.7 GB)
python scripts/download_data.py --sample-only   # one capture for smoke test (~100 MB)

# overrides:
python scripts/download_data.py --repo-id <user>/<dataset> --target /shared/cache
```

`$RGBENCH_HF_DATASET` overrides the default repo id; `--target` overrides
the destination (default: `./data/sample/` so the benchmark's `dataset_path`
in `configs/main.yaml` finds it without any further configuration).

## Contents

9 garments × 3 manipulation actions (`fling`, `fold`, `grasp`),
**98 captures** total, captured with a Piper bimanual gripper and an
Intel RealSense D455.

| Garment | Mesh | Paper baselines |
| --- | --- | --- |
| `beige_hoodie` | ✓ | ✓ |
| `blue_dress` | ✓ | ✓ |
| `brown_coat` | ✓ | ✓ |
| `green_tshirt` | ✓ (5 k / 10 k / 20 k / 40 k) | ✓ |
| `grey_pleat_skirt` | ✓ | ✓ |
| `grey_sunwear` | ✓ | — (non-manifold, see below) |
| `khaki_blazer` | ✓ | — (non-manifold, see below) |
| `white_cakeskirt` | ✓ | ✓ |
| `white_shirt` | ✓ | ✓ |

The exact capture list lives in
[`configs/experiment_library.yaml`](../configs/experiment_library.yaml).

### Non-manifold garments

`grey_sunwear` and `khaki_blazer` ship with raw data and meshes but
**no paper baseline numbers**. Their meshes are non-manifold
(self-intersecting / non-watertight), which PyBullet, Isaac Sim, and the
MuJoCo flex body all fail to load. Researchers using particle-based or
otherwise non-manifold-tolerant simulators can still evaluate on them.

### Multi-resolution `green_tshirt`

`meshes/Green_Tshirt/` ships the same garment at four mesh densities
(5 k, 10 k, 20 k, 40 k triangles), each with both `.obj` and `.usda`.
Each has a matching cloth-params yaml
(`configs/cloth_params/green_tshirt_{5k,10k,20k,40k}.yaml`) so you can
study how mesh resolution trades off simulator fidelity against runtime
without changing anything else:

```bash
make benchmark SIM=pybullet SAMPLE=green_tshirt/grasp/02 cloth_params=green_tshirt_5k
make benchmark SIM=pybullet SAMPLE=green_tshirt/grasp/02 cloth_params=green_tshirt_40k
```

## Per-capture layout

```
<garment>/<garment>_<action>_<timestamp>/
├── calibration/
│   ├── world_to_camera_transform.json    # RealSense world frame ↔ camera frame
│   └── landmarks.json                     # hand-eye calibration landmarks
├── joints/
│   ├── left_arm_joint_states_and_end_pose.csv
│   └── right_arm_joint_states_and_end_pose.csv
└── segment_pcds/
    └── pointcloud_<timestamp>_segmented.pcd    # one file per RealSense frame
```

### Joint CSV format

Loaded by `rgbench.csv_data.load_processed_data`:

| column | meaning |
| --- | --- |
| `header.stamp.secs` + `header.stamp.nsecs` | absolute ROS timestamp |
| `position` | JSON list of joint angles (7 arm joints + 1 gripper) |
| `name` | JSON list of joint names |
| (end-pose columns) | derived 6-DoF end-effector pose, used in `fixed_point` mode |

### Point cloud format

Open3D `.pcd` (binary or ASCII), already segmented to cloth-only with
outliers removed by GroundingDINO + SAM + DBSCAN. Files are in the
camera frame; `rgbench.transforms.TransformsManager` applies the
`world_to_camera_transform.json` to bring them into the world frame at
load time.

## Calibration workflow

How `world_to_camera_transform.json` and `landmarks.json` are produced
for each capture session, and how the benchmark consumes them.

### World coordinate system

The world frame is anchored to the bimanual robot rig: origin at the
midpoint between the two robot bases, +X pointing forward from the
robot toward the workspace, +Y to the robot's right, +Z up.

```
left robot base:  ( 0, -0.25, 0)
right robot base: ( 0, +0.25, 0)
```

Both arm trajectories are expressed in this frame (after subtracting
the per-arm base translation when loading the joint CSVs).

### Camera extrinsics (`world_to_camera_transform.json`)

4×4 SE(3) matrix mapping a point in world coordinates to camera
coordinates. The inverse is applied to every loaded `.pcd` to bring
the segmented cloth points into the world frame before metric
evaluation.

Produced once per garment session via
[`tools/pcd_alignment_and_initial_estimation_manual_1.py`](../tools/pcd_alignment_and_initial_estimation_manual_1.py)
using **interactive landmark picking**:

1. Load the cloth `.obj` (world frame) and a chosen reference scan
   `.pcd` (camera frame).
2. The script opens two Open3D windows side-by-side.
3. User clicks ≥3 corresponding landmark points on the model and the
   scan (shoulders, hem corners, sleeve ends — whatever is visually
   identifiable on both).
4. The script solves the rigid-body SE(3) that aligns the chosen
   point pairs and optionally refines it with ICP on the full point
   cloud.
5. The picked-point pairs are saved as `landmarks.json` and the
   resulting transform as `world_to_camera_transform.json`.

`landmarks.json` lets anyone reproduce or audit the calibration:

```json
{
  "model_points": [[x, y, z], ...],     // clicked on the .obj (world frame)
  "scan_points":  [[x, y, z], ...]      // clicked on the .pcd (camera frame)
}
```

Re-running the script with the same picks produces the same transform
within ICP-step numerical noise.

### Optional hand-eye refinement

If a per-arm hand-eye calibration is available (ArUco-marker or
robot-arm tip touch procedure, producing
`left_base_to_camera_transform.json` and / or
`right_base_to_camera_transform.json`), the benchmark can override
the landmark-derived extrinsic with the hand-eye result:

```yaml
# configs/main.yaml
calibration:
  use_refinement: true    # default: false (use landmark-derived as-is)
```

When enabled and both arm files are present,
[`TransformsManager.refine_with_hand_eye_calibrations`](../rgbench/transforms.py)
averages the two implied extrinsics (translation mean, rotation
SLERP at t=0.5).

The v1 captures ship only `world_to_camera_transform.json` (no
hand-eye files), so `use_refinement: true` is a no-op for them — the
landmark-derived extrinsic is final.

### Initial object pose (`initial_object_pose.json`)

A 7-vector `[x, y, z, qx, qy, qz, qw]` describing where the garment
sits in the world frame at the **start** of the capture. Used by the
simulator to place the cloth mesh before stepping.

The v1 captures **do not** ship per-capture `initial_object_pose.json`
files; the benchmark falls back to a default
`[0.1, 0.0, 0.01, 1, 0, 0, 0]` defined in
[`configs/main.yaml`](../configs/main.yaml). The same alignment script
above can produce a per-capture initial pose by aligning the cloth
mesh to the first-frame point cloud — useful for irregular initial
configurations.

## Time synchronization

The RealSense D455 camera and the robot joint-state stream are
recorded into the same rosbag but each carries its own per-frame
latency. Aligning them is critical for dynamic actions: a 30 ms
offset on a fast fling can dominate the Chamfer error.

### Per-sample `camera_delay`

The paper's approach (supplementary §B.4 *Alignment of Time System*)
is a **single fixed-delay compensation** per capture session:

> Real-world camera systems introduce latency, which can cause
> significant sim-to-real discrepancies in dynamic tasks. Through
> testing, we observed that for a stable camera frame rate, this
> temporal offset is highly consistent. We therefore apply a
> fixed-delay compensation to the incoming point cloud stream,
> synchronizing the real-world data with the simulation and
> minimizing errors during dynamic manipulation.

The value used is simply **the inverse of the rosbag's effective
RealSense frame rate** — i.e. the average per-frame interval observed
in the recording's `header.stamp` sequence. Each row in
`configs/experiment_library.yaml` carries this as `camera_delay`
(seconds), and the inline comment shows the corresponding FPS:

```yaml
'02':
  camera_delay: 0.059880  # 1 / 16.7   (RealSense ran at ~16.7 fps)
  data_subfolder: green_tshirt/green_tshirt_grasp_2025-07-19-19-14-58
  evaluate: ...
```

Typical observed rates across the v1 captures (and the implied
delays): 30 fps (0.033 s), 25 fps (0.040 s), 16.7 fps (0.060 s).
Faster recordings give smaller delays.

At runtime the benchmark applies it as a constant offset before
stepping the simulator:

```python
# scripts/run_benchmark.py
compensated_target_time = target_time + cfg.camera.system_delay_time + sim_prepare_time
env.step_to_time(compensated_target_time)
```

`sim_prepare_time` is the additional warm-up that fling mode needs
to settle the cloth in the grippers before the dynamic motion
starts; for grasp and fold it is 0.

When you add a new capture, set `camera_delay` to `1 / fps` measured
on your own rosbag (e.g. `rosbag info <bag> | grep <camera_topic>`
reports the average rate).

## meshes/

```
meshes/<Garment>/<garment>[_<resolution>].{obj,usda}
```

One subdirectory per garment, self-contained. Names align with the
garment directory (lowercase). Single-resolution garments have one
`.obj` + one `.usda`; `green_tshirt` has four resolutions.

`configs/cloth_params/<garment>.yaml` resolves these paths against
`$RGBENCH_CLOTH_MESH_ROOT` (default: `./data/sample/meshes`).

## Capturing your own data

If you want to extend the dataset:

1. Record aligned RGB-D + bimanual joint states with a RealSense D455
   into a rosbag.
2. `tools/exact_image_pcd_from_rosbag.py` extracts RGB-D frames.
3. `tools/batch_segment_cloth_pcd.py` runs GroundingDINO + SAM with
   parameters from `configs/segment_realsense_pcd.yaml`.
4. `tools/batch_exact_joints_from_rosbag.py` extracts joint state CSVs.
5. Add a `configs/cloth_params/<garment>.yaml` (see
   [`docs/CONFIG.md`](CONFIG.md)) and an entry in
   `configs/experiment_library.yaml` keyed on
   `<garment>.<action>.<robot>.<sample_id>` with `data_subfolder`,
   `camera_delay`, and an evaluation window.

PR a follow-on dataset version when you have a coherent batch.

## Versioning

The repo id is `RGBench/RGBench-Cloth-Sim2Real-v1`. For breaking changes
(new captures, recalibration, different mesh decimation) we'll bump the
suffix (`v2`, `v3`, ...) so existing papers that cite v1 stay
reproducible. Hugging Face datasets are git-versioned, so any specific
revision can also be pinned via `--revision <commit-sha>` to
`download_data.py`.

## License

The dataset is released under
[CC-BY 4.0](../DATA_LICENSE) — share and adapt with attribution.
Code in this repo is MIT.
