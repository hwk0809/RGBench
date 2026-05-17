# Sample data

A single smoke-test sample lives here once you've run `scripts/download_data.py
--sample-only`. The Hugging Face dataset at `hwk0809/RGBench-Cloth-Sim2Real-v1`
provides the full set (9 garments × 3 actions × multiple samples) plus a small
reference-results subset for cross-validation against published baselines.

Expected layout after download (each subfolder corresponds to one capture):

```
data/
└── sample/
    └── green_tshirt/
        └── green_tshirt_grasp_<timestamp>/
            ├── calibration/
            │   ├── world_to_camera_transform.json
            │   └── initial_object_pose.json
            ├── joints/
            │   ├── left_arm_joint_states_and_end_pose.csv
            │   └── right_arm_joint_states_and_end_pose.csv
            └── segment_pcds/
                ├── pcd_<timestamp>.pcd
                └── ...
```

Garment meshes (used by the simulator wrappers) ship under
`data/sample/meshes/` and are referenced via the `cloth_model_path` field
in `configs/main.yaml`.

The full dataset includes additional cloth captures and a `reference_results/`
subset (≤ 50 MB) — one sample's metrics CSV + simulated point cloud frames +
comparison GIF — so external implementers can verify their pipeline matches
the published baselines within tolerance.
