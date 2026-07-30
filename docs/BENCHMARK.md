# Benchmark protocol

RGBench measures the sim-to-real gap as a per-frame disagreement between
the simulated cloth mesh and the real-world segmented point cloud at the
same timestamp.

## Pipeline (one experiment)

```
1. Load real-world segmented point clouds (timestamped) and transform
   from camera frame to world frame using world_to_camera_transform.json.
2. Initialize the chosen simulator wrapper from configs/env/<sim>.yaml.
3. For each evaluation timestep t in [start_calculate_time, end_calculate_time]:
     a. compensated_t  = t + camera.system_delay_time + sim_prepare_time
     b. sim.step_to_time(compensated_t)
     c. sim_pcd        = sim.get_sim_vertices()           # (N, 3) in world frame
     d. target_pcd     = nearest real frame by timestamp
     e. metrics_row    = compute_cost(sim_pcd, target_pcd, t)
4. Write metrics CSV + comparison GIFs to outputs/<cloth>/<action>/<sim>/...
```

The compensation in step 3a accounts for camera-pipeline latency and
optional pre-fling preparation (see `action.fling_prepare_time`).

## Metrics

| Metric | Direction | Source |
| --- | --- | --- |
| Chamfer L1 sim→real | one-way | [`rgbench.metrics.chamfer_distance_single_direction_*`](../rgbench/metrics.py) |
| Chamfer L2 sim→real | one-way | same |
| Chamfer L1 real→sim | one-way | same |
| One-sided Hausdorff sim→real | one-way | same |
| One-sided Hausdorff real→sim | one-way | same |
| Cloth stability score | self | [`calculate_cloth_sim_stability`](../rgbench/metrics.py) |
| z-axis mean error | one-way | [`run_benchmark.compute_cost`](../scripts/run_benchmark.py) |

The Chamfer functions provide both a SciPy CPU implementation and a
GPU-accelerated PyKeOps path; the GPU path is selected automatically
when CUDA and `pykeops` are available.

## Experiment registry

Every benchmark run is keyed on `(cloth, action, robot, sample_id)`,
each of which resolves to an entry in
[`configs/experiment_library.yaml`](../configs/experiment_library.yaml).
A single entry looks like:

```yaml
experiments:
  green_tshirt:
    grasp:
      piper:
        '02':
          camera_delay: 0.0598
          data_subfolder: green_tshirt/green_tshirt_grasp_2025-07-27-21-35-28
          evaluate:
            start_record_time:    0.0
            end_record_time:      4.0
            start_calculate_time: 0.0
            end_calculate_time:   3.14
```

Override any field on the command line:

```bash
python scripts/run_benchmark.py \
    params.cloth_name=green_tshirt \
    params.action_type=grasp \
    params.sample_index=02 \
    params.sim_environment=pybullet
```

## Batch runs

```bash
python scripts/run_batch.py --env pybullet
python scripts/run_batch.py --env isaacsim --filter-cloth green_tshirt
```

`run_batch.py` checkpoints per-experiment so a crash or kill doesn't
restart from scratch.

## A note on the published cloth parameters

The per-garment values in [`configs/cloth_params/*.yaml`](../configs/cloth_params/)
(stretch, bending, density, friction, damping) were fitted against the
not-yet-released GarmentDynamics simulator. The PyBullet, Isaac Sim, and
plain MuJoCo wrappers ship in this repo as reproducible reference
implementations — they are not intended as ground-truth oracles, and
their sim-to-real gap on these parameters will differ from the
GarmentDynamics result reported in the paper. When you publish a new
simulator wrapper against RGBench, retune the cloth parameters for that
simulator before drawing conclusions.

## Reference results

The Hugging Face dataset ships a `reference_results/` subset for one
canonical sample (`green_tshirt / grasp / sample 02`). External
implementers can use it to confirm their pipeline reproduces the
published numbers within tolerance before scaling to the full set.
