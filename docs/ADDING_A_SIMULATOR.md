# Adding a simulator

RGBench couples benchmark code to simulator code only through a
four-method abstract base class. Adding a new simulator means writing
one subclass and one factory branch — about an afternoon's work plus
however long it takes to wire up your simulator's stepping logic.

## The contract

[`rgbench.envs.base.BaseEnvWrapper`](../rgbench/envs/base.py) requires:

```python
class MySimEnv(BaseEnvWrapper):
    def __init__(self, cfg: DictConfig, **kwargs):
        # Initialize your simulator from cfg.env / cfg.cloth_params /
        # cfg.sim_model / cfg.data.robot_joints. Load the robot trajectory
        # CSV with rgbench.csv_data.load_processed_data.
        ...

    def step_to_time(self, target_time: float) -> None:
        # Advance the simulator to absolute time `target_time` (seconds
        # since master_start_time). Run as many internal physics steps
        # as needed.
        ...

    def get_sim_vertices(self) -> np.ndarray:
        # Return (N, 3) cloth vertex positions in world frame at the
        # current simulator time. N can be anything > 0.
        ...

    def get_master_start_time(self) -> float:
        # Absolute timestamp (typically from the first row of the joint
        # trajectory CSV) used as t=0 reference.
        ...

    def get_current_sim_time(self) -> float:
        # Current simulator time relative to master_start_time.
        ...
```

Optional: override `close()` to release simulator resources.

## Factory registration

Edit [`rgbench/envs/__init__.py`](../rgbench/envs/__init__.py) and add a
branch:

```python
elif "my_sim" in env_name:
    from .my_sim.my_sim_env import MySimEnv
    return MySimEnv(cfg=env_cfg, **kwargs)
```

Then create a `configs/env/my_sim.yaml` that mirrors the existing
`pybullet.yaml` / `isaacsim.yaml` skeletons.

## Three reference implementations to crib from

| Wrapper | File | Notes |
| --- | --- | --- |
| PyBullet | [`rgbench/envs/pybullet/cloth_pybullet_env.py`](../rgbench/envs/pybullet/cloth_pybullet_env.py) | Soft-body via `p.loadSoftBody`; bimanual Piper IK |
| Isaac Sim | [`rgbench/envs/isaacsim/cloth_isaacsim_env.py`](../rgbench/envs/isaacsim/cloth_isaacsim_env.py) | Particle-cloth on USD assets |
| plain MuJoCo | [`rgbench/envs/mujoco/cloth_mujoco_env_fixed_point.py`](../rgbench/envs/mujoco/cloth_mujoco_env_fixed_point.py) | Native MuJoCo flex, fixed-point end-effector control |

A fourth slot exists for the closed-source **GarmentDynamics** simulator
via a lazy import:

```python
elif "mujoco_style3d" in env_name or "garment_dynamics" in env_name:
    from rgbench_garmentdynamics import GarmentDynamicsEnv
    return GarmentDynamicsEnv(cfg=env_cfg, **kwargs)
```

When the companion `rgbench-garmentdynamics` pip package is installed,
this branch lights up automatically.

## Common pitfalls

- **Coordinate frame** — `get_sim_vertices` must return world-frame
  coordinates. RGBench transforms real-world point clouds into world
  frame using the per-capture calibration; if your simulator works in a
  different frame, transform inside the wrapper.
- **Time alignment** — `target_time` in `step_to_time` is wall-clock
  seconds from `master_start_time`, *not* a physics-step count. Always
  inner-loop on the simulator's native timestep until you reach the
  target.
- **Vertex count** — point-cloud metrics are robust to differing point
  counts. You don't need to match the real-world cloud's resolution.
- **Cloth parameters** — `configs/cloth_params/*.yaml` are fitted against
  GarmentDynamics. Retune for your simulator before publishing
  sim-to-real numbers (see [BENCHMARK.md](BENCHMARK.md) for the
  rationale).

## Testing

Add a smoke test under `tests/test_<my_sim>.py` modelled on
[`tests/test_pybullet.py`](../tests/test_pybullet.py): load the wrapper
on the smoke-test sample, step it for 1-2 seconds, and assert
`get_sim_vertices()` returns a sensible shape.
