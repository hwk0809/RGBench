from omegaconf import DictConfig

from .base import BaseEnvWrapper


def get_env(env_cfg: DictConfig, **kwargs) -> BaseEnvWrapper:
    """Factory: build the simulator wrapper named in env_cfg.env.name.

    Built-in reference simulators in RGBench: pybullet, isaacsim, mujoco
    (plain MuJoCo flex). The closed-source GarmentDynamics simulator
    (formerly mujoco_style3d) is supported via a lazy import — install
    its companion pip package separately to enable it.
    """
    env_name = env_cfg.env.name.lower()
    sim_type = env_cfg.sim_model.get("type") if "sim_model" in env_cfg else None

    if "isaacsim" in env_name:
        if sim_type == "robot":
            from .isaacsim.cloth_isaacsim_env import IsaacsimEnv
            return IsaacsimEnv(cfg=env_cfg, **kwargs)
        if sim_type == "fixed_point":
            from .isaacsim.cloth_isaacsim_env_point import IsaacsimEnv
            return IsaacsimEnv(cfg=env_cfg, **kwargs)
        raise ValueError(f"Unknown sim_model.type for isaacsim: {sim_type!r}")

    if "pybullet" in env_name:
        if sim_type == "robot":
            from .pybullet.cloth_pybullet_env import PyBulletEnv
            return PyBulletEnv(cfg=env_cfg, **kwargs)
        if sim_type == "fixed_point":
            from .pybullet.cloth_pybullet_fixed_point_env import PyBulletFixedPointEnv
            return PyBulletFixedPointEnv(cfg=env_cfg, **kwargs)
        raise ValueError(f"Unknown sim_model.type for pybullet: {sim_type!r}")

    if "mujoco_style3d" in env_name or "garment_dynamics" in env_name:
        try:
            from rgbench_garmentdynamics import GarmentDynamicsEnv  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The GarmentDynamics simulator is not installed. "
                "RGBench ships a lazy-import slot for it; install the "
                "companion package to enable this backend. See the "
                "RGBench README for current installation instructions."
            ) from exc
        return GarmentDynamicsEnv(cfg=env_cfg, **kwargs)

    if "mujoco" in env_name:
        if sim_type == "fixed_point":
            from .mujoco.cloth_mujoco_env_fixed_point import MujocoFixedPointEnv
            return MujocoFixedPointEnv(cfg=env_cfg, **kwargs)
        raise NotImplementedError(
            "Plain MuJoCo wrapper only implements 'fixed_point' mode."
        )

    raise ValueError(f"Unknown env name: {env_name!r}")
