import abc
import numpy as np
from omegaconf import DictConfig


class BaseEnvWrapper(abc.ABC):
    """
    Abstract Base Class for Simulation Environments.
    This defines a common interface that all environment wrappers (e.g., for MuJoCo, PyBullet)
    must implement.
    """

    @abc.abstractmethod
    def __init__(self, cfg: DictConfig, **kwargs):
        """
        Initializes the simulation environment.

        Args:
            cfg (DictConfig): The configuration specific to this environment (e.g., from cfg.env).
        """
        self.cfg = cfg
        pass

    @abc.abstractmethod
    def step(self):
        """
        Steps the simulation forward by one time step.
        This method should handle all internal simulation steps (e.g., mujoco.mj_step or any backend-specific stepper)
        and robot controller updates.

        Returns:
            dict: A dictionary containing the current state of the environment, including
                  observations, rewards, done flags, and any additional information.
        """
        pass

    @abc.abstractmethod
    def step_to_time(self, target_time: float) -> None:
        """
        Steps the simulation forward until it reaches the specified target time.
        This method should handle all internal simulation steps (e.g., mujoco.mj_step or any backend-specific stepper)
        and robot controller updates.

        Args:
            target_time (float): The target simulation time to reach (relative time).
        """
        pass

    @abc.abstractmethod
    def get_sim_vertices(self) -> np.ndarray:
        """
        Retrieves the world coordinates of the vertices of the simulated soft body (e.g., cloth)
        at the current time step.

        Returns:
            np.ndarray: An (N, 3) numpy array of vertex positions.
        """
        pass


    @abc.abstractmethod
    def get_master_start_time(self) -> float:
        """
        Gets the absolute master start timestamp, which is often determined by the environment
        (e.g., from the start of a data playback). This serves as the time baseline for the entire test.

        Returns:
            float: The absolute start timestamp.
        """
        pass

    @abc.abstractmethod
    def get_current_sim_time(self) -> float:
        """
        Gets the current internal time of the simulation.

        Returns:
            float: The current simulation time.
        """
        pass

    def close(self):
        """
        Optional: Cleans up any resources used by the environment.
        """
        pass