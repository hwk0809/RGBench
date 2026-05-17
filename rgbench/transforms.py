import json
import dataclasses
import numpy as np
from os import path as osp
import os
from omegaconf import DictConfig
from loguru import logger
from scipy.spatial.transform import Rotation, Slerp

def make_default_eye():
    return np.eye(4)

def make_default_zeros():
    return np.zeros(3)

@dataclasses.dataclass()
class TransformsManager:
    """Manages all coordinate system transformations for a dual-arm robot setup.

    This class establishes the relationships between the world, camera, and robot
    base frames. Its primary role is to provide a single, reliable source for all
    4x4 transformation matrices needed by the application.

    A key feature is its ability to refine a baseline `world_to_camera` transform.
    It can take an initial estimate (e.g., from a manual alignment) and override it
    with a more accurate result calculated from precise, per-arm hand-eye
    calibration files. This refinement process is optional and controlled via the
    configuration.

    If no configuration is provided, the class initializes with default identity
    matrices, allowing it to be used in a "simulation" or "default" mode without
    crashing.

    Args:
        option (DictConfig, optional):
            A configuration object, typically loaded by Hydra from a YAML file.
            If `None`, the manager will use default identity matrices for all transforms.
            When provided, it is expected to contain:
            - `paths.calibration_dir` (str): The path to the folder containing
              all calibration `.json` files.
            - `calibration.use_refinement` (bool): A flag that controls the
              refinement process. If `True`, the manager will attempt to load
              `left_base_to_camera.json` and/or `right_base_to_camera.json` to
              calculate and override the main `world_to_camera_transform`.

    Attributes:
        world_to_camera_transform (np.ndarray):
            The final, possibly refined, 4x4 transform from the `{World}` frame
            to the `{Camera}` frame. This is the primary result of this class.
        camera_to_world_transform (np.ndarray):
            The calculated inverse of `world_to_camera_transform`.
        world_to_left_robot_transform (np.ndarray):
            The defined 4x4 transform from the `{World}` frame to the
            `{Left_Robot_Base}` frame. Generated programmatically.
        world_to_right_robot_transform (np.ndarray):
            The defined 4x4 transform from the `{World}` frame to the
            `{Right_Robot_Base}` frame. Generated programmatically.
        left_robot_to_world_transform (np.ndarray):
            The calculated inverse of `world_to_left_robot_transform`.
        right_robot_to_world_transform (np.ndarray):
            The calculated inverse of `world_to_right_robot_transform`.
        left_robot_base_pos (np.ndarray):
            The calculated 3D position [x, y, z] of the left robot's base
            in the `{World}` frame.
        right_robot_base_pos (np.ndarray):
            The calculated 3D position [x, y, z] of the right robot's base
            in the `{World}` frame.
    """
    option: DictConfig = dataclasses.field(default=None)
    world_to_camera_transform: np.ndarray = dataclasses.field(default_factory=make_default_eye)
    camera_to_world_transform: np.ndarray = dataclasses.field(default_factory=make_default_eye)
    world_to_left_robot_transform: np.ndarray = dataclasses.field(default_factory=make_default_eye)
    world_to_right_robot_transform: np.ndarray = dataclasses.field(default_factory=make_default_eye)
    left_robot_to_world_transform: np.ndarray = dataclasses.field(default_factory=make_default_eye)
    right_robot_to_world_transform: np.ndarray = dataclasses.field(default_factory=make_default_eye)
    left_robot_base_pos: np.ndarray = dataclasses.field(default_factory=make_default_zeros)
    right_robot_base_pos: np.ndarray = dataclasses.field(default_factory=make_default_zeros)

    def __post_init__(self):
        """
        After initialization, loading and calculations are performed
        only if a configuration option is provided.
        """

        # TODO： This should be replaced to you actual base transform
        # --- Generate Robot Transforms ---
        logger.info("⚙️  Generating fixed robot poses programmatically...")
        self.world_to_left_robot_transform[1, 3] = -0.25
        self.world_to_right_robot_transform[1, 3] = 0.25

        if self.option is not None:
            logger.info("Configuration detected. Initializing transform matrices...")

            # --- Load Camera Transform (from file) ---
            calibration_path = self.option.calibration.path
            world_to_camera_file = os.path.join(calibration_path, 'world_to_camera_transform.json')
            try:
                with open(world_to_camera_file, 'r') as f:
                    self.world_to_camera_transform = np.array(json.load(f))
                logger.success(f"Loaded camera calibration from: {world_to_camera_file}")
            except FileNotFoundError:
                logger.warning(
                    f"Camera calibration file not found at '{world_to_camera_file}'. Using default identity matrix.")

            # --- Refinement Step ---
            # Check the config file to see if we should refine the transform.
            if self.option.calibration.use_refinement:
                logger.info("Refinement enabled. Attempting to override 'world_to_camera' with hand-eye results.")
                self.refine_with_hand_eye_calibrations(calibration_path)
            else:
                logger.info("Refinement disabled. Using 'world_to_camera.json' as is.")

            # --- Unify calculation of all derived data ---
            logger.info("Calculating all derived transforms...")
            self.camera_to_world_transform = np.linalg.inv(self.world_to_camera_transform)
            self.left_robot_to_world_transform = np.linalg.inv(self.world_to_left_robot_transform)
            self.right_robot_to_world_transform = np.linalg.inv(self.world_to_right_robot_transform)
            self.left_robot_base_pos = self.world_to_left_robot_transform[:3, 3]
            self.right_robot_base_pos = self.world_to_right_robot_transform[:3, 3]

            logger.success("TransformsManager initialization complete.")
        else:
            logger.info("No configuration provided. TransformsManager will use default identity matrices.")

    def refine_with_hand_eye_calibrations(self, calibration_path: str):
        """
        Loads hand-eye calibration files for bi-manual arms and uses them to calculate a more accurate
        world_to_camera_transform, overriding the initial one.
        """
        # Load left arm hand-eye calibration result
        left_base_to_camera, t_w_c_from_left = None, None
        try:
            with open(os.path.join(calibration_path, 'left_base_to_camera_transform.json'), 'r') as f:
                left_base_to_camera = np.array(json.load(f))
            logger.success("Loaded left_base_to_camera.json for refinement.")
            # T_w_c = T_w_lb @ T_lb_c
            t_w_c_from_left = self.world_to_left_robot_transform @ left_base_to_camera
        except FileNotFoundError:
            logger.warning("Left arm hand-eye calibration file not found. Skipping refinement from left arm.")

        # (Optional but recommended) Load right arm hand-eye calibration result
        right_base_to_camera, t_w_c_from_right = None, None
        try:
            with open(os.path.join(calibration_path, 'right_base_to_camera_transform.json'), 'r') as f:
                right_base_to_camera = np.array(json.load(f))
            logger.success("Loaded right_base_to_camera.json for refinement.")
            # T_w_c = T_w_rb @ T_rb_c
            t_w_c_from_right = self.world_to_right_robot_transform @ right_base_to_camera
        except FileNotFoundError:
            logger.warning("Right arm hand-eye calibration file not found. Skipping refinement from right arm.")

        # --- Decision Logic ---
        if t_w_c_from_left is not None and t_w_c_from_right is not None:
            logger.info("Both hand-eye calibrations found. Averaging them for best result.")
            # Simple averaging for translation
            trans_left = t_w_c_from_left[:3, 3]
            trans_right = t_w_c_from_right[:3, 3]
            avg_trans = (trans_left + trans_right) / 2.0

            # Slerp (Spherical Linear Interpolation) for rotation
            rot_left = Rotation.from_matrix(t_w_c_from_left[:3, :3])
            rot_right = Rotation.from_matrix(t_w_c_from_right[:3, :3])
            slerp = Slerp([0, 1], Rotation.create_group([rot_left, rot_right]))
            avg_rot = slerp(0.5).as_matrix()

            # Combine into a new transform
            self.world_to_camera_transform = np.eye(4)
            self.world_to_camera_transform[:3, :3] = avg_rot
            self.world_to_camera_transform[:3, 3] = avg_trans
            logger.success("Successfully refined 'world_to_camera' by averaging both arm calibrations.")

        elif t_w_c_from_left is not None:
            logger.info("Only left hand-eye calibration found. Using it to refine 'world_to_camera'.")
            self.world_to_camera_transform = t_w_c_from_left
            logger.success("Successfully refined 'world_to_camera' using left arm calibration.")

        elif t_w_c_from_right is not None:
            logger.info("Only right hand-eye calibration found. Using it to refine 'world_to_camera'.")
            self.world_to_camera_transform = t_w_c_from_right
            logger.success("Successfully refined 'world_to_camera' using right arm calibration.")
        else:
            logger.error("Refinement was enabled, but no hand-eye calibration files were found. No refinement applied.")