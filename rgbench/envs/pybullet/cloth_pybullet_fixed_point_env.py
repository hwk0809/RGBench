import os
import os.path as osp
import time
import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data
import ast
from scipy.spatial.transform import Rotation, Slerp
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from typing import Dict, Tuple, Optional, List, Any, Literal, TypedDict, cast

from rgbench.csv_data import load_processed_data, PoseState
from rgbench.envs.base import BaseEnvWrapper

ArmType = Literal["left", "right"]


# the gripper box URDF path
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
except NameError:
    PROJECT_ROOT = os.getcwd()
GRIPPER_LEFT_URDF_PATH = os.path.join(PROJECT_ROOT, "assets","Urdf", "box_gripper", "simple_box.urdf")
GRIPPER_RIGHT_URDF_PATH = os.path.join(PROJECT_ROOT, "assets", "Urdf","box_gripper", "simple_box_2.urdf")

class GrabberState(TypedDict, total=False):
    is_active: bool
    is_grabbed: bool        # Flag to indicate if the grab time has passed
    grab_time: float
    pose_data: List[PoseState]
    pose_tracker: Dict[str, int]
    body_id: int
    vertex_id: int          #  Stores the anchored cloth vertex ID
    initial_grab_vertex_pose: Optional[List[float]]



class PyBulletFixedPointEnv(BaseEnvWrapper):
    def __init__(self, cfg: DictConfig, **kwargs):
        super().__init__(cfg)
        self.env_cfg = cfg.env
        self.cloth_cfg = cfg.cloth
        self.cloth_params = cfg.cloth_params
        self.data_cfg = cfg.data
        self.grab_points_cfg = self.env_cfg.get('grab_points', {})

        self.sim_timestep = self.env_cfg.get("timestep", 0.002)
        self._sim_time = 0.0

        self.left_base_offset = [0, 0.25, 0]
        self.right_base_offset = [0, -0.25, 0]

        self.grabbers: Dict[ArmType, GrabberState] = {}
        self.physics_client = -1

        # --- STAGE 0: Determine Action Type and Parameters ---
        self.action_type = self.cfg.action.get('type', 'grasp')  # Default to 'grasp' for backward compatibility
        logger.info(f"Action type selected: '{self.action_type}'")

        self.grab_time = float('inf')
        self.fling_prepare_time = 0.0

        if self.action_type == 'grasp' or self.action_type == 'fold':
            if 'evaluate' in self.cfg and 'start_calculate_time' in self.cfg.evaluate:
                self.grab_time = self.cfg.evaluate.start_calculate_time
                logger.info(f"Using grasp time from 'cfg.evaluation.start_calculate_time': {self.grab_time}s")
            elif 'grasp' in self.env_cfg and 'grasp_time' in self.env_cfg.grasp:
                self.grab_time = self.env_cfg.grasp.grasp_time
                logger.info(f"Using grasp time from 'cfg.env.grasp_time': {self.grab_time}s.")
            else:
                logger.warning("No grasp time specified for 'grasp' action. Grabbing will be disabled.")

        elif self.action_type == 'fling':
            self.grab_time = 0.0  # In fling mode, grabbing is always active from the start.
            self.fling_prepare_time = self.cfg.action.get('fling_prepare_time', 2.0)
            self.fling_wait_time = self.cfg.action.get('fling_wait_time', 3.0)
            logger.info("Fling mode configured:")
            logger.info(f"  - Initial Grasp Duration: {self.fling_prepare_time}s")
            logger.info(f"  - Pre-Fling Wait Time:    {self.fling_wait_time}s")



        # --- STAGE 1: Initialize the main simulation environment ---
        logger.info("--- [Stage 2/3] Initializing main PyBullet simulation environment ---")
        use_gui = True
        if "visualization" in cfg and cfg.visualization.get('vis_sim', False):
            use_gui = True 
        if use_gui:
            self.physics_client = p.connect(p.GUI)
            logger.info("PyBullet GUI mode enabled.")
        else:
            self.physics_client = p.connect(p.DIRECT)
            logger.info("PyBullet DIRECT mode enabled (no GUI).")

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=self.sim_timestep)

        # --- STAGE 2: Set up the scene and load data ---
        logger.info("--- [Stage 3/3] Setting up scene, creating static anchors, and loading data ---")
        self._load_scene()
        logger.info("Letting the cloth settle...")
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.physics_client)
        logger.info("Cloth has settled.")

        # --- STAGE 3: Set up the anchor point  ---
        pin_indices = self._initialize_anchor_point()

        # --- STAGE 4: Initialize the trajectory data  ---
        self._init_grabbers_data([self.left_grabber_id,self.right_grabber_id], pin_indices)

        # --- STAGE 4:Set the global playback timeline ---
        all_start_times = [g['pose_data'][0]['time'] for g in self.grabbers.values() if g.get('is_active')]
        all_end_times = [g['pose_data'][-1]['time'] for g in self.grabbers.values() if g.get('is_active')]

        if not all_start_times:
            raise ValueError("Simulation cannot start: No active grabbers with data were found.")

        self.playback_start_time_abs = min(all_start_times)
        self.playback_end_time_abs = max(all_end_times)
        duration = self.playback_end_time_abs - self.playback_start_time_abs

        logger.info("-" * 40)
        logger.info("Global Playback Timeline Determined:")
        logger.info(f"  - Absolute Data Start Time: {self.playback_start_time_abs:.4f}s")
        logger.info(f"  - Absolute Data End Time:   {self.playback_end_time_abs:.4f}s")
        logger.info(f"  - Total Duration:           {duration:.2f}s")
        logger.info("-" * 40)
        logger.success("PyBulletFixedPointEnv (Mujoco-Style) initialized successfully.")

    # ===================================================================
    # Initial Method
    # ===================================================================
    def _load_scene(self):

        p.loadURDF("plane.urdf", physicsClientId=self.physics_client)

        cloth_obj_path = self.cloth_cfg.model_path
        init_pos = self.cloth_cfg.init_pose[:3]
        init_quat_wxyz = self.cloth_cfg.init_pose[3:]
        init_quat_xyzw = [init_quat_wxyz[1], init_quat_wxyz[2], init_quat_wxyz[3], init_quat_wxyz[0]]

        # for NeoHookean model, but it used for caoutchouc
        # cloth_E = self.cloth_cfg.stretch # Young's modulus for the cloth
        # cloth_ν = self.cloth_cfg.poisson  # Poisson's ratio for the cloth
        # cloth_mu = cloth_E/ (2 * (1 + cloth_ν))  # Shear modulus
        # cloth_lambda = (cloth_E * cloth_ν) / ((1 + cloth_ν) * (1 - 2*cloth_ν))


        self.cloth_id = p.loadSoftBody(
            fileName=cloth_obj_path,
            basePosition=init_pos,
            baseOrientation=init_quat_xyzw,
            scale=1.0,
            mass=float(self.cloth_params.mass),
            collisionMargin=0.003,
            useSelfCollision=True,
            useBendingSprings=True,
            useFaceContact=True,
            useMassSpring=True,
            # useNeoHookean=True,
            # NeoHookeanMu=float(cloth_mu),
            # NeoHookeanLambda=float(cloth_lambda),
            # NeoHookeanDamping=float(self.cloth_cfg.damping),
            springElasticStiffness=float(self.cloth_params.stretch),
            springDampingStiffness=float(self.cloth_params.damping),
            springBendingStiffness=float(self.cloth_params.bending),
            frictionCoeff=float(self.cloth_params.friction),
            physicsClientId=self.physics_client
        )

        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=90,
            cameraPitch=-65,
            cameraTargetPosition=[0, 0, 0]
        )

        # p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25, physicsClientId=self.physics_client)

    def _init_grabbers_data(self, body_ids_list: List[int], pin_indices_list: List[int]):
        """Load trajectory data and populate the self.grabbers dictionary."""

        arm_order: List[ArmType] = ['left', 'right']
        initial_vertex_pos = self.get_sim_vertices()

        # Loop based on the fixed order to initialize each arm's state
        for i, arm_name in enumerate(arm_order):
            pose_csv_path = self.data_cfg.robot_joints.get(f'{arm_name}_arm_csv_path')

            # If a CSV path for an arm doesn't exist, we skip it.
            if not pose_csv_path:
                logger.warning(f"No CSV path for '{arm_name}', skipping its initialization.")
                continue

            base_offset = self.left_base_offset if arm_name == 'left' else self.right_base_offset
            pose_data = load_processed_data(pose_csv_path, mode='pose',
                                            start_time_offset=self.data_cfg.get('sim_start_time', 0.0),
                                            base_translation=base_offset
                                            )
            if not pose_data:
                logger.error(f"Could not load trajectory data for '{arm_name}', it will be disabled.")
                continue

            self.grabbers[arm_name] = GrabberState(
                is_active=True,
                body_id=body_ids_list[i],
                vertex_id=pin_indices_list[i],
                pose_data=pose_data,
                grab_time=self.grab_time,
                is_grabbed=False,
                pose_tracker={'idx': 0},
                initial_grab_vertex_pose= initial_vertex_pos[pin_indices_list[i]].tolist()
            )
            logger.info(
                f" -> Loaded {len(pose_data)} pose data points for '{arm_name}'. Grab time set to {self.grab_time}s.")

    # ===================================================================
    # Anchor Point Initialization
    # ===================================================================
    def _determine_pin_indices(self) -> List[int]:
        """Determines the vertex indices to pin based on the action type and mode."""
        if self.action_type == 'fling':
            pin_indices = self.cfg.cloth_params.get('shoulder_index')
            if not pin_indices or len(pin_indices) == 0:
                raise ValueError("Fling mode requires 'shoulder_index' in 'cloth_params'.")
            logger.info(f"Fling mode: Using specified pin indices: {pin_indices}")
            return pin_indices

        elif self.action_type == 'grasp' or self.action_type == 'fold':
            mode = self.grab_points_cfg.get('mode', 'auto')
            logger.info(f"Grasp mode: Pinning enabled with mode: '{mode}'")
            if mode == 'manual':
                pin_indices = self.grab_points_cfg.get('indices')
                if not pin_indices:
                    raise ValueError("Manual mode requires 'indices' in config.")
                logger.info(f"Using manually specified pin indices: {pin_indices}")
                return pin_indices
            elif mode == 'auto':
                if self.grab_time == float('inf'):
                    raise ValueError("Cannot run 'auto' pin mode without a valid 'grab_time'.")
                pin_indices = self._calculate_auto_indices()
                logger.success(f"Automatically determined pin indices: {pin_indices}")
                return pin_indices
            else:
                raise ValueError(f"Unknown pinning mode: '{mode}'. Must be 'auto' or 'manual'.")
        else:
            raise ValueError(f"Unknown action type: '{self.action_type}'.")

    def _initialize_anchor_point(self):

        # 1. Determine the pin indices based on the action type and mode
        pin_indices = self._determine_pin_indices()

        # 2. get anchor vertices positions
        self.left_anchor_vertex = pin_indices[0]
        self.right_anchor_vertex = pin_indices[1]
        all_verts_pos = self.get_sim_vertices()
        if all_verts_pos.size == 0:
            raise RuntimeError("Failed to get vertex positions after cloth settling.")
        left_anchor_verts_pos = all_verts_pos[self.left_anchor_vertex]
        right_anchor_verts_pos = all_verts_pos[self.right_anchor_vertex]

        # 5. load anchor objects and create anchors
        arm_order: List[ArmType] = ['left', 'right']
        self.left_grabber_id = p.loadURDF(GRIPPER_LEFT_URDF_PATH, basePosition=[0,0,0],useMaximalCoordinates=True)
        self.right_grabber_id = p.loadURDF(GRIPPER_RIGHT_URDF_PATH, basePosition=[0,0,0],useMaximalCoordinates=True)

        p.resetBasePositionAndOrientation(self.left_grabber_id, left_anchor_verts_pos, [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.right_grabber_id, right_anchor_verts_pos, [0, 0, 0, 1])

        p.createSoftBodyAnchor(self.cloth_id, self.left_anchor_vertex, self.left_grabber_id, -1, [0, 0, 0.0])
        logger.success(f"Created static anchor for 'left': Vertex #{self.left_anchor_vertex},vertex position {left_anchor_verts_pos}")
        p.createSoftBodyAnchor(self.cloth_id, self.right_anchor_vertex, self.right_grabber_id, -1, [0, 0, 0.0])
        logger.success(f"Created static anchor for 'right': Vertex #{self.right_anchor_vertex}, vertex position {right_anchor_verts_pos}")


        return pin_indices

    def _calculate_auto_indices(self) -> List[int]:
        logger.info("Calculating indices for 'auto' mode in the current simulation...")

        # 1.get stable vertices positions
        initial_verts_pos = self.get_sim_vertices()
        if initial_verts_pos.size == 0:
            raise RuntimeError("Auto-calc failed: Could not retrieve cloth vertex data.")

        # 2.load date to get anchor point
        pose_datas = {}
        temp_start_times = []
        arm_order: List[ArmType] = ['left', 'right']
        for arm_name in arm_order:
            pose_csv_path = self.data_cfg.robot_joints.get(f'{arm_name}_arm_csv_path')
            base_offset = self.left_base_offset if arm_name == 'left' else self.right_base_offset
            if pose_csv_path:
                pose_data = load_processed_data(pose_csv_path, mode='pose',
                                                start_time_offset=self.data_cfg.get('sim_start_time', 0.0),
                                                base_translation=base_offset)
                if pose_data:
                    pose_datas[arm_name] = pose_data
                    temp_start_times.append(pose_data[0]['time'])

        if not temp_start_times:
            raise ValueError("Auto-calc failed: Could not find any valid trajectory data.")

        # 3. calculate the grab time and target position
        playback_start_time_abs = min(temp_start_times)
        if self.grab_time is None:
            raise ValueError("Auto mode requires 'grasp' in config.")
        target_absolute_time = playback_start_time_abs + self.grab_time

        pin_indices_list: List[int] = []
        for arm_name in arm_order:
            if arm_name not in pose_datas:
                raise ValueError(f"Auto mode requires active data for arm '{arm_name}'.")

            interpolated_pose = self._get_interpolated_pose(target_absolute_time, pose_datas[arm_name], {'idx': 0})
            if not interpolated_pose:
                raise RuntimeError(f"Could not interpolate pose for '{arm_name}'.")

            grabber_target_pos = np.array(interpolated_pose['position'])
            logger.info(f" -> For '{arm_name}', target position at grab time: {grabber_target_pos}")
            distances = np.linalg.norm(initial_verts_pos - grabber_target_pos, axis=1)
            closest_vertex_idx = int(np.argmin(distances))
            pin_indices_list.append(closest_vertex_idx)
            logger.info(f" -> For '{arm_name}', found closest vertex #{closest_vertex_idx}")

        return pin_indices_list

    # ===================================================================
    # BaseEnvWrapper
    # ===================================================================
    def step(self):
        if self.action_type == 'grasp' or self.action_type == 'fold':
            self._step_grasp()
        elif self.action_type == 'fling':
            self._step_fling()
        else:
            pass
        p.stepSimulation(physicsClientId=self.physics_client)
        self._sim_time += self.sim_timestep

    def _step_grasp(self):
        current_absolute_time = self.get_master_start_time() + self._sim_time

        for arm_name, grabber in self.grabbers.items():
            if not grabber.get('is_active'):
                continue

            # Check if the grab time has been reached
            if not grabber['is_grabbed'] and self._sim_time >= grabber['grab_time']:
                grabber['is_grabbed'] = True
                logger.success(f"Grab triggered: '{arm_name}' starts moving at sim time {self._sim_time:.4f}s.")

            # Only move the grabber after the grab is triggered
            if not grabber['is_grabbed']:
                continue

            target_pose = self._get_interpolated_pose(current_absolute_time, grabber['pose_data'],
                                                      grabber['pose_tracker'])
            if not target_pose:
                continue

            target_pos = np.array(target_pose['position'])
            target_orn_wxyz = np.array(target_pose['orientation'])

            p.resetBasePositionAndOrientation(
                grabber['body_id'],
                target_pos,
                # target_orn_xyzw,
                [0, 0, 0, 1],
                physicsClientId=self.physics_client
            )


    def _step_fling(self):
        sim_time = self._sim_time
        for arm_name, grabber in self.grabbers.items():
            if not grabber.get('is_active') or 'vertex_id' not in grabber:
                continue

            # Phase 1: Initial Grasp
            if sim_time < self.fling_prepare_time:
                alpha = sim_time / self.fling_prepare_time if self.fling_prepare_time > 0 else 1
                start_pos = np.array(grabber['initial_grab_vertex_pose'])
                target_pos = np.array(grabber['pose_data'][0]['position'])
                current_pos = start_pos + alpha * (target_pos - start_pos)
                current_quat = np.array(grabber['pose_data'][0]['orientation'])

            # Phase 2: Pre-Fling Wait
            elif sim_time < self.fling_wait_time + self.fling_prepare_time:
                first_pose = grabber['pose_data'][0]
                current_pos = np.array(first_pose['position'])
                current_quat = np.array(first_pose['orientation'])

            # Phase 3: Fling Execution
            else:
                time_offset = self.fling_wait_time + self.fling_prepare_time
                adjusted_sim_time = sim_time - time_offset
                current_absolute_time = self.get_master_start_time() + adjusted_sim_time

                pose = self._get_interpolated_pose(
                    current_absolute_time, grabber['pose_data'], grabber['pose_tracker']
                )

                if pose:
                    current_pos = np.array(pose['position'])
                    current_quat = np.array(pose['orientation'])
                else:
                    last_pose = grabber['pose_data'][-1]
                    current_pos = np.array(last_pose['position'])
                    current_quat = np.array(last_pose['orientation'])


            # Apply the calculated pose to mocap and vertex
            current_orn_xyzw = current_quat[[1, 2, 3, 0]]
            p.resetBasePositionAndOrientation(
                grabber['body_id'],
                current_pos,
                [0,0,0,1],  # Use identity quaternion for no rotation
                # current_orn_xyzw,
                physicsClientId=self.physics_client
            )


    def get_master_start_time(self) -> float:
        return self.playback_start_time_abs or 0.0

    def get_current_sim_time(self) -> float:
        return self._sim_time

    def get_sim_vertices(self) -> np.ndarray:
        try:
            mesh_data = p.getMeshData(self.cloth_id,  flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.physics_client)
            return np.array(mesh_data[1]) if mesh_data and mesh_data[0] > 0 else np.array([])
        except p.error:
            return np.array([])

    def step_to_time(self, target_time: float):
        while self._sim_time < target_time:
            self.step()

    def close(self):
        p.disconnect(self.physics_client)
        logger.info("Disconnected from PyBullet.")

    def _get_interpolated_pose(self, target_time: float, pose_data: List[PoseState], tracker: Dict[str, int]) -> \
            Optional[PoseState]:
        """Interpolates pose from a specific data list and tracker."""
        # This helper function is now self-contained and operates on passed-in data
        if not pose_data: return None

        while (tracker['idx'] < len(pose_data) - 2 and pose_data[tracker['idx'] + 1]['time'] < target_time):
            tracker['idx'] += 1

        if target_time <= pose_data[0]['time']: return pose_data[0]
        if target_time >= pose_data[-1]['time']: return pose_data[-1]

        p0, p1 = pose_data[tracker['idx']], pose_data[tracker['idx'] + 1]
        t0, pos0, quat0 = p0['time'], np.array(p0['position']), np.array(p0['orientation'])
        t1, pos1, quat1 = p1['time'], np.array(p1['position']), np.array(p1['orientation'])

        interval = t1 - t0
        if interval <= 1e-6: return p0
        alpha = (target_time - t0) / interval

        inter_pos = pos0 + alpha * (pos1 - pos0)

        # slerp for quaternion interpolation
        quat0_xyzw = quat0[[1, 2, 3, 0]]
        quat1_xyzw = quat1[[1, 2, 3, 0]]
        key_times = [t0, t1]
        key_rots = Rotation.from_quat([quat0_xyzw, quat1_xyzw])
        slerp = Slerp(key_times, key_rots)
        interp_rot = slerp(target_time)
        inter_quat_xyzw = interp_rot.as_quat()
        inter_quat_wxyz = inter_quat_xyzw[[3, 0, 1, 2]]

        return {'time': target_time, 'position': inter_pos.tolist(), 'orientation': inter_quat_wxyz.tolist()}

    def run(self):
        """Main loop for interactive playback."""
        if self.playback_start_time_abs is None:
            logger.error("Playback time not set. Cannot run the simulation.")
            return

        end_sim_time = self.playback_end_time_abs - self.playback_start_time_abs
        while True:
            try:
                if self._sim_time < end_sim_time:
                    self.step()
                else:
                    p.stepSimulation()

                # time.sleep(self.sim_timestep)  #
            except p.error:
                logger.info("PyBullet GUI close。")
                break

    def record_video(
            self,
            output_path: str = None,
            speed: float = 1.0,
            fps: int = 60,
            width: int = 1920,
            height: int = 1080,
            camera_name: Optional[str] = None
    ):
        """
        Run the simulation and record the video, you can control the playback speed with the speed parameter.
        Args:
            output_path (str): path of the output video file (e.g. "output.mp4").
            speed (float, optional): playback speed.
                                    1.0 = real time (default).
                                    2.0 = 2x speed.
                                    0.5 = 0.5x slow playback.
                                    Default is 1.0.
            fps (int, optional): Frame rate of the video. Default is 60.
            width (int, optional): width of the video. Default is 1920.
            height (int, optional): height of the video. Default is 1080.
            camera_name: Optional[str]: Default camera viewpoint (not used in PyBullet)
        """
        import imageio
        from tqdm import tqdm

        print(f"--- Preparing to record video to '{output_path}' with interpolation ---")
        if speed <= 0:
            raise ValueError("Speed must be a positive number.")

        if camera_name:
            print(f"Using named camera: '{camera_name}' (Note: PyBullet uses fixed camera settings)")
        else:
            print("Using default camera settings.")

        if self.playback_start_time_abs is None or self.playback_end_time_abs is None:
            print("❌ Error: Playback start or end time is not set. Cannot record video.")
            return

        data_duration = self.playback_end_time_abs - self.playback_start_time_abs
        if data_duration <= 0:
            print("❌ Error: No data found.")
            return

        if output_path == "default" or output_path is None:
            output_path = (f"{self.data_cfg.root}/video/simulation_{self.env_cfg.name}_{self.cloth_cfg.name}"
                           f"_step_{self.sim_timestep}.mp4")
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # --- 1. Initialize simulation and rendering parameters ---
        sim_dt = self.sim_timestep

        # Calculate how many physical steps need to be performed in total
        total_sim_steps = int(data_duration / sim_dt)

        # Calculate the target duration and total frames of the video
        target_video_duration = data_duration / speed
        total_video_frames = int(target_video_duration * fps)

        # Calculate the time between rendering two frames (in simulation time)
        if total_video_frames > 0:
            render_interval = data_duration / total_video_frames
        else:
            render_interval = float('inf')

        print(f"  - Total simulation steps: {total_sim_steps}")
        print(f"  - Will render a frame every {render_interval:.4f} simulation seconds.")
        print(f"  - Target video duration: {target_video_duration:.2f} seconds")
        print(f"  - Total video frames: {total_video_frames}")

        # Reset simulation state
        self._sim_time = 0.0
        self.time_trackers = {'left': {'idx': 0}, 'right': {'idx': 0}}

        # Configure PyBullet camera for video recording
        # Set up camera parameters for better video quality
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.physics_client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.physics_client)
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0, physicsClientId=self.physics_client)
        p.configureDebugVisualizer(p.COV_ENABLE_VR_PICKING, 0, physicsClientId=self.physics_client)
        p.configureDebugVisualizer(p.COV_ENABLE_VR_TELEPORTING, 0, physicsClientId=self.physics_client)

        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.physics_client)
        frames = []
        next_render_time = 0.0

        try:
            # --- 2. Main loop: Evolution of the entire simulation in physical steps ---
            for step in tqdm(range(total_sim_steps), desc="Simulating Physics"):
                # --- 3. Rendering judgment: check if the current simulation time has reached the moment to render the next frame ---
                if self._sim_time >= next_render_time:
                    # Capture frame from PyBullet
                    view_matrix = p.computeViewMatrix(
                        cameraEyePosition=[2.0, 0, 1.5],
                        cameraTargetPosition=[0.5, 0, 0],
                        cameraUpVector=[0, 0, 1],
                        physicsClientId=self.physics_client
                    )

                    projection_matrix = p.computeProjectionMatrixFOV(
                        fov=60,
                        aspect=width / height,
                        nearVal=0.1,
                        farVal=50,
                        physicsClientId=self.physics_client
                    )

                    # Get camera image
                    (_, _, px, depth, seg) = p.getCameraImage(
                        width=width,
                        height=height,
                        viewMatrix=view_matrix,
                        projectionMatrix=projection_matrix,
                        renderer=p.ER_BULLET_HARDWARE_OPENGL,  # use hardware renderer
                        lightDirection=[1, 1, 1],  # light direction
                        lightColor=[1, 1, 1],  # light color
                        lightDistance=2,  # light distance
                        shadow=1,  # enable shadows
                        physicsClientId=self.physics_client
                    )

                    # Convert to RGB format for video
                    rgb_array = np.array(px, dtype=np.uint8)
                    rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
                    frames.append(rgb_array)
                    next_render_time += render_interval

                # --- 4. Physical update ---
                self.step()

            # --- 5. Save Video ---
            print(f"\nSimulation complete. Saving {len(frames)} rendered frames to video...")
            try:
                with imageio.get_writer(
                        output_path,
                        fps=fps,
                        codec='libx264',
                        quality=9,  # high quality (0-10, 10 is highest)
                        pixelformat='yuv420p',  # widely compatible pixel format
                        ffmpeg_params=['-crf', '18']  # video quality parameter, 18 is high quality
                ) as writer:
                    for frame in tqdm(frames, desc="Saving Video"):
                        writer.append_data(frame)
                print(f"✅ Video successfully saved to '{output_path}'")
            except Exception as e:
                print(f"❌ Error saving video: {e}")

        except Exception as e:
            print(f"❌ Error during video recording: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Re-enable GUI if it was disabled
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=self.physics_client)
            print("Video recording completed.")


if __name__ == "__main__":
    print("Running PyBulletFixedPointEnv test in standalone mode...")

    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir))
    DEFAULT_SAMPLE_ROOT = os.path.join(REPO_ROOT, "data", "sample")
    DATA_ROOT = os.environ.get("RGBENCH_DATA_ROOT", DEFAULT_SAMPLE_ROOT)
    CLOTH_MESH_ROOT = os.environ.get("RGBENCH_CLOTH_MESH_ROOT", os.path.join(DEFAULT_SAMPLE_ROOT, "cloth_meshes"))
    data_path = os.environ.get("RGBENCH_PIPER_DATA", os.path.join(DATA_ROOT, "Piper_Data"))


    active_run_for_debug = {
        'action': {
            "type": "fold",  # "grasp" or "fling" or "fold"
            "fling_prepare_time": 1.0,  # Only used in fling mode
            "fling_wait_time": 1.0,  # Only used in fling mode
        },
        'cloth': {
            'name': 'white_cakeskirt',
            'model_path': os.path.join(CLOTH_MESH_ROOT, 'CakeSkirt_Flat_Simple_30k.obj'),
            'init_pose': [0.1, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0],  # [x, y, z, w, qx, qy, qz]
        },
        'cloth_params': {
            'shoulder_index': [185, 76],  # Indices of the vertices to pin in fling mode
            'mass': 0.255,
            'stretch': 230000,
            'bending': 1000.0,
            'friction': 0.2,  # friction coefficient
            'poisson': 0.35,
            'damping': 0.0,  # damping coefficient
        },
        'env': {
            'name': 'pybullet_fixed_point',
            'timestep': 1. / 500.,  # smaller timestep is recommended for soft body simulation
            'grasp': {
                'enabled': True,  # Enable pinning of cloth vertices
                'grasp_time': 8.5,  # Time when the left arm grabs the cloth
                'mode': 'auto',  # 'auto' or 'manual'
                # 'indices': [2457, 2098] # Only used if mode is 'manual'
                'indices': [4351, 6071]  # Only used if mode is 'manual'
            }
        },
        'data': {
            'root': data_path,
            'robot_joints': {
                'left_arm_csv_path': osp.join(data_path, "joints", "left_arm_joint_states_and_end_pose.csv"),
                'right_arm_csv_path': osp.join(data_path, "joints", "right_arm_joint_states_and_end_pose.csv"),
            },
            'sim_start_time': 0.0,
        },
    }
    cfg = OmegaConf.create(active_run_for_debug)

    env = None
    try:
        env = PyBulletFixedPointEnv(cfg=cfg)
        env.run()

    except Exception as e:
        import traceback

        logger.error(f"An unexpected error occurred: {e}")
        logger.error(traceback.format_exc())
    finally:
        if env:
            env.close()