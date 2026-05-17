import abc
import os
import os.path as osp
import sys
import time
import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data
import ast
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from typing import Dict, Tuple, Optional, List, Any, Literal, TypedDict, cast

from rgbench.envs.base import BaseEnvWrapper
from rgbench.csv_data import load_processed_data, JointState

ArmType = Literal["left", "right"]


class ArmConfigData(TypedDict):
    """Stores configuration and data for one arm."""
    name: ArmType
    pybullet_joint_names: List[str]
    data: List[JointState]

class PyBulletEnv(BaseEnvWrapper):
    """
    An environment implemented using PyBullet, supporting dual-arm robot action
    playback from CSV files.
    """
    def __init__(self, cfg: DictConfig, **kwargs):
        super().__init__(cfg)
        self.env_cfg = cfg.env
        self.cloth_cfg = cfg.cloth
        self.cloth_params = cfg.cloth_params
        self.data_cfg = cfg.data


        # --- Initialize PyBullet ---
        use_gui = True
        if "visualization" in cfg and cfg.visualization.get('vis_sim', False):
            use_gui = True
        if use_gui:
            # self.physics_client = p.connect(p.GUI)
            # self.physics_client = p.connect(p.GUI, options="--width=2560 --height=1440")
            self.physics_client = p.connect(p.GUI, options="--width=1920 --height=1080")
            logger.info("PyBullet GUI mode enabled.")
        else:
            self.physics_client = p.connect(p.DIRECT)
            logger.info("PyBullet DIRECT mode enabled (no GUI).")
        logger.info(f"Successfully connected to PyBullet physics server (Client ID: {self.physics_client})")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.sim_timestep = self.env_cfg.get("timestep", 1. / 240.)
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=self.sim_timestep)

        # --- Load Scene Models ---
        self._load_scene()
        self._build_joint_map()
        # self._set_gripper_friction()

        # --- Initialize Playback-related Attributes ---
        self.arms: Dict[ArmType, ArmConfigData] = {}
        self.playback_start_time_abs: Optional[float] = None
        self.playback_end_time_abs: Optional[float] = None
        self.time_trackers = {'left': {'idx': 0}, 'right': {'idx': 0}}
        self._sim_time = 0.0

        # --- Load and Configure Arm Data from CSV ---
        self._init_arms_config(cfg)

        # arm control parameters
        self.arm_pos_gain = self.env_cfg.get("arm_pos_gain", 0.2)  # Kp:
        self.arm_vel_gain = self.env_cfg.get("arm_vel_gain", 1.0)  # Kd:
        self.arm_max_force = self.env_cfg.get("arm_max_force", 500)
        logger.info(
            f"Arm PD controller params set to: Kp={self.arm_pos_gain}, Kd={self.arm_vel_gain}, MaxForce={self.arm_max_force}")

        logger.info("PyBulletEnv initialized successfully and robot action data is loaded.")

    def _load_scene(self):
        self.plane_id = p.loadURDF("plane.urdf")
        robot_urdf_path = self.env_cfg.model_path
        if not os.path.exists(robot_urdf_path):
            raise FileNotFoundError(f"Robot URDF file not found: {robot_urdf_path}.")
        # TODO: basePosition and Orientation should be configurable
        self.robot_id_left = p.loadURDF(fileName=robot_urdf_path, basePosition=[0.0, 0.25, 0.0], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
        self.robot_id_right = p.loadURDF(fileName=robot_urdf_path, basePosition=[0.0, -0.25, 0.0], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
        cloth_obj_path = self.cloth_cfg.model_path
        if not os.path.exists(cloth_obj_path):
            raise FileNotFoundError(f"Cloth OBJ file not found: {cloth_obj_path}.")
        init_pos = self.cloth_cfg.init_pose[:3]
        init_quat_wxyz = self.cloth_cfg.init_pose[3:]
        init_quat_xyzw = [init_quat_wxyz[1], init_quat_wxyz[2], init_quat_wxyz[3], init_quat_wxyz[0]]
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
            cameraDistance=1.5,
            cameraYaw=90,
            cameraPitch=-65,
            cameraTargetPosition=[0.5, 0, 0]
        )
        initial_vertices = self.get_sim_vertices()
        logger.info(f"Cloth loaded, vertices num: {len(initial_vertices)}")


    def _build_joint_map(self):
        self.joint_map_left = {info[1].decode('utf-8'): info[0] for info in
                               [p.getJointInfo(self.robot_id_left, i) for i in
                                range(p.getNumJoints(self.robot_id_left))]}
        self.joint_map_right = {info[1].decode('utf-8'): info[0] for info in
                                [p.getJointInfo(self.robot_id_right, i) for i in
                                 range(p.getNumJoints(self.robot_id_right))]}

    def _set_gripper_friction(self):
        logger.info("Setting high friction for gripper links...")
        # Example: assumes joint7 and joint8 are the gripper link indices
        gripper_link_indices_left = [self.joint_map_left['joint7'], self.joint_map_left['joint8']]
        gripper_link_indices_right = [self.joint_map_right['joint7'], self.joint_map_right['joint8']]
        # high_friction = 1.0
        # spring_friction = 0.8
        # rollingFriction = 0.1
        #
        # for link_index in gripper_link_indices_left:
        #     p.changeDynamics(self.robot_id_left, link_index, lateralFriction=high_friction,spinningFriction=spring_friction,rollingFriction=rollingFriction)
        # for link_index in gripper_link_indices_right:
        #     p.changeDynamics(self.robot_id_right, link_index, lateralFriction=high_friction,spinningFriction=spring_friction,rollingFriction=rollingFriction)

    # ===================================================================
    # Data Loading and Processing
    # ===================================================================
    def _init_arms_config(self, cfg: DictConfig):
        data_cfg = cfg.data
        # Key: declare the joint names and order from the PyBullet URDF here
        arm_setups = {
            'left': {
                'csv_path': data_cfg.robot_joints.get('left_arm_csv_path'),
                # These names must match the joint names in your URDF exactly
                'pybullet_joints': [f'joint{i}' for i in range(1, 7)],
                'pybullet_gripper': ('joint7', 'joint8'),
            },
            'right': {
                'csv_path': data_cfg.robot_joints.get('right_arm_csv_path'),
                # The right arm may use a different naming convention, e.g., an "_arm2" suffix
                # We assume both arms use the same URDF here, so joint names are identical
                'pybullet_joints': [f'joint{i}' for i in range(1, 7)],
                'pybullet_gripper': ('joint7', 'joint8'),
            }
        }

        for arm_name_str, setup in arm_setups.items():
            arm_name = cast(ArmType, arm_name_str)
            if not setup['csv_path']:
                logger.warning(f"Skipping {arm_name} arm as no csv_path was provided.")
                continue

            try:
                filtered_data = load_processed_data(
                    csv_path=setup['csv_path'],
                    start_time_offset=data_cfg.get('sim_start_time', 0.0)
                )
            except FileNotFoundError:
                sys.exit(1)

            if filtered_data:
                # Build an ordered list of joint names
                pybullet_names = setup['pybullet_joints'] + list(setup['pybullet_gripper'])
                self.arms[arm_name] = ArmConfigData(
                    name=arm_name,
                    pybullet_joint_names=pybullet_names,
                    data=filtered_data
                )

        # Determine the global playback timeline from the loaded, filtered data
        all_start_times = [config['data'][0]['time'] for config in self.arms.values() if config['data']]
        all_end_times = [config['data'][-1]['time'] for config in self.arms.values() if config['data']]
        if all_start_times:
            self.playback_start_time_abs = min(all_start_times)
            self.playback_end_time_abs = max(all_end_times)
            duration = self.playback_end_time_abs - self.playback_start_time_abs
            print("-" * 30)
            print("Global Playback Timeline Determined:")
            print(f"  - Absolute Start Time: {self.playback_start_time_abs:.4f}")
            print(f"  - Absolute End Time:   {self.playback_end_time_abs:.4f}")
            print(f"  - Total Duration:      {duration:.2f} seconds")
            print("-" * 30)
        else:
            logger.warning("Could not load valid data for any arm. Playback is not possible.")

    def _get_interpolated_positions(self, target_time: float, arm_data: List[JointState], tracker: Dict[str, int]) -> \
            Optional[List[float]]:
        """Calculates a list of joint positions for a target time using linear interpolation."""
        if not arm_data: return None

        # Advance the tracker to find the current interpolation interval
        while tracker['idx'] < len(arm_data) - 2 and arm_data[tracker['idx'] + 1]['time'] < target_time:
            tracker['idx'] += 1

        # Handle boundary cases
        if target_time <= arm_data[0]['time']: return arm_data[0]['positions']
        if target_time >= arm_data[-1]['time']: return arm_data[-1]['positions']

        # Linear interpolation
        p0, p1 = arm_data[tracker['idx']], arm_data[tracker['idx'] + 1]
        t0, pos0_list = p0['time'], p0['positions']
        t1, pos1_list = p1['time'], p1['positions']

        interval = t1 - t0
        if interval <= 0: return pos0_list

        alpha = (target_time - t0) / interval

        # Use NumPy for clean and efficient list interpolation
        pos0_arr = np.array(pos0_list)
        pos1_arr = np.array(pos1_list)
        interpolated_pos_arr = pos0_arr + alpha * (pos1_arr - pos0_arr)

        return interpolated_pos_arr.tolist()

    def _apply_arm_state(self, arm_config: ArmConfigData, positions: List[float]):
        """
        Applies the given joint states to the corresponding arm in the PyBullet simulation.
        Args:
            arm_config (ArmConfigData): The configuration for the arm to be controlled.
            raw_positions (RawJointPositions): The dictionary of target joint positions.
        """
        arm_name = arm_config['name']
        robot_id = self.robot_id_left if arm_name == 'left' else self.robot_id_right
        joint_map = self.joint_map_left if arm_name == 'left' else self.joint_map_right

        joint_names = arm_config['pybullet_joint_names']

        if len(positions) != 7 or len(joint_names) != 8:
            logger.warning(f"Malformed data for arm '{arm_name}': "
                           f"Expected 7 positions and 8 joint names, got {len(positions)} and {len(joint_names)}.")
            return

        arm_joint_names = joint_names[:6]
        arm_positions = positions[:6]
        for name, pos in zip(arm_joint_names, arm_positions):
            try:
                joint_id = joint_map[name]
                p.setJointMotorControl2(robot_id,
                                        joint_id,
                                        p.POSITION_CONTROL,
                                        positionGain=self.arm_pos_gain,  # <--- increase Kp
                                        velocityGain=self.arm_vel_gain,  # <--- increase Kd
                                        force=self.arm_max_force,  # <--- increase max force
                                        targetPosition=pos)
            except KeyError:
                pass

        gripper_value = positions[-1]
        gripper_max_force = 1000.0
        gripper_jnt1_name, gripper_jnt2_name = joint_names[6], joint_names[7]
        try:
            gripper_jnt1_id = joint_map[gripper_jnt1_name]

            p.setJointMotorControl2(robot_id, gripper_jnt1_id, p.POSITION_CONTROL, targetPosition=gripper_value,force=gripper_max_force)

            gripper_jnt2_id = joint_map[gripper_jnt2_name]
            p.setJointMotorControl2(robot_id, gripper_jnt2_id, p.POSITION_CONTROL, targetPosition=-gripper_value,force=gripper_max_force)
        except KeyError:
            pass

    # ===================================================================
    # BaseEnvWrapper Interface Implementation
    # ===================================================================
    def get_master_start_time(self) -> float:
        return self.playback_start_time_abs or 0.0

    def get_current_sim_time(self) -> float:
        return self._sim_time

    def get_sim_vertices(self) -> np.ndarray:
        try:
            mesh_data = p.getMeshData(self.cloth_id, physicsClientId=self.physics_client)
            return np.array(mesh_data[1]) if mesh_data and mesh_data[0] > 0 else np.array([])
        except Exception:
            return np.array([])

    def step_to_time(self, target_time: float):
        while self._sim_time < target_time:
            self.step()

    def step(self):
        current_abs_time = self.get_master_start_time() + self._sim_time
        for arm_name, arm_config in self.arms.items():
            if arm_config['data']:
                interpolated_positions = self._get_interpolated_positions(current_abs_time, arm_config['data'],
                                                                          self.time_trackers[arm_name])
                if interpolated_positions:
                    self._apply_arm_state(arm_config, interpolated_positions)
        p.stepSimulation()
        self._sim_time += self.sim_timestep

    def close(self):
        p.disconnect(self.physics_client)
        logger.info("Disconnected from PyBullet.")

    # ===================================================================
    # Render functions
    # ===================================================================

    def run(self):
        """Main loop for interactive CSV data playback."""
        if self.playback_start_time_abs is None:
            logger.error("Playback start time not set. Cannot run simulation.")
            # Run a static simulation for observation
            while True:
                try:
                    p.stepSimulation()
                    time.sleep(self.sim_timestep)
                except p.error:
                    break
            return

        end_sim_time = self.playback_end_time_abs - self.playback_start_time_abs
        while self._sim_time < end_sim_time:
            try:
                self.step_to_time(self._sim_time + self.sim_timestep)
                # Optional: slow down playback for observation
                # time.sleep(self.sim_timestep)
            except p.error:
                logger.info("PyBullet GUI window closed, simulation ended.")
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
                    if camera_name == "gripper_view":
                        # near gripper view
                        view_matrix = p.computeViewMatrix(
                            cameraEyePosition=[0.47, -0.65, 1.05],
                            cameraTargetPosition=[0.47, 0.0515, 0.0],
                            cameraUpVector=[0, 0, 1],
                            physicsClientId=self.physics_client
                        )
                    else:
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
                    lightColor=[1, 1, 1],      # light color
                    lightDistance=2,           # light distance
                    shadow=1,                  # enable shadows
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

    print("Running PyBullet Cloth Simulation Environment...")
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir))
    DEFAULT_SAMPLE_ROOT = os.path.join(REPO_ROOT, "data", "sample")
    DATA_ROOT = os.environ.get("RGBENCH_DATA_ROOT", DEFAULT_SAMPLE_ROOT)
    CLOTH_MESH_ROOT = os.environ.get("RGBENCH_CLOTH_MESH_ROOT", os.path.join(DEFAULT_SAMPLE_ROOT, "cloth_meshes"))
    data_path = os.environ.get("RGBENCH_PIPER_DATA", os.path.join(DATA_ROOT, "Piper_Data"))
    model_path = os.path.join(REPO_ROOT, "assets", "piper_description", "piper_with_gripper.urdf")

    active_run_for_debug = {
        'cloth': {
            'name': 'green_tshirt',
            'sample_id': 2,
            'model_path': os.path.join(CLOTH_MESH_ROOT, 'Skirt_Flat_Simple_20k.obj'),
            # 'init_pose': [0.1, 0.0, 0.1, 0.5, 0.5, 0.5, 0.5],  # [x, y, z, w, qx, qy, qz]
            'init_pose': [0.1, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0],  # [x, y, z, w, qx, qy, qz]
            'init_pose_path': '/path/to/non_existent_file.json',
            'mass': 0.255,  # (kg)
            'stiffness': 230,  # stiffness coefficient
            'damping': 0.0,  # damping
            'friction': 1.0  # friction coefficient
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
            'name': 'pybullet',
            'model_path': model_path,
            'sim_flex_name': 'cloth',
            'timestep': 0.002
        },
        'data': {
            'root': data_path,
            'robot_joints': {
                'left_arm_csv_path': osp.join(data_path, "joints", "left_arm_joint_states_and_end_pose.csv"),
                'right_arm_csv_path': osp.join(data_path, "joints", "right_arm_joint_states_and_end_pose.csv"),
            },
            'sim_start_time': 1.0,
        },
    }

    cfg = OmegaConf.create(active_run_for_debug)

    env = None
    try:
        env = PyBulletEnv(cfg=cfg)
        # env.run()  # Start interactive playback
        env.record_video(output_path=os.path.join(REPO_ROOT, "outputs", "video", "test_output.mp4"), width=1920, height=1080, speed=1.0, fps=60)

    except Exception as e:
        import traceback

        logger.error("An unexpected error occurred:")
        logger.error(traceback.format_exc())
    finally:
        if env:
            env.close()