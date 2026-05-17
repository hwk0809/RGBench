import abc
import os
import os.path as osp
import time
import numpy as np
import pybullet as p
import pybullet_data
from omegaconf import DictConfig, OmegaConf

from rgbench.envs.base import BaseEnvWrapper

class PyBulletEnv(BaseEnvWrapper):
    """
    PyBullet-based environment.
    This class loads a URDF robot and an OBJ soft body (cloth) and drives the simulation loop.
    """
    def __init__(self, cfg: DictConfig, **kwargs):
        super().__init__(cfg)
        self.env_cfg = cfg.env
        self.cloth_cfg = cfg.cloth

        # --- 1. Connect to the PyBullet physics server ---
        # Use p.GUI for a visualization window, or p.DIRECT for headless.
        self.physics_client = p.connect(p.GUI)
        print(f"Connected to PyBullet physics server (Client ID: {self.physics_client})")

        # Add data path so we can load builtin models such as the ground plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # --- 2. Configure simulation parameters ---
        self.sim_timestep = self.env_cfg.get("timestep", 1./240.)
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=self.sim_timestep)

        # --- 3. Load scene models (ground, robot, cloth) ---
        self._load_scene()

        # --- 4. Build joint name -> index maps ---
        # Required for controlling specific joints by name
        self._build_joint_map()

        # Track simulation time manually
        self._sim_time = 0.0

        # TODO: like the MuJoCo version, initialize robot data loading/interpolation here
        print("PyBulletEnv initialized successfully.")


    def _load_scene(self):
        """Load all models into the scene."""
        # --- Load ground plane ---
        self.plane_id = p.loadURDF("plane.urdf")
        print(f"Ground plane loaded, ID: {self.plane_id}")

        # --- Load robot ---
        # PyBullet uses URDF files. Convert your robot model to URDF.
        # MuJoCo XML `pos` and `quat` can be set here.
        robot_urdf_path = self.env_cfg.model_path
        if not os.path.exists(robot_urdf_path):
            raise FileNotFoundError(f"Robot URDF not found: {robot_urdf_path}. Please check the path.")

        # The XML shows two arms; we load two URDFs accordingly
        # Left arm
        self.robot_id_left = p.loadURDF(
            fileName=robot_urdf_path,
            basePosition= [0.0,0.25, 0.0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True # robot base is normally fixed
        )
        # Right arm
        self.robot_id_right = p.loadURDF(
            fileName=robot_urdf_path,
            basePosition=[0.0,-0.25, 0.0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True
        )
        print(f"Left arm loaded, ID: {self.robot_id_left}")
        print(f"Right arm loaded, ID: {self.robot_id_right}")


        # --- Load cloth (soft body) ---
        # PyBullet can load a soft body directly from an .obj file
        cloth_obj_path = self.cloth_cfg.model_path
        if not os.path.exists(cloth_obj_path):
            raise FileNotFoundError(f"Cloth OBJ not found: {cloth_obj_path}. Please check the path.")

        # Read initial pose from config
        init_pos = self.cloth_cfg.init_pose[:3]
        # Note: PyBullet quaternions are [x, y, z, w]
        init_quat_wxyz = self.cloth_cfg.init_pose[3:]
        init_quat_xyzw = [init_quat_wxyz[1], init_quat_wxyz[2], init_quat_wxyz[3], init_quat_wxyz[0]]

        self.cloth_id = p.loadSoftBody(
            fileName=cloth_obj_path,
            basePosition=init_pos,
            baseOrientation=init_quat_xyzw,
            scale=1.0,
            mass=self.cloth_cfg.mass,
            useNeoHookean=True,
            useMassSpring=True,
            springElasticStiffness=self.cloth_cfg.stiffness,
            springDampingStiffness=self.cloth_cfg.damping,
            useSelfCollision=True,
            frictionCoeff=self.cloth_cfg.friction,
            useFaceContact=True,
        )
        # Anchors between the soft body and the rigid bodies (robot) allow interaction
        # The lines below show an example of pinning 4 corners in the air
        # p.createSoftBodyAnchor(self.cloth_id, 0, -1, -1)
        # p.createSoftBodyAnchor(self.cloth_id, 20, -1, -1)
        # p.createSoftBodyAnchor(self.cloth_id, 400, -1, -1)
        # p.createSoftBodyAnchor(self.cloth_id, 420, -1, -1)
        print(f"Cloth loaded, ID: {self.cloth_id}")


    def _build_joint_map(self):
        """Build joint-name to joint-index maps for both arms."""
        self.joint_map_left = {}
        self.joint_map_right = {}
        num_joints_left = p.getNumJoints(self.robot_id_left)
        num_joints_right = p.getNumJoints(self.robot_id_right)

        print("\n--- Left arm joint info ---")
        for i in range(num_joints_left):
            info = p.getJointInfo(self.robot_id_left, i)
            joint_name = info[1].decode('utf-8')
            self.joint_map_left[joint_name] = i
            print(f"  index {i}: {joint_name}")

        print("\n--- Right arm joint info ---")
        for i in range(num_joints_right):
            info = p.getJointInfo(self.robot_id_right, i)
            joint_name = info[1].decode('utf-8')
            self.joint_map_right[joint_name] = i
            print(f"  index {i}: {joint_name}")
        print("-" * 20)


    # ===================================================================
    # BaseEnvWrapper abstract method implementations
    # ===================================================================

    def get_master_start_time(self) -> float:
        # Depends on your data loader; returning 0.0 as a placeholder.
        # After CSV integration this can return the first timestamp.
        return 0.0

    def get_current_sim_time(self) -> float:
        # PyBullet does not expose sim.data.time, so we track it manually
        return self._sim_time

    def get_sim_vertices(self) -> np.ndarray:
        """Return cloth soft-body vertex data."""
        try:
            # PyBullet's getMeshData returns vertex data
            # Format: [num_vertices, [pos_x, pos_y, pos_z], [norm_x, norm_y, norm_z]]
            mesh_data = p.getMeshData(self.cloth_id, physicsClientId=self.physics_client)
            num_vertices = mesh_data[0]
            if num_vertices == 0:
                return np.array([])
            # Extract vertex positions
            vertex_positions = np.array(mesh_data[1])
            return vertex_positions
        except Exception as e:
            print(f"Error reading vertex data: {e}")
            return np.array([])

    def step_to_time(self, target_time: float) -> None:
        """
        Step simulation until target_time.
        """
        # TODO: integrate CSV interpolation and robot control inside this loop
        while self._sim_time < target_time:
            # Robot control logic goes here
            # Example: self.apply_robot_control(self._sim_time)
            p.stepSimulation()
            self._sim_time += self.sim_timestep

    def step(self):
        p.stepSimulation()
        self._sim_time += self.sim_timestep

    def apply_joint_positions(self, joint_positions: dict, arm: str = 'left'):
        """
        Apply target joint positions to the specified arm.

        Args:
            joint_positions (dict): Mapping of joint name (str) to target angle (float).
            arm (str): 'left' or 'right'
        """
        robot_id = self.robot_id_left if arm == 'left' else self.robot_id_right
        joint_map = self.joint_map_left if arm == 'left' else self.joint_map_right

        for joint_name, target_pos in joint_positions.items():
            if joint_name in joint_map:
                joint_index = joint_map[joint_name]
                p.setJointMotorControl2(
                    bodyIndex=robot_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    force=500  # tune as needed
                )

    def close(self):
        """Disconnect from the PyBullet server."""
        p.disconnect(self.physics_client)
        print("Disconnected from PyBullet.")

if __name__ == "__main__":
    print("Running PyBulletEnv test in standalone mode...")

    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    DEFAULT_SAMPLE_ROOT = os.path.join(REPO_ROOT, "data", "sample")
    DATA_ROOT = os.environ.get("RGBENCH_DATA_ROOT", DEFAULT_SAMPLE_ROOT)
    CLOTH_MESH_ROOT = os.environ.get("RGBENCH_CLOTH_MESH_ROOT", os.path.join(DEFAULT_SAMPLE_ROOT, "cloth_meshes"))
    data_path = os.environ.get("RGBENCH_PIPER_DATA", os.path.join(DATA_ROOT, "Piper_Data"))
    model_path = os.path.join(REPO_ROOT, "assets", "piper_description", "piper_with_gripper.urdf")

    active_run_for_debug = {
        'cloth': {
            'name': 'green_tshirt',
            'sample_id': 2,
            'model_path': os.path.join(CLOTH_MESH_ROOT, 'LargeT_Flat_Simple_10k.obj'),
            'init_pose': [0.1, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0],  # [x, y, z, w, qx, qy, qz]
            'init_pose_path': '/path/to/non_existent_file.json',
            'mass': 0.22,  # (kg)
            'stiffness': 230,  # stiffness coefficient
            'damping': 0.0,  # damping
            'friction': 0.1  # friction coefficient
        },
        'env': {
            'name': 'pybullet',
            'model_path': model_path,
            'sim_flex_name': 'cloth',
            'enable_gripper_calibrate': False,  # Enable gripper calibration
            'timestep': 0.001
        },
        'data': {
            'root': data_path,
            'robot_joints': {
                'left_arm_csv_path': osp.join(data_path, "right_arm_joint_states.csv"),
                'right_arm_csv_path': osp.join(data_path, "left_arm_joint_states.csv"),
            },
            'sim_start_time': 5.0,
        },
    }

    cfg = OmegaConf.create(active_run_for_debug)

    env = None
    try:
        # --- Initialize environment ---
        env = PyBulletEnv(cfg=cfg)

        # --- Verify model loading ---
        print("\n--- Verifying model loading ---")
        print(f"Left arm ID: {env.robot_id_left}")
        print(f"Right arm ID: {env.robot_id_right}")
        print(f"Cloth ID: {env.cloth_id}")
        initial_vertices = env.get_sim_vertices()
        print(f"Initial cloth vertex count: {len(initial_vertices)}")
        assert len(initial_vertices) > 0, "Cloth vertex loading failed!"
        print("Initial model-loading check passed.")

        # Set simulation duration to 1 second and compute the required number of steps
        simulation_duration = 1.0  # seconds
        num_steps = int(simulation_duration / env.sim_timestep)

        print(f"\n--- Starting simulation for {simulation_duration} seconds ---")
        print("Robot arms stay still for a clean drop test.")

        for i in range(num_steps):
            # Robot motion control is intentionally disabled for a clean physics drop test
            # current_time = env.get_current_sim_time()
            # angle = np.sin(current_time * 2.0) * 1.5
            # control_command = {target_joint_name: angle}
            # env.apply_joint_positions(control_command, arm='left')
            # env.apply_joint_positions(control_command, arm='right')

            p.stepSimulation()
            # We bypass step_to_time, so update internal sim time manually
            env._sim_time += env.sim_timestep

        # Read vertex data after simulation and check the minimum X coordinate
        final_vertices = env.get_sim_vertices()
        if final_vertices.size > 0:
            all_x_coords = final_vertices[:, 0]  # extract X coordinates of all vertices
            min_x_coord = np.min(all_x_coords)   # compute minimum

            print("\n--- Simulation result ---")
            print(f"After {simulation_duration}s, min X coordinate of cloth vertices: {min_x_coord:.6f}")

            # Check whether the result is near the expected 0.1 (small tolerance allowed)
            if np.isclose(min_x_coord, 0.1, atol=0.05):
                 print("Result matches expectation: min X is very close to the initial 0.1.")
            else:
                 print("Result does not match expectation: min X has drifted from the initial 0.1.")
        else:
            print("Could not retrieve final cloth vertex data.")

        print("Simulation finished.")

    except Exception as e:
        import traceback

        print("\n--- Error occurred ---")
        traceback.print_exc()
    finally:
        if env:
            env.close()
