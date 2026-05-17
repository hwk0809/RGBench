from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# load external package
import cv2
import os
import sys
import time
import numpy as np
import open3d as o3d
from termcolor import cprint
import threading

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# load isaac-relevant package
import omni.replicator.core as rep
import isaacsim.core.utils.prims as prims_utils
from pxr import UsdGeom,UsdPhysics,PhysxSchema, Gf
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.api import SimulationContext
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid
from isaacsim.core.utils.prims import is_prim_path_valid, set_prim_visibility
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
from isaacsim.core.prims import SingleXFormPrim, SingleClothPrim, SingleRigidPrim, SingleGeometryPrim, SingleParticleSystem, SingleDeformablePrim
from isaacsim.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
from omni.physx.scripts import deformableUtils,particleUtils,physicsUtils

# load custom package
sys.path.append(os.getcwd())
from rgbench.envs.isaacsim.Env_Config.BaseEnv import BaseEnv
from rgbench.envs.isaacsim.Env_Config.Garment.Particle_Garment import Particle_Garment
from rgbench.envs.isaacsim.Env_Config.Camera.Recording_Camera import Recording_Camera
from rgbench.envs.isaacsim.Env_Config.Room.Real_Ground import Real_Ground
from isaacsim.core.utils.rotations import euler_angles_to_quat
from rgbench.envs.isaacsim.Env_Config.Utils_Project.Code_Tools import get_unique_filename, normalize_columns
from rgbench.envs.base import BaseEnvWrapper
print(sys.prefix)

from omegaconf import DictConfig, OmegaConf
import os.path as osp
import ast
import time
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List, Any, Literal, TypedDict, cast
import numpy as np
import json
from loguru import logger
from omni.physx.scripts import particleUtils
import omni.usd
from rgbench.envs.isaacsim.Env_Config.Utils_Project.Attachment_Block import attach_fixedblock,AttachmentBlock
from pxr import PhysxSchema
from rgbench.csv_data import load_processed_data, JointState
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
import pickle
from rgbench.envs.base import BaseEnvWrapper
from typing import List, Dict, Tuple, Literal, TypedDict,Optional,Union
# --- Constants ---
MUJOCO_GRIPPER_TARGET_MAX_OPENING: float = 0.035 # TODO: This should be revise to your model
MAX_RAW_GRIPPER_DATA = 0.0688 # TODO: This should be revise to your data
MIN_RAW_GRIPPER_DATA = 0.0000

# --- Type Definitions ---
ArmType = Literal["left", "right"]
RawJointPositions = Dict[str, float]

class ArmDataEntry(TypedDict):
    """Data for a single arm at a single timestamp (with raw joint names from CSV)."""
    time: float
    positions: RawJointPositions

class GripperCalibration(TypedDict):
    """Gripper calibration parameters."""
    gripper_factor: float  # Scaling factor from CSV to MuJoCo
    min_raw_gripper: float    # Minimum gripper value from CSV data
    max_raw_gripper: float    # Maximum gripper value from CSV data

class ArmConfigData(TypedDict):
    """Complete configuration and data for a single arm."""
    name: ArmType
    suffix: str # Suffix for joints in MuJoCo XML (e.g., '_arm2')
    mujoco_gripper_joint_names: Tuple[str, str] # The two gripper joint names in MuJoCo
    joint_map: Dict[str, str] # Map from CSV joint names to MuJoCo joint names
    data: List[ArmDataEntry] # Raw data loaded from CSV
    gripper_calibration: GripperCalibration

class JointNameMap(TypedDict):
    """mapping csv joint-name -> mujoco joint name"""
    gripper: str
    joint1: str
    joint2: str
    joint3: str
    joint4: str
    joint5: str
    joint6: str

class JointState(TypedDict):
    """
    Represents the state of all arm joints at a single point in time.
    'positions' is an ordered list: [j1, j2, j3, j4, j5, j6, calibrated_gripper_val]
    """
    time: float          # Absolute timestamp
    positions: List[float] # The values for each joint, in a pre-defined order.

class PoseState(TypedDict):
    """
    Represents the 3D pose of an object, like an end-effector.
    """
    time: float
    position: List[float]      # [x, y, z]
    orientation: List[float]   # [w, x, y, z] quaternion

class GrabberState(TypedDict, total=False):
    is_active: bool
    is_grabbed: bool        # Flag to indicate if the grab time has passed
    grab_time: float
    pose_data: List[PoseState]
    pose_tracker: Dict[str, int]
    body_id: int
    vertex_id: int          #  Stores the anchored cloth vertex ID
    initial_grab_vertex_pose: Optional[List[float]]



def get_flex_vertices(prim_path: str, downsample: int = 1):
    
    """
    Display the cloth point cloud after the fold action completes
    :param prim_path: path to the cloth USD Prim (e.g., "/World/Garment/garment")
    :param downsample: point cloud downsample ratio
    """
    # Get the current cloth Prim
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"Error: Prim not found at path {prim_path}")
        return
    
    # Get point cloud data
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    if not points:
        print("Error: No points data in the mesh")
        return
    # Assume point is a Vt.Vec3fArray object
    point_array = np.array([
        [vec[0], vec[1], vec[2]]  # extract x, y, z components
        for vec in points
    ], dtype=np.float32)  # use float32 dtype

    # Get the world transform matrix
    xform = UsdGeom.Xformable(prim)
    world_transform = xform.ComputeLocalToWorldTransform(0)  # local-to-world transform matrix

    # Convert to the world frame
    world_points = []
    for p in points:
        # Convert the point from local to world coordinates
        world_point = world_transform.Transform(Gf.Vec3f(p))  # apply transform matrix
        world_points.append([world_point[0], world_point[1], world_point[2]])

    # Downsample
    world_points = np.array(world_points, dtype=np.float32)
    world_points = world_points[::downsample]

    return world_points

def get_flex_vertices1(prim_path: str, downsample: int = 1):
    
    """
    Display the cloth point cloud after the fold action completes
    :param prim_path: path to the cloth USD Prim (e.g., "/World/Garment/garment")
    :param downsample: point cloud downsample ratio
    """
    # Get the current cloth Prim
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"Error: Prim not found at path {prim_path}")
        return
    
    # Get point cloud data
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    if not points:
        print("Error: No points data in the mesh")
        return
    

    # Get the world transform matrix
    xform = UsdGeom.Xformable(prim)
    world_transform = xform.ComputeLocalToWorldTransform(0)  # local-to-world transform matrix

    # Convert to the world frame
    world_points = []
    for p in points:
        # Convert the point from local to world coordinates
        world_point = world_transform.Transform(Gf.Vec3f(p))  # apply transform matrix
        world_points.append([world_point[0], world_point[1], world_point[2]])
    
    # Downsample
    world_points = np.array(world_points, dtype=np.float32)
    world_points = world_points[::downsample]
    
    # Find the point with the minimum x value
    if len(world_points) > 0:
        min_x_idx = np.argmin(world_points[:, 0])
        min_x_point = world_points[min_x_idx]
        # Print world-frame coordinates
        print(f"world-frame min X point: index={min_x_idx}, coord=({min_x_point[0]:.6f}, {min_x_point[1]:.6f}, {min_x_point[2]:.6f})")
    else:
        print("warning: point cloud is empty")

    points=world_points
    
    # Build Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Compute colors (based on Z height)
    z_min, z_max = np.min(points[:,2]), np.max(points[:,2])
    colors = np.zeros((len(points), 3))
    colors[:, 0] = (points[:,2] - z_min) / (z_max - z_min + 1e-6)  # R
    colors[:, 1] = 1 - colors[:,0]  # G
    colors[:, 2] = 0.5  # B
    #colors[:, 0] = 1.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd
    
class PiperCloth_Env(BaseEnv):
    def __init__(
        self, 
        usd_path:str=None, 
        cfg: DictConfig=None,
        **kwargs
    ):
        self.cfg=cfg
        self.data_cfg = cfg.data
        self.env_cfg = cfg.env
        # load BaseEnv
        physics_timestep = cfg.env.get('physics_timestep', 1/500)
        rendering_timestep = cfg.env.get('rendering_timestep', 1/60)
        friction = cfg.env.get('friction', 25.0)
        damping = cfg.env.get('damping', 0)
        stretch_stiffness = cfg.env.get('stretch_stiffness', 2.3e5)
        bend_stiffness = cfg.env.get('bend_stiffness', 1e3)
        shear_stiffness = stretch_stiffness

        springElasticStiffness=float(self.cfg.cloth_params.stretch)
        springDampingStiffness=float(self.cfg.cloth_params.damping)
        springBendingStiffness=float(self.cfg.cloth_params.bending)
        frictionCoeff=float(self.cfg.cloth_params.friction)
        # Use the fully-resolved absolute USDA path from active_run.cloth,
        # not the bare subpath under cloth_params (which would be relative).
        cloth_usd_path = self.cfg.cloth.usda_model_path
        #print(springElasticStiffness)
        # friction=
        # damping= 
        # stretch_stiffness:float=1e12, #1e6  
        # bend_stiffness:float=100.0, 
        # shear_stiffness:float=100.0, 

        super().__init__(physics_timestep, rendering_timestep)
        
        # ------------------------------------ #
        # ---        Add Env Assets        --- #
        # ------------------------------------ #

        self.ground = Real_Ground(
            self.scene, 
            # visual_material_usd = "assets/Assets/Material/Floor/Fabric001.usd"
            # you can use materials in 'Assets/Material/Floor' to change the texture of ground.
        )
        _isaac_assets = os.path.join(PROJECT_ROOT, "assets", "isaacsim_assets")
        visual_material_usd_white   = os.path.join(_isaac_assets, "Material", "Garment", "linen_White.usd")
        visual_material_usd_Beige   = os.path.join(_isaac_assets, "Material", "Garment", "linen_Beige.usd")
        visual_material_usd_Blue    = os.path.join(_isaac_assets, "Material", "Garment", "linen_Blue.usd")
        visual_material_usd_Pumpkin = os.path.join(_isaac_assets, "Material", "Garment", "linen_Pumpkin.usd")
        
        self.garment = Particle_Garment(
            self.world, 
            pos=np.array([0.1, 0, 0.01]),
            ori=np.array([0.0, 0.0, 0.0]),
            scale =np.array([1.0, 1.0, 1.0]),
            usd_path = cloth_usd_path,
            visual_material_usd = visual_material_usd_white,
            # usd_path="Assets/Garment/Tops/test/10k.usda" if usd_path is None else usd_path,
            #particle_mass=255/10000,          # important parameter default 1e-2
            stretch_stiffness = springElasticStiffness,
            bend_stiffness = springBendingStiffness,
            # shear_stiffness = shear_stiffness,
            # friction = friction,
            # damping = damping,
            # particle_mass=1e-2,               # important parameter default 1e-2
            # stretch_stiffness=1e6, #1e6 default
            # friction=100.0,         #10 default
            # adhesion=1.0,
            # particle_adhesion_scale = 1.0,     # important parameter
            # particle_friction_scale = 1.0,
            # contact_offset=0.015,         
            # rest_offset=0.01,            
            # particle_contact_offset=0.015,
            # bend_stiffness=1000.0, 
        )
        
        vis_cube = VisualCuboid(
            prim_path = "/World/vis_cube_left",
            color=np.array([255, 0, 0]),
            name = "vis_cube_left", 
            position = [0.0, 0.0, 2.0],
            size = 0.1,
            visible = True,
        )
        self.phys_cube = DynamicCuboid(
            prim_path="/World/phys_cube_left",
            position=[-0.2, 0.0, 1.0],
            size=0.1,
            color=np.array([1.0, 0.0, 0.0]),  # RGB red
            mass=0.01  # mass must be set
        )
        self.world.scene.add(self.phys_cube)
        self.world.scene.add(
            vis_cube
        )

        # self.env_camera = Recording_Camera(
        #     camera_position=np.array([2, 0, 1.5]),
        #     #camera_orientation=np.array([0, 60, -90.0]),
        #     prim_path="/World/env_camera",
        # )
        
        # set_camera_view(
        #     eye=[4, 0, 3],
        #     target=[0.1, 0.0, 0.0],
        #     camera_prim_path="/World/env_camera",
        # )

        self.attachment_blocks = AttachmentBlock(
            self.world,
            prim_path="/World/AttachmentBlocks",
            garment_path=[self.garment.garment_prim_path]  # garment_path must be a list
        )
         # --- STAGE 0: Determine Action Type and Parameters ---
        self.left_base_offset = [0, 0.25, 0]
        self.right_base_offset = [0, -0.25, 0]
        self.grab_points_cfg = self.env_cfg.get('grab_points', {})
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

        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])
        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])
        cprint("World Ready!", "green", "on_green")

    
    
    def setup_attachment_points_from_arms(self):
        """Automatically set attachment points based on arm position."""
        # 1. Get cloth point cloud
        pcd = get_flex_vertices1(prim_path="/World/Garment/garment/mesh", downsample=1)
        garment_points = np.asarray(pcd.points)
        
        # 2. Find the cloth points closest to each arm
        # auto find
        # left_idx = np.argmin(np.linalg.norm(garment_points - left_pos, axis=1))
        # right_idx = np.argmin(np.linalg.norm(garment_points - right_pos, axis=1))

        # manual set
        left_idx, right_idx = self._determine_pin_indices()

        point1 = garment_points[left_idx]
        point2 = garment_points[right_idx]

        
        print(f"Auto selection: near left arm {point1} | near right arm {point2}")
        
        # 3. Create attachment blocks
        self._create_attachment_blocks(point1, point2)
        
        # 4. Step physics to settle the scene
        for _ in range(5):
            self.world.step(render=True)
        
        return left_idx, right_idx
    
    def _create_attachment_blocks(self, point1, point2):
        """Create and configure an attachment block."""
        # Create the block
        self.attachment_blocks.create_block("block_left", point1, True)
        self.attachment_blocks.create_block("block_right", point2, True)
        
        # Disable gravity
        self.attachment_blocks.enable_disable_gravity(0, False)
        self.attachment_blocks.enable_disable_gravity(1, False)
        
        # Set initial position
        self.attachment_blocks.set_block_position(0, point1)
        self.attachment_blocks.set_block_position(1, point2)

        # Attach to the cloth
        self.attachment_blocks.attach([0, 1])

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
        
    def _calculate_auto_indices(self) -> List[int]:
        logger.info("Calculating indices for 'auto' mode in the current simulation...")

        # 1.get stable vertices positions
        initial_verts_pos = get_flex_vertices("/World/Garment/garment/mesh")
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
        #self.reset()

        

    def select_points_interactively(self, points):
        """
        Interactively select two points on the point cloud and return their indices.

        Args:
            point_cloud: cloth point cloud (o3d.geometry.PointCloud)

        Returns:
            tuple: (index1, index2) indices of the two selected points
        """
        # Create visualizer

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        vis = o3d.visualization.VisualizerWithVertexSelection()
        vis.create_window(window_name='Cloth point cloud picker', width=1280, height=720)
        
        # Add the point cloud
        vis.add_geometry(point_cloud)
        
        # Configure point cloud rendering
        vis.get_render_option().point_size = 3.0
        vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
        
        # Run visualizer (blocking)
        vis.run()
        
        # Retrieve picked points
        picked_points = vis.get_picked_points()
        
        # Close window
        vis.destroy_window()
        
        # Extract the first two picked indices
        if len(picked_points) >= 2:
            return picked_points[0].index, picked_points[1].index
        else:
            print("Warning: please make sure two points were selected")
            return 4587, 841


import abc
import numpy as np
from omegaconf import DictConfig


class IsaacsimEnv(BaseEnvWrapper):
    def __init__(self, cfg: DictConfig, **kwargs):
        # First, call the base class __init__
        super().__init__(cfg)
        self.data_cfg = cfg.data
        self.env_cfg = cfg.env
        # self.sim_timestep = self.env_cfg.get("timestep", 0.005)
        # --- Initialize attributes from DualArmController ---
        self.playback_start_time_abs: Optional[float] = None
        self.playback_end_time_abs: Optional[float] = None
        self.arms: Dict[ArmType, ArmConfigData] = {}
        self.left_base_offset = [0, 0.25, 0]
        self.right_base_offset = [0, -0.25, 0]
        # --- Initialize interpolation trackers ---
        # This tracker is crucial for the step_to_time method
        self.time_trackers = {'left': {'idx': 0}, 'right': {'idx': 0}}
        self.grabbers: Dict[ArmType, GrabberState] = {}
        # --- Execute the original configuration and data loading process ---
        # initial setting
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
        

        usd_path = cfg.cloth.get('model_path', None)
        print(usd_path)
        self.robot=PiperCloth_Env(usd_path, cfg)


        # --- STAGE 3: Set up the anchor point  ---
        pin_indices = self.robot.setup_attachment_points_from_arms()
        # --- STAGE 4: Initialize the trajectory data  ---
        self.left_grabber_id = 1 
        self.right_grabber_id = 2
        self._init_grabbers_data([self.left_grabber_id,self.right_grabber_id], pin_indices)
        # --- Set Global Playback Timeline ---
        self.playback_start_time_abs: Optional[float] = None
        self.playback_end_time_abs: Optional[float] = None
        # Extract all timestamps
        # --- STAGE 4:Set the global playback timeline ---
        all_start_times = [g['pose_data'][0]['time'] for g in self.grabbers.values() if g.get('is_active')]
        all_end_times = [g['pose_data'][-1]['time'] for g in self.grabbers.values() if g.get('is_active')]

        if not all_start_times:
            raise ValueError("Simulation cannot start: No active grabbers with data were found.")

        self.playback_start_time_abs = min(all_start_times)
        self.playback_end_time_abs = max(all_end_times)
        duration = self.playback_end_time_abs - self.playback_start_time_abs

        duration = self.playback_end_time_abs - self.playback_start_time_abs
        logger.info("-" * 30)
        logger.info("Global Playback Timeline Determined:")
        logger.info(f"  - Absolute Start Time: {self.playback_start_time_abs:.4f}")
        logger.info(f"  - Absolute End Time:   {self.playback_end_time_abs:.4f}")
        logger.info(f"  - Total Duration:      {duration:.2f} seconds")
        logger.info("-" * 30)
        logger.success("MujocoDualFixedPointEnv initialization successful.")
        
        self.start_grasp_time = 1.0

        target_time_abs = self.playback_start_time_abs + self.start_grasp_time


        self.robot.reset()
    
        # step world to make it ready
        for i in range(100):
            self.robot.step()
        

        self.time_trackers = {
            'left': {'idx': 0},
            'right': {'idx': 0}
        }

        print("IsaacEnv initialization successful.")


    # ===================================================================
    # Implementation of BaseEnvWrapper's abstract methods
    # ===================================================================
    def get_master_start_time(self) -> float:
        """Returns the absolute start time from the loaded data."""
        return self.playback_start_time_abs

    def get_current_sim_time(self) -> float:
        """Returns the current MuJoCo simulation time."""
        return self.robot.world.current_time

    def get_sim_vertices(self) -> np.ndarray:
        """Gets the vertices of the soft body."""
        return get_flex_vertices("/World/Garment/garment/mesh")
    
    def step(self) -> None:

        if self.action_type == 'grasp' or self.action_type == 'fold':
            self._step_grasp()

        elif self.action_type == 'fling':
            self._step_fling()

    def _step_grasp(self):
        self._sim_time = self.robot.world.current_time
        current_absolute_time = self.get_master_start_time() + self._sim_time
        # print("Now time", self._sim_time)
        for arm_name, grabber in self.grabbers.items():
            if not grabber.get('is_active'):
                continue
            
            # Check if the grab time has been reached
            
            # print("Grab time", grabber['grab_time'])
            if not grabber['is_grabbed'] and self._sim_time >= grabber['grab_time']:
                
                grabber['is_grabbed'] = True
                logger.success(f"Grab triggered: '{arm_name}' starts moving at sim time {self._sim_time:.4f}s.")

            # Only move the grabber after the grab is triggered
            if not grabber['is_grabbed']:
                self.robot.world.step(render=True)
                continue

            target_pose = self._get_interpolated_pose(current_absolute_time, grabber['pose_data'],
                                                      grabber['pose_tracker'])
            if not target_pose:
                continue

            target_pos = np.array(target_pose['position'])
            target_orn_wxyz = np.array(target_pose['orientation'])

            if grabber['body_id'] == 1:
                self.robot.attachment_blocks.set_block_position(0, target_pos)
            else:
                self.robot.attachment_blocks.set_block_position(1, target_pos)

            # print("Target pos:", target_pos)
            self.robot.world.step(render=True)
        

    def _step_fling(self):
        sim_time = self.robot.world.current_time
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
            if grabber['body_id'] == 1:
                self.robot.attachment_blocks.set_block_position(0, current_pos)
            else:
                self.robot.attachment_blocks.set_block_position(1, current_pos)

            self.robot.world.step(render=True)


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
            
    def step_to_time(self, target_time: float) -> None:
        """
        Step the simulation to the specified time.

        Args:
            target_time: target simulation time (relative to start)
        """
        while self.robot.world.current_time < target_time:
            self.step()


    def _init_arms_config(self, left_csv, right_csv, data_play_start_time, enable_gripper_calibrate):
        """Initializes configurations, data, mappings, and calibration for both arms."""
        arm_setup: Dict[ArmType, Dict[str, Any]] = {
            'left': {'suffix': '', 'csv_path': left_csv, 'gripper_joints': ('joint7', 'joint8'),
                    'data': self._load_arm_data(left_csv, data_play_start_time)},
            'right': {'suffix': '_arm2', 'csv_path': right_csv, 'gripper_joints': ('joint7_arm2', 'joint8_arm2'),
                    'data': self._load_arm_data(right_csv, data_play_start_time)}
        }
        for arm_name, setup in arm_setup.items():
            arm_type = cast(ArmType, arm_name)
            joint_map_dict: Dict[str, str] = {f'joint{i}': f'joint{i}{setup["suffix"]}' for i in range(1, 9)}
            joint_map_dict['gripper'] = setup['gripper_joints'][0]
            print("Init each arm config")
            gripper_calibrate = self._calibrate_gripper_data(setup['csv_path'], enable_gripper_calibrate)
            self.gripper_calibrate=gripper_calibrate
            print(
                f"{arm_type.capitalize()} gripper calibrated: min={gripper_calibrate['min_raw_gripper']:.3f}, max={gripper_calibrate['max_raw_gripper']:.3f}")
            self.arms[arm_type] = {'name': arm_type, 'suffix': setup['suffix'],
                                'mujoco_gripper_joint_names': setup['gripper_joints'], 'joint_map': joint_map_dict,
                                'data': setup['data'], 'gripper_calibration': gripper_calibrate}

        all_start_times = [config['data'][0]['time'] for config in self.arms.values() if config['data']]
        all_end_times = [config['data'][-1]['time'] for config in self.arms.values() if config['data']]
        if all_start_times:
            self.playback_start_time_abs = min(all_start_times)
            self.playback_end_time_abs = max(all_end_times)
            duration = self.playback_end_time_abs - self.playback_start_time_abs
            print("-" * 30);
            print(f"Global Playback Timeline Determined:");
            print(f"  - Absolute Start Time: {self.playback_start_time_abs:.4f}");
            print(f"  - Absolute End Time:   {self.playback_end_time_abs:.4f}");
            print(f"  - Total Duration:      {duration:.2f} seconds");
            print("-" * 30)
        else:
            print("⚠️ WARNING: Could not load valid data for any arm. Playback is not possible.")


    def _load_arm_data(self, csv_path: str, start_time_offset: float = 0.0) -> List[ArmDataEntry]:
        """Loads single-arm CSV data."""
        df = pd.read_csv(csv_path)
        df['position'] = df['position'].apply(ast.literal_eval)
        df['name'] = df['name'].apply(ast.literal_eval)
        df['time'] = df['header.stamp.secs'] + df['header.stamp.nsecs'] * 1e-9
        if df.empty or 'time' not in df.columns: raise ValueError(
            f"CSV file {csv_path} is empty or missing 'time' column.")
        absolute_start_time_for_playback = df['time'].iloc[0] + start_time_offset
        df_filtered = df[df['time'] >= absolute_start_time_for_playback].copy()
        if df_filtered.empty: print(f"Warning: No data remains after time filtering for {csv_path}."); return []
        print(
            f"Loaded from {csv_path}. Effective data duration: {df_filtered['time'].iloc[-1] - df_filtered['time'].iloc[0]:.2f} s.")
        return [ArmDataEntry(time=row['time'], positions=dict(zip(row['name'], row['position']))) for _, row in
                df_filtered.iterrows()]     

    def _calibrate_gripper_data(self, csv_path: str, enable_calibrate: bool = True) -> GripperCalibration:
        """
        Calibrates gripper data.
        Because the gripper value is not always stable, we use these function to calibrate the gripper data.
        TODO: if you gripper is stable, you can set enable_calibrate to False.
        """
        max_grip, min_grip = MAX_RAW_GRIPPER_DATA, MIN_RAW_GRIPPER_DATA
        if enable_calibrate:
            try:
                df = pd.read_csv(csv_path);
                df['position'] = df['position'].apply(ast.literal_eval)
                raw_gripper_positions: List[float] = [row['position'][-1] for _, row in df.iterrows()]
                if raw_gripper_positions: max_grip, min_grip = max(raw_gripper_positions), min(raw_gripper_positions)
            except Exception:
                print(f"Warning: Gripper calibration failed for {csv_path}. Using defaults.")
        range_raw = max_grip - min_grip
        gripper_factor = 0.035 / range_raw if range_raw != 0 else 0
        print("Gripper Calibration CSV Max Gripper",max_grip)
        print("Gripper Calibration CSV Min Gripper",min_grip)
        print("Gripper Calibration Gripper Factor",gripper_factor)
        return GripperCalibration(min_raw_gripper=min_grip, max_raw_gripper=max_grip, gripper_factor=gripper_factor)

    def _get_interpolated_positions(self, target_time: float, arm_data: List[ArmDataEntry], tracker: Dict[str, int]) -> \
    Optional[RawJointPositions]:
        """Calculates joint positions for a target time using linear interpolation."""
        if not arm_data: return None
        while tracker['idx'] < len(arm_data) - 1 and arm_data[tracker['idx'] + 1]['time'] < target_time: tracker[
            'idx'] += 1
        if target_time <= arm_data[0]['time']: return arm_data[0]['positions']
        if target_time >= arm_data[-1]['time']: return arm_data[-1]['positions']
        p0, p1 = arm_data[tracker['idx']], arm_data[tracker['idx'] + 1]
        t0, pos0 = p0['time'], p0['positions'];
        t1, pos1 = p1['time'], p1['positions']
        interval = t1 - t0
        if interval <= 0: return pos0
        alpha = (target_time - t0) / interval
        interpolated_pos: RawJointPositions = {}
        for name in pos0.keys():
            if name in pos1: interpolated_pos[name] = pos0[name] + alpha * (pos1[name] - pos0[name])
        return interpolated_pos


    def run(self):
        """Main control loop for the simulation with attachment control."""
        if self.playback_start_time_abs is None:
            print("Error: Playback start time not set. Aborting simulation.")
            return

        # Initialize time trackers and physics-step parameters
        time_trackers = {name: {'idx': 0} for name, config in self.arms.items() if config['data']}
        sim_dt = 0.002  # physics timestep (5ms)
        max_runtime = 20.0  # maximum simulation time (10 seconds)
        current_sim_time = 10.0  # current simulation time (relative)
        
        # Physics-step loop
        while simulation_app.is_running() and current_sim_time <= max_runtime:
            step_start_time = time.time()  # record step start time
            
            # --- Core update steps ---
            # 1. Compute current absolute timestamp
            abs_time = self.playback_start_time_abs + current_sim_time
            
            # 2. Update attachment positions (core change)
            self.step_to_time(current_sim_time)  # update attachments via step_to_time
            
            current_sim_time += sim_dt
            #print(f"Sim Time: {current_sim_time:.2f}s | Real Time: {elapsed:.4f}s")
        
        # End-of-run handling
        print("Simulation completed successfully")
        simulation_app.close()
    
if __name__=="__main__":
    
    try:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    except NameError:
        PROJECT_ROOT = os.getcwd()

    print("Running IsaacEnv in standalone debug mode...")
    DEFAULT_SAMPLE_ROOT = os.path.join(PROJECT_ROOT, "data", "sample")
    DATA_ROOT = os.environ.get("RGBENCH_DATA_ROOT", DEFAULT_SAMPLE_ROOT)
    CLOTH_MESH_ROOT = os.environ.get("RGBENCH_CLOTH_MESH_ROOT", os.path.join(DEFAULT_SAMPLE_ROOT, "cloth_meshes"))
    data_path = os.environ.get("RGBENCH_PIPER_DATA", os.path.join(DATA_ROOT, "Piper_Data"))

    active_run_for_debug = {
        'cloth': {
            'name': 'green_tshirt',
            'sample_id': 2,
            'model_path': os.path.join(CLOTH_MESH_ROOT, 'cloth_model.obj'),
            'init_pose': [0.1, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
            'init_pose_path': '/path/to/non_existent_file.json',
        },
        'env': {
            'name': 'isaacsim',
            'model_path': os.path.join(PROJECT_ROOT, 'assets', 'mujoco_model', 'cloth_scene.xml'),
            'sim_flex_name': 'cloth',
            'enable_gripper_calibrate': False,  # Enable gripper calibration
            'timestep': 0.001 # no use
        },
        'data': {
            'root': data_path,
            'robot_joints': {
                'left_arm_csv_path':  osp.join(data_path, "right_arm_end_pose_piper.csv"),
                'right_arm_csv_path': osp.join(data_path, "left_arm_end_pose_piper.csv"),
            },
            'sim_start_time': 0.0,
        },
    }
    cfg = OmegaConf.create(active_run_for_debug)
    
    try:
        player = IsaacsimEnv(cfg=cfg)

        # Option 1: Interactive Preview
        print("\nStarting interactive simulation...")
        player.run()



    except FileNotFoundError as e:
        print(f"Error: File not found. Please check paths. Details: {e}")
    except ValueError as e:
        print(f"Error: Value error during initialization or data loading. Details: {e}")
    except Exception as e: # other unknown errors
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    

    while simulation_app.is_running():
        simulation_app.update()



