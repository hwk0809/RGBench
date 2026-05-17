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

import omni.usd

try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
except NameError:
    PROJECT_ROOT = os.getcwd()

from rgbench.csv_data import load_processed_data, JointState

ArmType = Literal["left", "right"]


class ArmConfigData(TypedDict):
    """Stores configuration and data for one arm."""
    name: ArmType
    isaacsim_joint_names: List[str]
    data: List[JointState]

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

    # # Find the point with the minimum x value
    # if len(world_points) > 0:
    #     min_x_idx = np.argmin(world_points[:, 0])
    #     min_x_point = world_points[min_x_idx]
    #     # Print world-frame coordinates
    #     print(f"world-frame min X point: index={min_x_idx}, coord=({min_x_point[0]:.6f}, {min_x_point[1]:.6f}, {min_x_point[2]:.6f})")
    # else:
    #     print("warning: point cloud is empty")


    # # Convert to numpy array and downsample
    # points = np.array([(p[0], p[1], p[2]) for p in points], dtype=np.float32)
    # points = points[::downsample]

    # point=point_array
    
    # # Build Open3D point cloud object
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    #
    # # Compute colors (based on Z height)
    # z_min, z_max = np.min(points[:,2]), np.max(points[:,2])
    # colors = np.zeros((len(points), 3))
    # colors[:, 0] = (points[:,2] - z_min) / (z_max - z_min + 1e-6)  # R
    # colors[:, 1] = 1 - colors[:,0]  # G
    # colors[:, 2] = 0.5  # B
    # #colors[:, 0] = 1.0
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    #
    #
    # # Create visualization window for debugging
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name=f"Cloth Point Cloud - {prim_path}", width=1024, height=768)
    # vis.add_geometry(pcd)
    #
    # # Add coordinate axes
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # vis.add_geometry(coordinate_frame)
    #
    # # Set camera viewpoint
    # ctr = vis.get_view_control()
    # ctr.set_front([0, -1, 0.5])  # camera direction
    # ctr.set_up([0, 0, 1])       # camera up vector
    #
    # # Display point cloud (blocking)
    # vis.run()
    # vis.destroy_window()

    return world_points
    
class PiperCloth_Env(BaseEnv):
    def __init__(
        self, 
        usd_path:str=None, 
        cfg: DictConfig=None,
        **kwargs
    ):
        self.cfg = cfg
        # load BaseEnv
        physics_timestep = cfg.env.get('physics_timestep', 1/500)
        rendering_timestep = cfg.env.get('rendering_timestep', 1/500)
        super().__init__(physics_timestep, rendering_timestep)
        
        # ------------------------------------ #
        # ---        Add Env Assets        --- #
        # ------------------------------------ #

        # Warnning: it will expode if you change the param
        my_particle_mass = cfg.env.get('particle_mass', 1e-2)
        my_stretch_stiffness = cfg.env.get('stretch_stiffness', 1e6)
        my_friction = cfg.env.get('friction', 100)

        # springElasticStiffness=float(self.cfg.cloth_params.stretch)
        # springDampingStiffness=float(self.cfg.cloth_params.damping)
        # springBendingStiffness=float(self.cfg.cloth_params.bending)
        # frictionCoeff=float(self.cfg.cloth_params.friction)


        robot_defalt_path =  os.path.join(PROJECT_ROOT, "assets","isaacsim_assets","Robot","piper_usd", "piper_urdf.usda")
        print("Robot USD Path:", robot_defalt_path)
        left_arm_path = cfg.env.get('piper_usda', robot_defalt_path)
        right_arm_path = cfg.env.get('piper_usda', robot_defalt_path)

        self.ground = Real_Ground(
            self.scene, 
            visual_material_usd = None,
            usd_path = os.path.join(PROJECT_ROOT, "assets","isaacsim_assets", "Scene", "default_ground.usd"),
            # you can use materials in 'Assets/Material/Floor' to change the texture of ground.
        )

        # cloth parameters
        friction = cfg.env.get('friction', 25.0)
        damping = cfg.env.get('damping', 0)
        stretch_stiffness = cfg.env.get('stretch_stiffness', 2.3e5)
        bend_stiffness = cfg.env.get('bend_stiffness', 1e3)
        shear_stiffness = stretch_stiffness
        cloth_usd_path = self.cfg.cloth.usda_model_path
        print("Cloth USD Path:", cloth_usd_path)

        # cloth color 
        material_path = os.path.join(PROJECT_ROOT, "assets","isaacsim_assets", "Material", "Garment")
        visual_material_usd_white = os.path.join(material_path, "linen_White.usd")
        visual_material_usd_Beige = os.path.join(material_path, "linen_Beige.usd")
        visual_material_usd_Blue = os.path.join(material_path, "linen_Blue.usd")
        visual_material_usd_Pumpkin = os.path.join(material_path, "linen_Pumpkin.usd")

        self.garment = Particle_Garment(
            self.world, 
            pos=np.array([0.1, 0, 0.01]),
            ori=np.array([0.0, 0.0, 0.0]),
            scale =np.array([1.0, 1.0, 1.0]),
            usd_path = cloth_usd_path,
            visual_material_usd = visual_material_usd_white,
            # stretch_stiffness= stretch_stiffness,
            # bend_stiffness = bend_stiffness,
            # shear_stiffness = shear_stiffness,
            friction = 100,
            # damping = damping,
            #particle_mass=255/10000,          # important parameter default 1e-2
            # stretch_stiffness=2.3e5,
            # bend_stiffness = 1e3,
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
        # self.garment.set_mass(255)  # mass 255g
        
        
        left_arm_path = robot_defalt_path if left_arm_path is None else left_arm_path
        add_reference_to_stage(usd_path=left_arm_path, prim_path="/World/PiperLeft")  # left arm path
        self.left_arm = Robot(
            prim_path="/World/PiperLeft", 
            name="left_piper_arm",
            translation=np.array([0.5, 0, 0.00]) 
        )
        self.world.scene.add(self.left_arm)

        # 3. Load right arm
        #right_arm_path = "./piper_description.usd"  # same model
        right_arm_path = robot_defalt_path if right_arm_path is None else right_arm_path
        add_reference_to_stage(usd_path=right_arm_path, prim_path="/World/PiperRight")  # right arm path
        self.right_arm = Robot(
            prim_path="/World/PiperRight", 
            name="right_piper_arm",
            translation=np.array([-0.5, 0, 0.00]) #
        )
        self.world.scene.add(self.right_arm)

        # vis cube (debug tools)
        vis_cube = VisualCuboid(
            prim_path = "/World/vis_cube_left",
            color=np.array([255, 0, 0]),
            name = "vis_cube_left", 
            position = [0.0, 0.0, 2.0],
            size = 0.1,
            visible = True,
        )
        self.world.scene.add(
            vis_cube
        )
        
        self.env_camera = Recording_Camera(
            camera_position=np.array([2, 0, 1.5]),
            #camera_orientation=np.array([0, 60, -90.0]),
            prim_path="/World/env_camera",
        )
        
        set_camera_view(
            eye=[4, 0, 3],
            target=[0.1, 0.0, 0.0],
            camera_prim_path="/World/env_camera",
        )
        # ------------------------------------ #
        # --- Initialize World to be Ready --- #
        # ------------------------------------ #
        # initialize world
        self.reset()

        self.left_arm.pos=np.array([0, -0.25, 0.00])
        self.right_arm.pos=np.array([0, 0.25, 0.00])
        self.left_arm.ori=np.array([0.0, 0.0, 0.0])
        self.right_arm.ori=np.array([0.0, 0.0, 0.0])
        self.left_arm.quat = euler_angles_to_quat(self.left_arm.ori, degrees=True)
        self.right_arm.quat = euler_angles_to_quat(self.right_arm.ori, degrees=True)
        self.left_arm.set_world_pose(position=self.left_arm.pos, orientation = self.left_arm.quat)
        self.right_arm.set_world_pose(position=self.right_arm.pos, orientation = self.right_arm.quat)
        
        self.left_arm.set_default_state(position=self.left_arm.pos, orientation=self.left_arm.quat)
        self.left_arm.set_joints_default_state(
            np.array([
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ])
        )
        self.left_arm.post_reset()

        self.right_arm.set_default_state(position=self.right_arm.pos, orientation=self.right_arm.quat)
        self.right_arm.set_joints_default_state(
            np.array([
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ])
        )
        self.right_arm.post_reset()


                # initialize gif camera to obtain rgb with the aim of creating gif
        self.env_camera.initialize(depth_enable=True)
        
        # add thread and record gif Asynchronously(use to collect rgb data for generating gif)
        # Video recording parameters
        self.recording = True
        self.video_fps = 30  # target frame rate (30fps approx. normal speed)
        
        # Modify recording thread startup
        if self.recording:
            self.thread_record = threading.Thread(
                target=self.env_camera.collect_rgb_graph_for_vedio,
            )
            self.thread_record.daemon = True
            self.thread_record.start()


        # # step world to make it ready
        # for i in range(100):
        #     self.step()

        self.left_arm.arm_dof_names = [
        "joint1", 
        "joint2", 
        "joint3",
        "joint4", 
        "joint5", 
        "joint6",
        "joint7",
        "joint8"
        ]
        self.left_arm.arm_dof_indices = [self.left_arm.get_dof_index(dof_name) for dof_name in self.left_arm.arm_dof_names]

        left_random_pose = ArticulationAction(
            joint_positions=np.deg2rad(np.array([70,0,0,0,0,0,0,0])),
            joint_indices=np.array(self.left_arm.arm_dof_indices)
        )

        left_random_pose2 = ArticulationAction(
            joint_positions=np.deg2rad(np.array([-30,0,0,0,0,0,0,0])),
            joint_indices=np.array(self.left_arm.arm_dof_indices)
        )

        left_random_pose3 = ArticulationAction(
            joint_positions=np.deg2rad(np.array([0,0,0,0,0,0,0,0])),
            joint_indices=np.array(self.left_arm.arm_dof_indices)
        )

        


        self.right_arm.arm_dof_names = [
        "joint1", 
        "joint2", 
        "joint3",
        "joint4", 
        "joint5", 
        "joint6",
        "joint7",
        "joint8"
        ]
        self.right_arm.arm_dof_indices = [self.right_arm.get_dof_index(dof_name) for dof_name in self.right_arm.arm_dof_names]


    
        right_random_pose = ArticulationAction(
            joint_positions=np.array([-0.45048693, 1.22996843, -0.28687977, -1801916, 0.53476888, 0.021485,0 ,0]),
            #joint_positions=np.deg2rad(np.array([10,20,30,40,50,60,0.3,-0.3])),
            joint_indices=np.array(self.right_arm.arm_dof_indices)
        )

        right_random_pose2 = ArticulationAction(
            joint_positions=np.deg2rad(np.array([30,0,0,0,0,0,0,0])),
            joint_indices=np.array(self.left_arm.arm_dof_indices)
        )

        right_random_pose3 = ArticulationAction(
            joint_positions=np.deg2rad(np.array([0,0,0,0,0,0,np.rad2deg(0.035),np.rad2deg(-0.035)])),
            joint_indices=np.array(self.left_arm.arm_dof_indices)
        )

        # # Piper_Move
        # self.left_arm.apply_action(left_random_pose)
        # self.right_arm.apply_action(right_random_pose)

        # for i in range(100):
        #     self.step()

        # self.left_arm.apply_action(left_random_pose2)
        # self.right_arm.apply_action(right_random_pose2)

        

        # self.left_arm.apply_action(right_random_pose3)
        # self.right_arm.apply_action(right_random_pose3)
        for i in range(100):
            self.step()

        # print(self.right_arm.get_joint_positions(joint_indices=np.array([6, 7])))
        # #print("11111111111111111111111")
        # print("Time NOW", self.world.current_time)

        # FK
        # ee_prim_path = f"/World/PiperLeft/link6"  # or "gripper", "tool0", etc.
        
        # # 3. Get the end-effector USD Prim
        # stage = self.world.stage
        # ee_prim = stage.GetPrimAtPath(ee_prim_path)
        
        # if not ee_prim.IsValid():
        #     raise ValueError(f"end-effector Prim does not exist: {ee_prim_path}; check the model structure")
        
        # # 4. Compute the pose in world coordinates
        # # Get the local-to-world transform matrix
        # xform = UsdGeom.Xformable(ee_prim)
        # world_transform = xform.ComputeLocalToWorldTransform(self.world.current_time_step_index)
        
        # # Extract position (translation part of the transform)
        # translation = world_transform.ExtractTranslation()
        # position = np.array([translation[0], translation[1], translation[2]])
        
        # # Extract orientation (quaternion from the transform)
        # rotation = world_transform.ExtractRotationQuat()
        # orientation = np.array([rotation.imaginary[0], 
        #                         rotation.imaginary[1],
        #                         rotation.imaginary[2],
        #                         rotation.real])
        # print(position)


        #print("Pose",self.right_arm.set_joint_positions([0,0,0,0,0,0,0.035,-0.035]))

        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])
        cprint(f"usd_path: {usd_path}", "magenta")
        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])
        
        cprint("World Ready!", "green", "on_green")

        # while simulation_app.is_running():
        #     self.world.step(render=True)  # physics step + render


import abc
import numpy as np
from omegaconf import DictConfig


class IsaacsimEnv(BaseEnvWrapper):
    def __init__(self, cfg: DictConfig, **kwargs):
        # First, call the base class __init__
        super().__init__(cfg)

        self.env_cfg = cfg.env
        # self.sim_timestep = self.env_cfg.get("timestep", 0.005)
        # --- Initialize attributes from DualArmController ---
        self.playback_start_time_abs: Optional[float] = None
        self.playback_end_time_abs: Optional[float] = None
        self.arms: Dict[ArmType, ArmConfigData] = {}

        # --- Initialize interpolation trackers ---
        # This tracker is crucial for the step_to_time method
        self.time_trackers = {'left': {'idx': 0}, 'right': {'idx': 0}}
        self._init_arms_config(cfg)

        self.max_gripper = 0
        self.min_gripper = 0
        # initial setting
        usd_path = cfg.cloth.get('model_path', None)
        print(usd_path)
        self.robot=PiperCloth_Env(usd_path, cfg)

        #self.robot.thread_record.start()

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
        """
        Core method: Steps the simulation forward to a specific point in time.
        This drives the robot motion and physics evolution.
        """
        
        current_absolute_time = self.get_master_start_time() + self.robot.world.current_time
        judge_right=True
        # Update interpolated joint positions for each arm
        for arm_name, arm_config in self.arms.items():
            if not arm_config['data']:
                continue

            interpolated_positions = self._get_interpolated_positions(
                target_time=current_absolute_time,
                arm_data=arm_config['data'],
                tracker=self.time_trackers[arm_name]
            )

            if interpolated_positions:
                # print("11111111111111111111111")
                # print(arm_name)
                # print(interpolated_positions)
                if judge_right:
                    self.right_state = self._apply_arm_state(arm_config, interpolated_positions)
                    judge_right = False
                else:
                    self.left_state = self._apply_arm_state(arm_config, interpolated_positions)

        self.robot.right_arm.apply_action(self.right_state)
        self.robot.left_arm.apply_action(self.left_state)
        # Perform one physics simulation step
        self.robot.world.step()
        simulation_app.update()

    def step_to_time(self, target_time: float) -> None:
        """
        Core method: Steps the simulation forward to a specific point in time.
        This drives the robot motion and physics evolution.
        """
        while self.robot.world.current_time < target_time:
            # Calculate the absolute time for interpolation
            current_absolute_time = self.get_master_start_time() + self.robot.world.current_time
            judge_right=True
            # Update interpolated joint positions for each arm
            for arm_name, arm_config in self.arms.items():
                if not arm_config['data']:
                    continue

                interpolated_positions = self._get_interpolated_positions(
                    target_time=current_absolute_time,
                    arm_data=arm_config['data'],
                    tracker=self.time_trackers[arm_name]
                )

                if interpolated_positions:
                    # print("11111111111111111111111")
                    # print(arm_name)
                    # print(interpolated_positions)
                    if judge_right:
                        self.right_state = self._apply_arm_state(arm_config, interpolated_positions)
                        judge_right = False
                    else:
                        self.left_state = self._apply_arm_state(arm_config, interpolated_positions)

            self.robot.right_arm.apply_action(self.right_state)
            self.robot.left_arm.apply_action(self.left_state)
            # Perform one physics simulation step
            self.robot.world.step()
            simulation_app.update()

    def _init_arms_config(self, cfg: DictConfig):
        data_cfg = cfg.data
        # Key: declare joint names and order from the URDF here
        arm_setups = {
            'left': {
                'csv_path': data_cfg.robot_joints.get('left_arm_csv_path'),
                # These names must match the joint names in your URDF exactly
                'isaacsim_joints': [f'joint{i}' for i in range(1, 7)],
                'isaacsim_gripper': ('joint7', 'joint8'),
            },
            'right': {
                'csv_path': data_cfg.robot_joints.get('right_arm_csv_path'),
                # The right arm may use a different naming convention, e.g., an "_arm2" suffix
                # We assume both arms use the same URDF here, so joint names are identical
                'isaacsim_joints': [f'joint{i}' for i in range(1, 7)],
                'isaacsim_gripper': ('joint7', 'joint8'),
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
                isaacsim_names = setup['isaacsim_joints'] + list(setup['isaacsim_gripper'])
                self.arms[arm_name] = ArmConfigData(
                    name=arm_name,
                    isaacsim_joint_names=isaacsim_names,
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


    def _apply_arm_state(self, config, interpolated_positions):
        """Apply interpolated joint positions to the arm controller.

        Args:
            config (dict): arm configuration parameters
            interpolated_positions (dict): target joint position dict, e.g., {'joint1': radians, ..., 'gripper': opening}

        Returns:
            ArticulationAction: encapsulated joint control action
        """
        # 1. Build the joint position array (in arm_dof_names order)
        joint_positions = []
        joint_positions = interpolated_positions.copy()
        joint_positions.append(joint_positions[-1]* -1)

        return ArticulationAction(
            joint_positions=np.array(joint_positions, dtype=np.float32),
            joint_indices=np.array(self.robot.right_arm.arm_dof_indices)
        )


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


    # ===================================================================
    # Online and offline run simulation in isaacsim
    # ===================================================================
    def run(self):
        """Main control loop for the simulation."""

        if self.playback_start_time_abs is None:
            print("Wrong: Playback start time is not set. Cannot run simulation.")
            return

        time_trackers = {name: {'idx': 0} for name, config in self.arms.items() if config['data']}
        sim_dt = 0.005
        step_start_wall_time = time.time()
        #print(step_start_wall_time)
        
        
        while simulation_app.is_running():
            
            current_target_time = self.robot.world.current_time + self.playback_start_time_abs
            judge_right=True
            for arm_name, config in self.arms.items():
                if not config['data']:
                    continue

                # interpolated joint positions
                interpolated_positions = self._get_interpolated_positions(
                    current_target_time,
                    config['data'],
                    time_trackers[arm_name]
                )
                
                if interpolated_positions:
                    # print("11111111111111111111111")
                    # print(arm_name)
                    #print(interpolated_positions)
                    if judge_right:
                        self.right_state = self._apply_arm_state(config, interpolated_positions)
                        judge_right = False
                    else:
                        self.left_state = self._apply_arm_state(config, interpolated_positions)
                        
            self.robot.right_arm.apply_action(self.right_state)
            self.robot.left_arm.apply_action(self.left_state)

            self.robot.world.step()
            simulation_app.update()


        video_dir = "isaacsim/video"
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        self.robot.env_camera.create_mp4(get_unique_filename(os.path.join(video_dir, "video"), ".mp4"))
        simulation_app.close()
    
if __name__=="__main__":
    

    print("Running IsaacEnv in standalone debug mode...")
    data_path = os.path.join(PROJECT_ROOT, "data","blue_dress_2025-11-18-18-13-48")
    cloth_path = os.environ.get(
        "RGBENCH_CLOTH_MESH_ROOT",
        os.path.join(PROJECT_ROOT, "data", "sample", "cloth_meshes"),
    )
    
    active_run_for_debug = {
        'cloth': {
            'name': 'green_tshirt',
            'sample_id': 2,
            'model_path': os.path.join(cloth_path, "Green_Tshirt", "green_tshirt_10k.usda"),
            'usda_model_path': os.path.join(cloth_path, "Green_Tshirt", "green_tshirt_10k.usda"),
            'init_pose': [0.1, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
            'init_pose_path': '/path/to/non_existent_file.json',
        },
        'env': {
            'name': 'isaacsim',
            'sim_flex_name': 'cloth',
            'timestep': 0.001 # no use
        },
        'cloth_params':{
         'cloth_model_usda': os.path.join(cloth_path, "Green_Tshirt", "green_tshirt_10k.usda"),
        },
        'data': {
            'root': data_path,
            'robot_joints': {
                'left_arm_csv_path':  osp.join(data_path, "joints", "left_arm_joint_states_and_end_pose.csv"),
                'right_arm_csv_path': osp.join(data_path, "joints", "right_arm_joint_states_and_end_pose.csv"),
            },
            'sim_start_time': 1.0,
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
    
#simulation_app.close()


