import os.path as osp
import os
import sys
import mujoco
from mujoco import viewer
import ast
import time
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
from typing import Dict, Tuple, Optional, List, Any, Literal, TypedDict, cast
import numpy as np
import json
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import hydra
import imageio
from tqdm import tqdm
import xml.etree.ElementTree as ET

from rgbench.envs.base import BaseEnvWrapper
from rgbench.csv_data import load_processed_data, JointState, PoseState

# --- Type Definitions ---
ArmType = Literal["left", "right"]

class GrabberState(TypedDict, total=False):
    """Stores the state and data for one grabber (arm end pose)."""
    is_active: bool
    is_grabbed: bool
    pose_data: List[PoseState]
    pose_tracker: Dict[str, int]
    vertex_id: int
    mocap_id: int
    initial_grab_vertex_pose: Optional[List[float]]  # Initial grab pose for the vertex, if applicable

def get_flex_vertices(model: mujoco.MjModel, data: mujoco.MjData, flex_name: str) -> np.ndarray:
    """Retrieves the vertices of a flex object from the MuJoCo simulation."""
    try:
        flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, flex_name)
        vert_adr = model.flex_vertadr[flex_id]
        vert_num = model.flex_vertnum[flex_id]
        if vert_num == 0:
            return np.array([])
        return data.flexvert_xpos[vert_adr: vert_adr + vert_num - 1] # the last point is ghost point
    except KeyError:
        print(f"Warning: Flexcomp '{flex_name}' not found in model.")
        return np.array([])

class MujocoFixedPointEnv(BaseEnvWrapper):
     def __init__(self, cfg: DictConfig, **kwargs):
        # First, call the base class __init__
        super().__init__(cfg)

        # --- Initialize config ---
        self.env_cfg = cfg.env
        self.grab_points_cfg = self.env_cfg.get('grab_points', {'enabled': False})
        self.sim_timestep = self.env_cfg.get("timestep", 0.002)
        self.left_base_offset = [0, 0.25, 0]
        self.right_base_offset = [0, -0.25, 0]

        # --- STAGE 0: Determine Action Type and Parameters ---
        self.action_type = self.cfg.action.get('type', 'grasp') # Default to 'grasp' for backward compatibility
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

        # --- STAGE 1: Determine Pin Indices ---
        pin_indices = self._determine_pin_indices()

        # --- STAGE 2: Load Final Model with Pinning Configured ---
        self.model = self._load_and_configure_model(pin_indices=pin_indices)
        self.data = mujoco.MjData(self.model)
        # --- initial mujoco cloth id and body id ---
        try:
            self.flex_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_FLEX, self.env_cfg.sim_flex_name)
            self.flex_vert_adr = self.model.flex_vertadr[self.flex_id]
        except KeyError as e:
            raise RuntimeError(
                f"FATAL: Could not find required object in XML: {e}. Please check sim_flex_name ('{self.env_cfg.sim_flex_name}') and body name ('cloth_body').")
        mujoco.mj_forward(self.model, self.data)

        # --- STAGE 3: Initialize Data and Grab Management ---
        self.grabbers : Dict[str, Dict[str, Any]]= self._init_grabbers(self.model, pin_indices=pin_indices)
        if not self.grabbers:
            raise ValueError("Simulation cannot start: No active grabbers were configured.")


        # --- Set Global Playback Timeline ---
        self.playback_start_time_abs: Optional[float] = None
        self.playback_end_time_abs: Optional[float] = None

        all_start_times = [g['pose_data'][0]['time'] for g in self.grabbers.values() if g.get('is_active')]
        all_end_times = [g['pose_data'][-1]['time'] for g in self.grabbers.values() if g.get('is_active')]
        self.playback_start_time_abs = min(all_start_times)
        self.playback_end_time_abs = max(all_end_times)

        duration = self.playback_end_time_abs - self.playback_start_time_abs
        logger.info("-" * 30)
        logger.info("Global Playback Timeline Determined:")
        logger.info(f"  - Absolute Start Time: {self.playback_start_time_abs:.4f}")
        logger.info(f"  - Absolute End Time:   {self.playback_end_time_abs:.4f}")
        logger.info(f"  - Total Duration:      {duration:.2f} seconds")
        logger.info("-" * 30)
        logger.success("MujocoDualFixedPointEnv initialization successful.")

        # --- STAGE 4: Initialize Viewer if enabled ---
        self.viewer_handle = None
        if "visualization" in cfg and cfg.visualization.get('vis_sim', False):
            logger.info("Viewer enabled. Launching passive viewer.")
            self.viewer_handle = viewer.launch_passive(self.model, self.data)
        else:
            logger.info("Viewer is disabled.")

     # ===================================================================
     # Initial Method
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
                 pin_indices = self._find_pin_indices_auto()
                 logger.success(f"Automatically determined pin indices: {pin_indices}")
                 return pin_indices
             else:
                 raise ValueError(f"Unknown pinning mode: '{mode}'. Must be 'auto' or 'manual'.")
         else:
             raise ValueError(f"Unknown action type: '{self.action_type}'.")


     def _init_grabbers(self, model: mujoco.MjModel, pin_indices: Optional[List[int]] = None) -> Dict[str, GrabberState]:
         """Initializes all grabbers based on the configuration. Now only loads pose data."""
         active_grabbers: Dict[str, GrabberState] = {}

         arm_index = 0
         for arm_name in ['left', 'right']:
             arm_cfg_path = self.cfg.data.robot_joints.get(f'{arm_name}_arm_csv_path')
             if not arm_cfg_path:
                 continue

             base_offset = self.left_base_offset if arm_name=='left' else self.right_base_offset
             pose_data = load_processed_data(
                 arm_cfg_path, mode='pose',
                 start_time_offset=self.cfg.data.get('sim_start_time', 0.0),
                 base_translation=base_offset
             )
             if not pose_data:
                 logger.error(f"Could not load data for '{arm_name}' arm. It will be disabled.")
                 continue

             try:
                 # Use the passed-in model object
                 mocap_id = model.body(f'grab_point_body_{arm_name}').mocapid[0]
             except KeyError:
                 mocap_id = -1

             active_grabbers[arm_name] = {
                 'is_active': True,
                 'pose_data': pose_data,
                 'pose_tracker': {'idx': 0},
                 'mocap_id': mocap_id,
             }

             if pin_indices is not None:
                 vertex_id = pin_indices[arm_index]
                 active_grabbers[arm_name]['vertex_id'] = vertex_id
                 arm_index += 1
                 initial_vertex_pos = self.get_sim_vertices()
                 active_grabbers[arm_name]['initial_grab_vertex_pose'] = initial_vertex_pos[vertex_id].tolist()
                 logger.info(f"Assigning vertex_id {vertex_id} to grabber '{arm_name}'. position: {initial_vertex_pos[vertex_id]}")



         return active_grabbers

     def _find_pin_indices_auto(self) -> List[int]:
         """
         Performs a pre-calculation to find optimal pin indices without a full simulation.
         It loads a temporary model to get initial positions.
         """
         logger.info("--- Pre-calculating grabber vertices based on initial state ---")

         # 1. Load a temporary, unpinned model just for this calculation.
         # Passing pin_indices=None ensures no pin config is written.
         temp_model = self._load_and_configure_model(pin_indices=None)
         temp_data = mujoco.MjData(temp_model)

         # We need to load temporary grabber data to get the timeline
         temp_grabbers = self._init_grabbers(temp_model)

         mujoco.mj_forward(temp_model, temp_data)
         initial_verts_pos = get_flex_vertices(temp_model, temp_data, self.env_cfg.sim_flex_name)
         if initial_verts_pos.size == 0:
             raise RuntimeError("Pre-calculation failed: Cloth has no vertices.")

         # 2. Load pose data to find grabber positions at grab_time
         temp_start_times = [g['pose_data'][0]['time'] for g in temp_grabbers.values() if g.get('is_active')]
         if not temp_start_times:
             raise ValueError("Pre-calculation failed: No active grabbers with pose data.")

         playback_start_time_abs = min(temp_start_times)
         target_sim_time = self.grab_time
         if target_sim_time is None:
             raise ValueError("Pre-calculation failed: 'grasp_time' is not defined for auto mode.")
         target_absolute_time = playback_start_time_abs + target_sim_time

         pin_indices = []
         for arm_name, grabber in temp_grabbers.items():
             if not grabber.get('is_active'):
                 continue

             interpolated_pose = self._get_interpolated_pose(
                 target_absolute_time, grabber['pose_data'], {'idx': 0}
             )
             if not interpolated_pose:
                 raise RuntimeError(f"Could not determine pose for '{arm_name}' at grab_time.")

             grabber_target_pos = np.array(interpolated_pose['position'])
             distances = np.linalg.norm(initial_verts_pos - grabber_target_pos, axis=1)
             closest_vertex_idx = int(np.argmin(distances))
             pin_indices.append(closest_vertex_idx)
             logger.info(f"For '{arm_name}', target pos is closest to initial vertex #{closest_vertex_idx}")

         if not pin_indices:
             raise RuntimeError("Could not find any pin vertices for active arms.")

         return pin_indices

     # ===================================================================
     # Initial XML from Config
     # ===================================================================
     def _load_and_configure_model(self, pin_indices: Optional[List[int]] = None) -> mujoco.MjModel:
         """
         Loads the XML file as a template, modifies it based on the config,
         and then compiles it into a MuJoCo model.
         """
         xml_path = self.env_cfg.fixed_point_model_path
         xml_dir = os.path.dirname(os.path.abspath(xml_path))
         print(f"Loading XML template from: {xml_path}")
         tree = ET.parse(xml_path)
         root = tree.getroot()

         # Resolve asset paths
         compiler = root.find('compiler')
         if compiler is not None:
             for attr in ['meshdir', 'texturedir']:
                 original_path = compiler.get(attr)
                 if original_path:
                     absolute_path = os.path.join(xml_dir, original_path)
                     compiler.set(attr, absolute_path)

         # 1. Configure simulation options
         option_element = root.find('option')
         option_element.set('timestep', str(self.sim_timestep))
         logger.info(f"Set Sim timestep to: {self.sim_timestep}")

         # 2. Configure cloth object
         cloth_flex_element = root.find(f".//flexcomp[@name='{self.env_cfg.sim_flex_name}']")
         if cloth_flex_element is None:
             raise ValueError(f"Could not find <flexcomp name='{self.env_cfg.sim_flex_name}'> in {xml_path}")

         # Modify the cloth's geometry (.obj file) based on the config
         if 'model_path' in self.cfg.cloth and self.cfg.cloth.model_path:
             new_obj_path = self.cfg.cloth.model_path
             print("new_obj_path", new_obj_path)
             cloth_flex_element.set('file', new_obj_path)
             logger.info(f"Set cloth geometry to: {new_obj_path}")

         # Modify the cloth's initial pose based on the config
         pose = self._get_initial_pose_from_config()
         pos_str = " ".join(map(str, pose['pos']))
         quat_str = " ".join(map(str, pose['quat']))
         cloth_flex_element.set('pos', pos_str)
         cloth_flex_element.set('quat', quat_str)
         logger.info(f"Set cloth initial pose to: pos={pos_str}, quat={quat_str}")

         # Compile the modified XML string into a model
         modified_xml_string = ET.tostring(root, encoding='unicode')
         return mujoco.MjModel.from_xml_string(modified_xml_string)

     def _update_plugin_config(self, parent_element: ET.Element, key: str, value: any):
         """help to find key in plugin config and update its value"""
         xpath = f"./plugin/config[@key='{key}']"
         config_element = parent_element.find(xpath)
         if config_element is not None:
             str_value = str(value)
             config_element.set('value', str_value)
             logger.info(f"Set cloth plugin param '{key}' to: {str_value}")
         else:
             logger.warning(f"Cloth plugin param key '{key}' not found in the XML template.")

     def _get_initial_pose_from_config(self) -> Optional[Dict[str, List[float]]]:
         """
         Loads the initial pose, prioritizing a file path if provided,
         otherwise falling back to the pose defined directly in the YAML.
         """
         cloth_cfg = self.cfg.cloth
         # Check if a specific pose file is provided
         pose_path = cloth_cfg.get('init_pose_path', None)
         if pose_path and os.path.exists(pose_path):
             try:
                 logger.info(f"Attempting to load initial pose from file: {pose_path}")
                 with open(pose_path, 'r') as f:
                     return json.load(f)
             except (FileNotFoundError, json.JSONDecodeError) as e:
                 logger.warning(f"Failed to load pose from {pose_path}: {e}. Falling back to default pose in config.")

         # If file failed, check if the pose is defined directly in the config
         pose_direct_list = cloth_cfg.get('init_pose')
         if pose_direct_list:
             logger.info("Using initial pose directly from YAML config.")
             pose_list = OmegaConf.to_container(pose_direct_list, resolve=True)
             if isinstance(pose_list, list) and len(pose_list) == 7:
                 #  [x, y, z, w, qx, qy, qz]
                 return {'pos': pose_list[0:3], 'quat': pose_list[3:7]}
             else:
                 logger.warning(f"init_pose in YAML is not a list of 7 numbers. Falling back...")

         logger.error("CRITICAL: No valid initial pose found. Using hardcoded default pose at origin.")
         return {
             'pos': [0.0, 0.0, 0.0],
             'quat': [1.0, 0.0, 0.0, 0.0]  # identity quaternion in (w, x, y, z) format
         }

     # ===================================================================
     # Core Simulation Step Logic
     # ===================================================================
     def step(self) -> None:
         """Dispatcher for the main simulation step."""
         if self.action_type == 'grasp' or self.action_type == 'fold':
             self._step_grasp()
         elif self.action_type == 'fling':
             self._step_fling()
         else:
             # Default behavior if action type is unknown: just step physics
             mujoco.mj_step(self.model, self.data)

     def _step_grasp(self) -> None:
         """Handles the logic for a single step in 'grasp' mode."""
         current_absolute_time = self.get_master_start_time() + self.data.time

         for arm_name, grabber in self.grabbers.items():
             if not grabber.get('is_active'): continue


             interpolated_pose = self._get_interpolated_pose(
                 current_absolute_time, grabber['pose_data'], grabber['pose_tracker'])

             if interpolated_pose:
                 target_pos = np.array(interpolated_pose['position'])
                 target_quat = np.array(interpolated_pose['orientation'])

                 if grabber.get('mocap_id', -1) != -1:
                     self.data.mocap_pos[grabber['mocap_id']] = target_pos
                     self.data.mocap_quat[grabber['mocap_id']] = target_quat

                 if not grabber.get('is_grabbed') and self.data.time >= self.grab_time:
                     grabber['is_grabbed'] = True
                     logger.success(f"GRAB TRIGGERED for '{arm_name}' at sim time {self.data.time:.4f}s!")

                 if grabber.get('is_grabbed') and 'vertex_id' in grabber:
                     self.model.flex_vert[self.flex_vert_adr + grabber['vertex_id']][:] = target_pos

         mujoco.mj_step(self.model, self.data)

     def _step_fling(self) -> None:
         """Handles the logic for a single step in 'fling' mode, including all phases."""
         sim_time = self.data.time

         for arm_name, grabber in self.grabbers.items():
             if not grabber.get('is_active') or 'vertex_id' not in grabber:
                 continue


             # Phase 1: Initial Grasp
             if sim_time < self.fling_prepare_time:
                 alpha = sim_time / self.fling_prepare_time
                 start_pos = grabber['initial_grab_vertex_pose']
                 target_pos = np.array(grabber['pose_data'][0]['position'])
                 current_pos = start_pos + alpha * (target_pos - start_pos)
                 # For simplicity, we snap to the target orientation
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
                     current_absolute_time, grabber['pose_data'], grabber['pose_tracker'])

                 if pose:
                     current_pos = np.array(pose['position'])
                     current_quat = np.array(pose['orientation'])
                 else:  # Should not happen if timeline is correct
                     last_pose = grabber['pose_data'][-1]
                     current_pos = np.array(last_pose['position'])
                     current_quat = np.array(last_pose['orientation'])

             # Apply the calculated pose to mocap and vertex
             if grabber.get('mocap_id', -1) != -1:
                 self.data.mocap_pos[grabber['mocap_id']] = current_pos
                 self.data.mocap_quat[grabber['mocap_id']] = current_quat

             # In fling mode, grab is always active
             self.model.flex_vert[self.flex_vert_adr + grabber['vertex_id']][:] = current_pos

         mujoco.mj_step(self.model, self.data)

     # ===================================================================
     # Helper & Abstract Methods
     # ===================================================================
     def get_master_start_time(self) -> float:
         """Returns the absolute start time from the loaded data."""
         return self.playback_start_time_abs

     def get_current_sim_time(self) -> float:
         """Returns the current MuJoCo simulation time."""
         return self.data.time

     def get_sim_vertices(self) -> np.ndarray:
         """Gets the vertices of the soft body."""
         return get_flex_vertices(self.model, self.data, self.cfg.env.sim_flex_name)

     def step_to_time(self, target_time: float) -> None:
         """
         Core method: Steps the simulation forward to a specific point in time.
         This drives the robot motion and physics evolution.
         """
         while self.data.time < target_time:
             self.step()
             self.render()

     def close(self):
        if self.viewer_handle:
            self.viewer_handle.close()

     def render(self):
         if self.viewer_handle and self.viewer_handle.is_running():
             self.viewer_handle.sync()

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

     # ===================================================================
     # Simulation Execution (Interactive & Recording)
     # ===================================================================
     def run(self):
         """Main control loop for the simulation."""

         viewer_handle = viewer.launch_passive(self.model, self.data)

         if self.playback_start_time_abs is None:
             print("Wrong: Playback start time is not set. Cannot run simulation.")
             return

         sim_dt = self.model.opt.timestep

         while viewer_handle.is_running():
             step_start_wall_time = time.time()
             self.step()
             viewer_handle.sync()

             time_to_wait = sim_dt - (time.time() - step_start_wall_time)
             if time_to_wait > 0:
                 time.sleep(time_to_wait)

         viewer_handle.close()

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
             camera_name: Optional[str]: Default camera viewpoint BaseEnvWrapper
        """
         print(f"--- Preparing to record video to '{output_path}' with interpolation ---")
         if speed <= 0: raise ValueError("Speed must be a positive number.")

         if camera_name:
             print(f"Using named camera: '{camera_name}'")
         else:
             print("Using default free camera.")

         if self.playback_start_time_abs is None or self.playback_end_time_abs is None:
             print("❌ Error: Playback start or end time is not set. Cannot record video.")
             return

         data_duration = self.playback_end_time_abs - self.playback_start_time_abs
         if data_duration <= 0:
             print("❌ Error: No data found.")
             return

         if output_path=="default":
             output_path = (f"{cfg.data.root}/video/simulation_{self.env_cfg.name}_{self.cfg.cloth.name}"
                                f"_step_{player.model.opt.timestep}.mp4")
         output_dir = os.path.dirname(output_path)
         os.makedirs(output_dir, exist_ok=True)

         # --- 1. Initialize simulation and rendering parameters ---
         sim_dt = self.model.opt.timestep

         # Calculate how many physical steps need to be performed in total
         total_sim_steps = int(data_duration / sim_dt)

         # Calculate the target duration and total frames of the video
         target_video_duration = data_duration / speed
         total_video_frames = int(target_video_duration * fps)

         # Calculate the time between rendering two frames (in simulation time)
         # This is the key to deciding when to “sample”.
         if total_video_frames > 0:
             render_interval = data_duration / total_video_frames
         else:
             render_interval = float('inf')  # infinite interval if no rendering is needed

         print(f"  - Total simulation steps: {total_sim_steps}")
         print(f"  - Will render a frame every {render_interval:.4f} simulation seconds.")

         mujoco.mj_resetData(self.model, self.data)

         try:
             renderer = mujoco.Renderer(self.model, height=height, width=width)
         except ValueError as e:
             if 'Image width' in str(e) or 'Image height' in str(e):
                 print(f"⚠️ Warning: Resolution too high. Falling back to 640x480.");
                 width, height = 640, 480
                 renderer = mujoco.Renderer(self.model, height=height, width=width)
             else:
                 raise e

         frames = []
         next_render_time = 0.0

         # --- 2. Main loop: Evolution of the entire simulation in physical steps---
         for step in tqdm(range(total_sim_steps), desc="Simulating Physics"):

             # --- 3. rendering judgment: check if the current simulation time has reached the moment to render the next frame ---
             if self.data.time >= next_render_time:
                 if camera_name:
                     renderer.update_scene(self.data, camera=camera_name)
                 else:
                     renderer.update_scene(self.data)
                 frames.append(renderer.render())
                 next_render_time += render_interval

             # --- 4. physical update ---
             self.step()

         # --- 5. Save Video ---
         print(f"\nSimulation complete. Saving {len(frames)} rendered frames to video...")
         try:
             with imageio.get_writer(output_path, fps=fps) as writer:
                 for frame in tqdm(frames, desc="Saving Video"):
                     writer.append_data(frame)
             print(f"✅ Video successfully saved to '{output_path}'")
         except Exception as e:
             print(f"❌ Error saving video: {e}")
         finally:
             renderer.close()


if __name__ == "__main__":
    try:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    except NameError:
        PROJECT_ROOT = os.getcwd()

    print("Running MujocodEnv in standalone debug mode...")
    DEFAULT_SAMPLE_ROOT = os.path.join(PROJECT_ROOT, "data", "sample")
    DATA_ROOT = os.environ.get("RGBENCH_DATA_ROOT", DEFAULT_SAMPLE_ROOT)
    CLOTH_MESH_ROOT = os.environ.get("RGBENCH_CLOTH_MESH_ROOT", os.path.join(DEFAULT_SAMPLE_ROOT, "cloth_meshes"))
    data_path = os.environ.get("RGBENCH_PIPER_DATA", os.path.join(DATA_ROOT, "Piper_Data"))

    target_camera = 'video_third_view'  # camera name; change if needed

    # debug 1: manual set
    active_run_for_debug = {
        'action': {
            "type": "fling", # "grasp" or "fling"
            "fling_prepare_time": 5.0,  # Only used in fling mode
            "fling_wait_time": 3.0,  # Only used in fling mode
        },
        'cloth': {
            'name': 'green_tshirt',
            'sample_id': 2,
            'model_path': os.path.join(CLOTH_MESH_ROOT, 'LargeT_Flat_Simple_10k.obj'),
            'init_pose': [0.1, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0], # [x, y, z, w, qx, qy, qz]
            # 'init_pose': [0.1, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], # [x, y, z, w, qx, qy, qz]
            'init_pose_path': '/path/to/non_existent_file.json',
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
            'name': 'mujoco',
            'fixed_point_model_path': os.path.join(PROJECT_ROOT, 'assets', 'mujoco_model', 'FlexCloth.xml'),
            'sim_flex_name': 'cloth',
            'timestep': 0.002,
            'grasp': {
                'enabled': True,  # Enable pinning of cloth vertices
                'grasp_time': 8.5,  # Time when the left arm grabs the cloth
                'mode': 'auto',  # 'auto' or 'manual'
                # 'indices': [2457, 2098] # Only used if mode is 'manual'
                'indices': [6020, 4934] # Only used if mode is 'manual'
            },
        },
        'data': {
            'root': data_path,
            'robot_joints': {
                'left_arm_csv_path':  osp.join(data_path, "joints","left_arm_end_pose_piper.csv"),
                'right_arm_csv_path': osp.join(data_path, "joints","right_arm_end_pose_piper.csv"),
            },
            'sim_start_time': 0.0,
        },
    }
    cfg = OmegaConf.create(active_run_for_debug)



    # debug 2: load from config file
    # debug_config_path = os.path.join(PROJECT_ROOT, "config/main.yaml")
    # cfg = OmegaConf.load(debug_config_path).active_run
    # cfg.env = active_run_for_debug['env']


    try:
        player = MujocoFixedPointEnv(cfg=cfg)

        # Option 1: Interactive Preview
        print("\nStarting interactive simulation...")
        player.run()


        # Option 2: Record a video of the same duration as the CSV data (1x speed)
        print("\n--- Recording video at 1.0x speed (real-time match) ---")
        output_filename = f"{data_path}/video/simulation_{target_camera}_step_{player.model.opt.timestep}_40k.mp4"
        # player.record_video(
        #     output_path= output_filename,  # use default path
        #     speed=1.0,
        #     camera_name=target_camera,  # use view
        #     fps=120,
        # )

    except FileNotFoundError as e:
        print(f"Error: File not found. Please check paths. Details: {e}")
    except ValueError as e:
        print(f"Error: Value error during initialization or data loading. Details: {e}")
    except Exception as e: # other unknown errors
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()