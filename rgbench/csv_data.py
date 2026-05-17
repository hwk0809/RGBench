import pandas as pd
import numpy as np
import pickle
import ast
from typing import List, Dict, Tuple, Literal, TypedDict,Optional,Union
from loguru import logger
import os.path as osp
import os
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
# --- Constants ---
from rgbench.robot import PiperRobot, K1DualArmRobot


# --- Constants ---
# The maximum opening of the gripper actuator/joint in the URDF model.
DEFAULT_MUJOCO_PIPER_GRIPPER_MAX_OPENING: float = 0.035
DEFAULT_MUJOCO_K1_GRIPPER_MAX_OPENING: float = 0.01875
# Default raw gripper data min/max values, used as a fallback.
DEFAULT_MAX_RAW_GRIPPER_DATA = 0.0688
DEFAULT_MIN_RAW_GRIPPER_DATA = 0.0000

# --- Type Definitions ---
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


class TrajectoryInterpolator:
    def __init__(self, trajectory_poses: List[PoseState]):
        if len(trajectory_poses) < 2:
            raise ValueError("Trajectory must contain at least two poses for interpolation.")

        times = np.array([p['time'] for p in trajectory_poses])
        positions = np.array([p['position'] for p in trajectory_poses])

        self.min_time = times[0]
        self.max_time = times[-1]

        # self.interp_x = interp1d(times, positions[:, 0], bounds_error=False,
        #                          fill_value=(positions[0, 0], positions[-1, 0]))
        # self.interp_y = interp1d(times, positions[:, 1], bounds_error=False,
        #                          fill_value=(positions[0, 1], positions[-1, 1]))
        # self.interp_z = interp1d(times, positions[:, 2], bounds_error=False,
        #                          fill_value=(positions[0, 2], positions[-1, 2]))
        self.interp_func = interp1d(
            times,
            positions,
            axis=0,  # Interpolate along the first axis (the time axis)
            bounds_error=False,
            fill_value=(positions[0], positions[-1])  # Use first and last positions as fill values
        )

    def get_position(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """
        Gets the interpolated [x, y, z] position(s) at time(s) t.
        This now correctly handles both single floats and arrays of times.
        """
        return self.interp_func(t)


    def get_velocity(self, t: float, axis_idx: int = 0, dt: float = 1e-3) -> float:
        """Calculates the instantaneous velocity on a given axis at time t."""
        p1 = self.get_position(t - dt)
        p2 = self.get_position(t + dt)
        velocity = (p2[axis_idx] - p1[axis_idx]) / (2 * dt)
        return velocity

def process_and_save_end_pose_data(
        csv_path: str,
        robot_name: Literal['piper', 'k1'],
        output_dir: Optional[str] = None,
        csv_joint_names: Optional[List[str]] = None,
        save: bool = True
) -> List[PoseState]:
    """
    [Offline Function] Reads a raw CSV, processes joint data, calculates
    end-effector forward kinematics, and saves the resulting pose data.
    """
    logger.info(f"Starting offline processing for robot '{robot_name}' using data: {csv_path}")

    # --- Robot Model Initialization ---
    try:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        PROJECT_ROOT = os.getcwd()

    URDF_PATHS = {
        'piper': osp.join(PROJECT_ROOT, "assets","Urdf", "piper_description", "piper_with_gripper.urdf"),
        'k1': osp.join(PROJECT_ROOT, "assets", "Urdf","k1_description", "k1.urdf")  # Placeholder path
    }
    EE_LINK_NAMES = {
        'piper': 'link7',  # As determined from previous debugging
        'k1': 'link6'  # Placeholder name
    }

    if robot_name not in URDF_PATHS:
        raise ValueError(f"Robot name '{robot_name}' is not recognized. Available options: {list(URDF_PATHS.keys())}")

    urdf_path = URDF_PATHS[robot_name]
    ee_link_name = EE_LINK_NAMES[robot_name]

    try:
        robot_model = PiperRobot(urdf_path, ee_link_name=ee_link_name)
    except Exception as e:
        logger.error(f"Failed to initialize robot model for '{robot_name}': {e}")
        return []

    # --- Core processing logic ---
    try:
        df = pd.read_csv(csv_path)
        df['position'] = df['position'].apply(ast.literal_eval)
        df['name'] = df['name'].apply(ast.literal_eval)
        df['time'] = df['header.stamp.secs'] + df['header.stamp.nsecs'] * 1e-9
        temp_data = [{'time': r['time'], 'joints': dict(zip(r['name'], r['position']))} for _, r in df.iterrows()]


        # --- FK Calculation and Data Combination ---
        processed_data: List[PoseState] = []
        for entry in temp_data:
            # Get joint positions for FK calculation
            position_list = [entry['joints'].get(name, 0.0) for name in csv_joint_names]

            # Calculate End-Effector Pose
            end_position, end_orientation = robot_model.get_fk_solution(position_list + [0.0])  # Add a placeholder for gripper joint

            # Create PoseState (with placeholder orientation)
            pose_state: PoseState = {
                'time': entry['time'],
                'position': end_position.tolist(),
                'orientation': end_orientation.tolist()  # Placeholder: w,x,y,z
            }

            processed_data.append(pose_state)

    except Exception as e:
        logger.error(f"Failed during core processing of {csv_path}: {e}")
        raise

    # --- Optional Saving Logic ---
    if save:
        if output_dir is None: output_dir = os.path.dirname(csv_path)
        base_name = os.path.basename(csv_path).replace('_joint_states.csv', '_end_pose_'+ robot_name)  # Changed suffix

        output_path_pkl = os.path.join(output_dir, f"{base_name}.pkl")
        output_path_csv = os.path.join(output_dir, f"{base_name}.csv")

        logger.info(f"Saving {len(processed_data)} pose states to PKL: {output_path_pkl}")
        with open(output_path_pkl, 'wb') as f:
            pickle.dump(processed_data, f)

        logger.info(f"Saving {len(processed_data)} pose states to CSV: {output_path_csv}")
        flat_data_for_csv = []
        for item in processed_data:
            row = {
                'time': item['time'],
                'pos_x': item['position'][0],
                'pos_y': item['position'][1],
                'pos_z': item['position'][2],
                'orn_w': item['orientation'][0],
                'orn_x': item['orientation'][1],
                'orn_y': item['orientation'][2],
                'orn_z': item['orientation'][3]
            }
            flat_data_for_csv.append(row)

        df_to_save = pd.DataFrame(flat_data_for_csv)
        df_to_save.to_csv(output_path_csv, index=False)

    logger.success(f"Successfully processed pose data with FK for {csv_path}.")
    return processed_data


def process_and_save_joint_data(
        csv_path: str,
        output_dir: Optional[str] = None,
        csv_joint_names: Optional[List[str]] = None,
        csv_gripper_name: str = 'gripper',
        enable_gripper_calibrate: bool = True,
        mujoco_gripper_max_opening: float = 0.035,
        save: bool = True
) -> List[JointState]:
    """
    [Offline Function] Reads and processes a raw CSV file.

    It can save the complete, untruncated result as both a .pkl file (for fast
    Python loading) and a clean .csv file (for inspection). It always returns
    the processed data.

    Args:
        csv_path (str): Path to the input raw CSV file.
        output_dir (Optional[str], optional): Directory to save output files. Defaults to the same directory as the input CSV.
        save (bool, optional): If True, saves the processed data to files. Defaults to True.
        ... (other processing parameters) ...

    Returns:
        List[JointState]: A list containing the full, processed joint state data.
    """
    logger.info(f"Starting offline processing for: {csv_path}")

    if csv_joint_names is None:
        csv_joint_names = [f'joint{i}' for i in range(1, 7)]

    # --- Core processing logic ---
    try:
        df = pd.read_csv(csv_path)
        df['position'] = df['position'].apply(ast.literal_eval)
        df['name'] = df['name'].apply(ast.literal_eval)
        df['time'] = df['header.stamp.secs'] + df['header.stamp.nsecs'] * 1e-9
        temp_data = [{'time': r['time'], 'joints': dict(zip(r['name'], r['position']))} for _, r in df.iterrows()]

        min_grip, max_grip = DEFAULT_MIN_RAW_GRIPPER_DATA, DEFAULT_MAX_RAW_GRIPPER_DATA
        all_gripper_values = [d['joints'].get(csv_gripper_name, 0.0) for d in temp_data]
        if enable_gripper_calibrate and all_gripper_values:
            min_grip_data, max_grip_data = min(all_gripper_values), max(all_gripper_values)
            # protect the gripper calibrate by wrong data
            if min_grip_data < DEFAULT_MIN_RAW_GRIPPER_DATA:
                min_grip = min_grip_data
                logger.info(f"Gripper min value set to {min_grip_data:.3f} from csv")
            if max_grip_data > DEFAULT_MAX_RAW_GRIPPER_DATA:
                max_grip = max_grip_data
                logger.info(f"Gripper max value set to {max_grip_data:.3f} from csv")
            min_grip = min(min_grip, min_grip_data)
            max_grip = max(max_grip, max_grip_data)
        range_raw = max_grip - min_grip
        scale_factor = mujoco_gripper_max_opening / range_raw if range_raw != 0 else 0

        processed_data: List[JointState] = []
        for entry in temp_data:
            position_list = [entry['joints'].get(name, 0.0) for name in csv_joint_names]
            raw_gripper_val = entry['joints'].get(csv_gripper_name, min_grip)
            calibrated_gripper_val = (raw_gripper_val - min_grip) * scale_factor
            position_list.append(calibrated_gripper_val)
            processed_data.append({'time': entry['time'], 'positions': position_list})
    except Exception as e:
        logger.error(f"Failed during core processing of {csv_path}: {e}")
        raise

    # --- Optional Saving Logic ---
    if save:
        if output_dir is None:
            output_dir = os.path.dirname(csv_path)
        base_name = os.path.basename(csv_path).replace('.csv', '_processed')

        output_path_pkl = os.path.join(output_dir, f"{base_name}.pkl")
        output_path_csv = os.path.join(output_dir, f"{base_name}.csv")

        # Save as a .pkl file
        logger.info(f"Saving {len(processed_data)} processed states to PKL: {output_path_pkl}")
        with open(output_path_pkl, 'wb') as f:
            pickle.dump(processed_data, f)

        # Save as a clean .csv file
        logger.info(f"Saving {len(processed_data)} processed states to CSV: {output_path_csv}")
        # Flatten data for DataFrame
        flat_data_for_csv = []
        for state in processed_data:
            row = {'time': state['time']}
            for i, pos in enumerate(state['positions']):
                # Name columns pos_1, pos_2, ..., pos_gripper
                col_name = f"pos_{i + 1}" if i < 6 else "pos_gripper"
                row[col_name] = pos
            flat_data_for_csv.append(row)
        df_to_save = pd.DataFrame(flat_data_for_csv)
        df_to_save.to_csv(output_path_csv, index=False)

    logger.success(f"Successfully processed data for {csv_path}.")
    return processed_data


def process_and_save_combined_data(
        csv_path: str,
        robot_name: Literal['piper', 'k1'],
        arm_key: Optional[Literal['left', 'right']] = None,
        output_dir: Optional[str] = None,
        csv_joint_names: Optional[List[str]] = None,
        csv_gripper_name: str = 'gripper',
        enable_gripper_calibrate: bool = True, # only use in piper
        jump_threshold: Optional[float] = 0.15, # ADDED: Max allowed jump in joint values (radians)
        save: bool = True
) -> dict:
    """
    [Core Function] Reads a raw CSV, processes both joint and end-effector pose data,
    and saves them into a single "wide" CSV file and a backup PKL file.
    Piper robot 6+1, K1 robot 7+1
    """
    logger.info(f"Processing position-based data for robot '{robot_name}' on: {os.path.basename(csv_path)}")

    # --- 1. Initialization and Data Loading ---
    try:
        df = pd.read_csv(csv_path)
        df['position'] = df['position'].apply(ast.literal_eval)
        df['time'] = df['header.stamp.secs'] + df['header.stamp.nsecs'] * 1e-9

        if robot_name == 'piper':
            if csv_joint_names is None:
                csv_joint_names = [f'joint{i}' for i in range(1, 7)]
            df['name'] = df['name'].apply(ast.literal_eval)
            temp_data = [{'time': r['time'], 'joints': dict(zip(r['name'], r['position']))} for _, r in df.iterrows()]
        elif robot_name == 'k1':
            temp_data = [{'time': r['time'], 'positions': r['position']} for _, r in df.iterrows()]

    except Exception as e:
        logger.error(f"Failed during initial data loading of {csv_path}: {e}")
        raise

    # --- 2. Joint Data Processing ---

    all_potential_points: List[JointState] = []
    last_valid_positions: Optional[List[float]] = None

    if robot_name =="piper":
        all_gripper_values = [d['joints'].get(csv_gripper_name, 0.0) for d in temp_data]
        min_grip = DEFAULT_MIN_RAW_GRIPPER_DATA
        max_grip = DEFAULT_MAX_RAW_GRIPPER_DATA
        print(min(all_gripper_values), max(all_gripper_values))
        if enable_gripper_calibrate and all_gripper_values:
            min_grip = min(all_gripper_values) if min(all_gripper_values) < DEFAULT_MIN_RAW_GRIPPER_DATA else DEFAULT_MIN_RAW_GRIPPER_DATA
            max_grip = max(all_gripper_values) if max(all_gripper_values) > DEFAULT_MAX_RAW_GRIPPER_DATA else DEFAULT_MAX_RAW_GRIPPER_DATA
        print(f"Gripper calibration: min {min_grip:.4f}, max {max_grip:.4f}")

        range_raw = max_grip - min_grip
        scale_factor = DEFAULT_MUJOCO_PIPER_GRIPPER_MAX_OPENING / range_raw if range_raw != 0 else 0
        for entry in temp_data:
            position_list = [entry['joints'].get(name, 0.0) for name in csv_joint_names]
            raw_gripper_val = entry['joints'].get(csv_gripper_name, min_grip)
            calibrated_gripper_val = (raw_gripper_val - min_grip) * scale_factor
            final_gripper_val = max(0.0, min(calibrated_gripper_val, DEFAULT_MUJOCO_PIPER_GRIPPER_MAX_OPENING))
            position_list.append(final_gripper_val)

            all_potential_points.append({'time': entry['time'], 'positions': position_list})

    else: # k1
        for entry in temp_data:
            full_positions = entry['positions']
            final_positions_to_process = list(full_positions[:8])
            final_positions_to_process[-1] = (1- final_positions_to_process[-1]) * DEFAULT_MUJOCO_K1_GRIPPER_MAX_OPENING
            all_potential_points.append({'time': entry['time'], 'positions': final_positions_to_process})

    # ---3: Spike Filtering using a 3-point window# Stage 2: Spike Filtering using a 3-point window
    processed_joint_data: List[JointState] = []
    if len(all_potential_points) < 3:
        # Not enough data for spike detection, accept all
        logger.warning("Data has fewer than 3 points, skipping spike filter.")
        processed_joint_data = all_potential_points
    else:
        # Always accept the first point as the anchor
        processed_joint_data.append(all_potential_points[0])

    # Iterate through the middle points where we have a 'before' and 'after'
    for i in range(1, len(all_potential_points) - 1):
        # We compare to the last *accepted* point, not necessarily the previous point in the original list
        prev_positions = processed_joint_data[-1]['positions']
        curr_positions = all_potential_points[i]['positions']
        next_positions = all_potential_points[i + 1]['positions']

        # Calculate jumps relative to neighbors
        jump_from_prev = max(abs(c - p) for c, p in zip(curr_positions, prev_positions))
        jump_to_next = max(abs(n - c) for n, c in zip(next_positions, curr_positions))

        # This checks if the neighbors themselves are close, indicating the current point is a spike
        span_of_neighbors = max(abs(n - p) for n, p in zip(next_positions, prev_positions))

        # Define the spike condition: jump is high from both sides, but neighbors are close to each other.
        is_a_spike = (jump_from_prev > jump_threshold) and \
                     (jump_to_next > jump_threshold) and \
                     (span_of_neighbors < jump_threshold)

        if is_a_spike:
            logger.warning(
                f"Spike detected and filtered at time {all_potential_points[i]['time']:.4f}. "
                f"Jump from prev: {jump_from_prev:.3f}, Jump to next: {jump_to_next:.3f}"
            )
            # If it's a spike, we simply skip it and do not add it to processed_joint_data
            continue

        # If it's not a spike, it might still be a large step change worth warning about.
        # This is your original warning logic for "single direction jumps"
        if jump_from_prev > jump_threshold:
            logger.warning(
                f"Note: Large step change at time {all_potential_points[i]['time']:.4f}. "
                f"Difference to last accepted point: {jump_from_prev:.4f}"
            )

        # The point is valid (not a spike), so we accept it.
        processed_joint_data.append(all_potential_points[i])


    # --- 4. End-Effector Pose Processing (FK) ---
    processed_pose_data: List[PoseState] = []
    try:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        PROJECT_ROOT = os.getcwd()

    if robot_name == 'piper':
        urdf_path = os.path.join(PROJECT_ROOT, "assets", "piper_description", "piper_with_gripper.urdf")
        robot_model = PiperRobot(urdf_path, ee_link_name='link7')
    else:  # k1
        urdf_path = os.path.join(PROJECT_ROOT, "assets", "k1_description", "k1_pgc_fix_v1.urdf")
        robot_model = K1DualArmRobot(urdf_path)

    for joint_data in processed_joint_data:
        fk_input_joints = joint_data['positions']
        if robot_name == 'k1':
            # K1 has 8 joints, so we need to ensure we only use the first 8
            pos, orn = robot_model.get_fk_solution(fk_input_joints[:7], arm=arm_key)
        else:# piper
            arm_joints_for_fk = fk_input_joints[:6]
            pos, orn = robot_model.get_fk_solution(arm_joints_for_fk + [0.0])
        processed_pose_data.append(
            {'time': joint_data['time'], 'position': pos.tolist(), 'orientation': orn.tolist()})


    # --- 4. Combine and Save ---
    if save:
        if output_dir is None: output_dir = os.path.dirname(csv_path)
        base_name = os.path.basename(csv_path).replace('.csv', '_and_end_pose')

        # Convert both data types to DataFrames to merge them
        num_joints = len(processed_joint_data[0]['positions'])
        joint_cols_map = {}
        if robot_name == 'piper':
            for i in range(num_joints - 1):
                joint_cols_map[i] = f'pos_{i + 1}'
            joint_cols_map[num_joints - 1] = 'pos_gripper'
        else:
            for i in range(num_joints):
                joint_cols_map[i] = f'pos_{i + 1}'

        df_joint = pd.DataFrame([
            {'time': s['time'], **{joint_cols_map[i]: p for i, p in enumerate(s['positions'])}}
            for s in processed_joint_data
        ])
        df_pose = pd.DataFrame([{'time': s['time'], 'pos_x': s['position'][0], 'pos_y': s['position'][1],
                                 'pos_z': s['position'][2], 'orn_w': s['orientation'][0], 'orn_x': s['orientation'][1],
                                 'orn_y': s['orientation'][2], 'orn_z': s['orientation'][3]} for s in
                                processed_pose_data])

        df_combined = pd.merge(df_joint, df_pose, on='time', how='inner')

        output_path_csv = os.path.join(output_dir, f"{base_name}.csv")
        logger.info(f"Saving combined data to WIDE CSV: {output_path_csv}")
        df_combined.to_csv(output_path_csv, index=False)

        output_path_pkl = os.path.join(output_dir, f"{base_name}.pkl")
        with open(output_path_pkl, 'wb') as f:
            pickle.dump({'joint': processed_joint_data, 'pose': processed_pose_data}, f)

    logger.success(f"Successfully processed data for {os.path.basename(csv_path)}.")
    return {'joint': processed_joint_data, 'pose': processed_pose_data}

def load_processed_data(
    csv_path: str,
    mode: Literal['joint', 'pose'] = 'joint',
    start_time_offset: float = 0.0,
    base_translation: Optional[List[float]] = None,
    base_rotation_quat: Optional[List[float]] = None
) -> Union[List[JointState], List[PoseState]]:
    """
    [Online Function] Loads a pre-processed, clean .csv file, converts it
    back to the standard JointState or Position format, and then truncates the data
    in memory based on the start_time_offset.

    Args:
        csv_path (str): The processed CSV file path.
        start_time_offset (float, optional): The runtime start time offset in seconds. Defaults to 0.0.
        mode: Literal['joint', 'pose']: The mode to load data in, either 'joints' or 'poses'.
        base_translation (Optional[List[float]], optional): The [x, y, z] translation of the
            data's base frame relative to the world frame. Defaults to None (no translation).
        base_rotation_quat (Optional[List[float]], optional): The [x, y, z, w] quaternion of the
            data's base frame relative to the world frame. Defaults to None (no rotation).
    Returns:
        Union[List[JointState], List[PoseState]]: The truncated list of data.
    """

    if not os.path.exists(csv_path):
        logger.error(f"Processed CSV file not found: {csv_path}")
        raise FileNotFoundError(f"Processed CSV file not found: {csv_path}")

    logger.info(f"Loading '{os.path.basename(csv_path)}' in '{mode}' mode.")
    df = pd.read_csv(csv_path)
    reconstructed_data = []

    if mode == 'joint':
        position_cols = sorted(
            [col for col in df.columns if col.startswith('pos_') and col not in ['pos_x', 'pos_y', 'pos_z']],
            key=lambda x: (x.replace('pos_gripper', 'pos_7')))
        if not position_cols:
            raise ValueError(f"Mode is 'joint' but no joint columns (e.g., 'pos_1') found in {csv_path}.")
        for _, record in df.iterrows():
            positions = [record[col] for col in position_cols]
            reconstructed_data.append({'time': record['time'], 'positions': positions})

    elif mode == 'pose':
        pose_cols = ['pos_x', 'pos_y', 'pos_z', 'orn_w', 'orn_x', 'orn_y', 'orn_z']
        if not all(col in df.columns for col in pose_cols):
            raise ValueError(f"Mode is 'pose' but some pose columns are missing in {csv_path}.")

        # If an argument is not provided, use the identity value
        if base_translation is None:
            base_translation = [0.0, 0.0, 0.0]
        if base_rotation_quat is None:
            base_rotation_quat = [0.0, 0.0, 0.0, 1.0]  # Identity quaternion [x, y, z, w]

        t_base_to_world = np.array(base_translation)
        R_base_to_world = R.from_quat(base_rotation_quat)

        if not (np.all(t_base_to_world == 0) and np.all(base_rotation_quat == [0, 0, 0, 1])):
            logger.info(f"Applying base frame transformation to all loaded poses.translation: {t_base_to_world}, "
                        f"rotation (quat): {base_rotation_quat}")

        for _, record in df.iterrows():
            # Load original pose from file
            local_pos = np.array([record['pos_x'], record['pos_y'], record['pos_z']])
            local_orn_quat = [record['orn_x'], record['orn_y'], record['orn_z'],
                              record['orn_w']]  # Scipy expects [x, y, z, w]

            # Always apply the transform. If no args were given, this is an identity transform.
            world_pos = R_base_to_world.apply(local_pos) + t_base_to_world
            R_local = R.from_quat(local_orn_quat)
            R_world = R_base_to_world * R_local
            world_orn_quat = R_world.as_quat()

            # Store the transformed values, converting back to standard list formats
            position = world_pos.tolist()
            orientation = [world_orn_quat[3], world_orn_quat[0], world_orn_quat[1],
                           world_orn_quat[2]]  # Convert back to [w, x, y, z]

            reconstructed_data.append({'time': record['time'], 'position': position, 'orientation': orientation})

    else:
        raise ValueError(f"Invalid load mode: '{mode}'. Choose 'joint' or 'pose'.")

    logger.info(f"Loaded and reconstructed {len(reconstructed_data)} states.")

    if not reconstructed_data or start_time_offset <= 0:
        return reconstructed_data

    playback_start_time = reconstructed_data[0]['time'] + start_time_offset
    start_index = next((i for i, state in enumerate(reconstructed_data) if state['time'] >= playback_start_time), -1)

    if start_index == -1:
        return []

    return reconstructed_data[start_index:]

if __name__ == "__main__":
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    DEFAULT_SAMPLE_ROOT = os.path.join(REPO_ROOT, "data", "sample")
    DATA_ROOT = os.environ.get("RGBENCH_DATA_ROOT", DEFAULT_SAMPLE_ROOT)

    # k1 data
    k1_csv_data_path = os.environ.get("RGBENCH_K1_DATA", os.path.join(DATA_ROOT, "K1_Data"))
    left_k1_csv_path = osp.join(k1_csv_data_path, "joints","left_arm_joint_states.csv")
    right_k1_csv_path = osp.join(k1_csv_data_path, "joints", "right_arm_joint_states.csv")

    piper_csv_data_path = os.environ.get("RGBENCH_PIPER_DATA", os.path.join(DATA_ROOT, "Piper_Data"))
    left_piper_csv_path = osp.join(piper_csv_data_path, "joints", "left_arm_joint_states.csv")
    right_piper_csv_path = osp.join(piper_csv_data_path, "joints", "right_arm_joint_states.csv")

    left_combined_data = process_and_save_combined_data(left_piper_csv_path,
                                                        robot_name='piper', arm_key='left', save=False,
                                                        enable_gripper_calibrate=False)
    right_combined_data = process_and_save_combined_data(right_piper_csv_path,
                                                         robot_name='piper', arm_key='right', save=False,
                                                         enable_gripper_calibrate=True)

    print("")
