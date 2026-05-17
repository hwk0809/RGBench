import roboticstoolbox as rtb
import numpy as np
import os
from spatialmath.base import r2q


class PiperRobot:
    def __init__(self, urdf_path, ee_link_name='link7'):
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found at: {urdf_path}")

        try:
            self.robot = rtb.ERobot.URDF(
                file_path=urdf_path,
                gripper=ee_link_name
            )
            print("Robot load Successfully.")

        except Exception as e:
            print(f"Error loading robot from URDF: {e}")
            print(f"Please ensure the URDF file is correctly formatted and the end effector link '{ee_link_name}' exists.")

            raise

    def get_fk_solution(self, joint_angles):
        """
        Calculate the forward kinematics solution for given joint angles.

        Args:
            joint_angles (list or np.ndarray): Joint angles matching the robot's degrees of freedom.

        Returns:
            np.ndarray: end effector position [x, y, z], Orientation in quaternion [w, x, y, z].
        """
        if len(joint_angles) != self.robot.n:
            raise ValueError(f"The number of joint angles provided ({len(joint_angles)}) does not match the robot DoF ({self.robot.n}).")

        all_poses = self.robot.fkine_all(joint_angles)
        fk_matrix = all_poses[-1]

        rotation_matrix = fk_matrix.R
        orientation_wxyz = r2q(rotation_matrix)

        return fk_matrix.t, orientation_wxyz


class K1DualArmRobot:
    def __init__(self, urdf_path):
        """
        Initialize the K1 dual-arm robot and keep an internal full joint state.
        Args:
            urdf_path (str): URDF file path.
        """
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        self.robot = rtb.ERobot.URDF(file_path=urdf_path)
        print(f"K1 Robot model loaded from: {urdf_path}, found {self.robot.n} DoFs.")
        # print(self.robot)

        # --- Hard-coded joint/link indices ---
        self.right_arm_indices = [0, 1, 2, 3, 4, 5, 6]
        self.right_gripper_indices = [7, 8]
        self.left_arm_indices = [9, 10, 11, 12, 13, 14, 15]
        self.left_gripper_indices = [16, 17]

        self.left_ee_link = "lt"
        self.right_ee_link = "rt"
        self.left_ee_link = "left_gripper_adapter"
        self.right_ee_link = "right_gripper_adapter"

        # --- Internal state management ---
        self.q_home = np.zeros(self.robot.n)
        self.q_current = self.q_home.copy()

    def update_full_state(self, q_full):
        if len(q_full) != self.robot.n:
            raise ValueError(f"Full state vector length ({len(q_full)}) does not match robot DoF ({self.robot.n}).")
        self.q_current = np.array(q_full)
        print("Robot full state updated.")

    def _format_pos_and_orientation(self, fk_matrix):
        """[Helper] Convert an SE3 matrix to a (position, quaternion) tuple."""
        position = fk_matrix.t
        orientation_wxyz = r2q(fk_matrix.R)
        return position, orientation_wxyz

    def get_fk_solution(self, joint_angles, arm='both'):
        """
        Minimal forward-kinematics interface.
        Args:
            joint_angles: joint angles.
                - arm='both': flat list of all 18 joint angles.
                - arm='left'/'right': list of 7 or 9 joint angles for the corresponding arm.
            arm (str): 'left', 'right', or 'both'.
        """
        # Cast to numpy array
        joint_angles = np.array(joint_angles)

        if arm == 'both':
            # Use the full joint vector as-is
            q_calc = joint_angles

            # Compute FK twice
            fk_L = self.robot.fkine(q_calc, end=self.left_ee_link)
            fk_R = self.robot.fkine(q_calc, end=self.right_ee_link)

            # Return result
            pos_L, ori_L = fk_L.t, r2q(fk_L.R)
            pos_R, ori_R = fk_R.t, r2q(fk_R.R)
            return {'left': (pos_L, ori_L), 'right': (pos_R, ori_R)}

        # For single-arm calculations, start from the current state
        q_calc = self.q_current.copy()

        if arm == 'left':
            # Update left-arm joints directly
            q_calc[self.left_arm_indices] = joint_angles[:7]
            if len(joint_angles) == 9:
                q_calc[self.left_gripper_indices] = joint_angles[7:]

            # Compute FK and return
            fk_L = self.robot.fkine(q_calc, end=self.left_ee_link)
            return fk_L.t, r2q(fk_L.R)

        elif arm == 'right':
            # Update right-arm joints directly
            q_calc[self.right_arm_indices] = joint_angles[:7]
            if len(joint_angles) == 9:
                q_calc[self.right_gripper_indices] = joint_angles[7:]

            # Compute FK and return
            fk_R = self.robot.fkine(q_calc, end=self.right_ee_link)
            return fk_R.t, r2q(fk_R.R)
        else:
            raise ValueError(f"Argument 'arm' must be 'left', 'right', or 'both'")

if __name__ == "__main__":
    try:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        PROJECT_ROOT = os.getcwd()

    JOINT_ANGLES_TO_TEST = [-0.45048693, 1.22996843, -0.28687977, -0.11801916, 0.53476888, 0.021485]
    URDF_FILE_PATH = os.path.join(PROJECT_ROOT,"assets","piper_description","piper_with_gripper.urdf")
    piper_robot = PiperRobot(URDF_FILE_PATH, ee_link_name="link7")
    t,q = piper_robot.get_fk_solution(JOINT_ANGLES_TO_TEST+ [0.0])
    print(f"\n[PiperRobot] Forward Kinematics for end effector:\n  -> Position: {t}, Orientation: {q}")

    # Position: [ 0.14182882 -0.08372661  0.00450508]

    q_full_robot = np.array([-1.768, -58.3,  -54.936,  -118.927,  -127.422,      42.904,  157.592,
                             0.0, 0.0,
                            1.792, -58.533,   54.750,  -119.007,  127.363,     -43.143, -157.583,
                            0.0, 0.0])*np.pi/180

    K1_urdf_path = os.path.join(PROJECT_ROOT, "assets", "Urdf","k1_description", "k1_pgc_fix_v1.urdf")
    k1_robot = K1DualArmRobot(K1_urdf_path)
    both_poses = k1_robot.get_fk_solution(q_full_robot)  # arm='both' is the default
    print("--- Both arms simultaneously ---")
    print(f"Left arm position: {both_poses['left'][0]}")
    print(f"Right arm position: {both_poses['right'][0]}")
    print("-" * 20)
    left_q = q_full_robot[k1_robot.left_arm_indices + k1_robot.left_gripper_indices]
    print("left_q", left_q)
    pos_L, ori_L = k1_robot.get_fk_solution(left_q, arm='left')
    print("--- Left arm only ---")
    print(f"Left arm position: {pos_L}")
    print(f"Left arm orientation: {ori_L}")
    print("-" * 20)

    right_q = q_full_robot[k1_robot.right_arm_indices + k1_robot.right_gripper_indices]
    print("right_q",right_q)
    pos_R, ori_R = k1_robot.get_fk_solution(right_q, arm='right')
    print("--- Right arm only ---")
    print(f"Right arm position: {pos_R}")
    print(f"Right arm orientation: {ori_R}")
    print("-" * 20)