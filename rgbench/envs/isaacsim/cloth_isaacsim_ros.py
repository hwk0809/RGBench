from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import sys
import os
import time
import threading
import numpy as np
from typing import Dict, List, Optional
from pynput import keyboard

# ROS imports
try:
    import rospy
    from sensor_msgs.msg import JointState
except ImportError:
    print("Error: 'rospy' or 'sensor_msgs' not found.")
    simulation_app.close()
    exit(1)

# Isaac Sim imports
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat

# Custom imports
sys.path.append(os.getcwd())
from rgbench.envs.isaacsim.Env_Config.BaseEnv import BaseEnv
from rgbench.envs.isaacsim.Env_Config.Garment.Particle_Garment import Particle_Garment
from rgbench.envs.isaacsim.Env_Config.Room.Real_Ground import Real_Ground
from rgbench.envs.isaacsim.Env_Config.Camera.Recording_Camera import Recording_Camera
from isaacsim.core.utils.viewports import set_camera_view
from omegaconf import DictConfig, OmegaConf


class JointStateData:
    def __init__(self, time: float, positions: List[float]):
        self.time = time
        self.positions = positions


class IsaacSimRosController(BaseEnv):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.lock = threading.Lock()
        
        # Initialize BaseEnv
        physics_timestep = cfg.env.get('physics_timestep', 1/500)
        rendering_timestep = cfg.env.get('rendering_timestep', 1/500)
        super().__init__(physics_timestep, rendering_timestep)
        
        # ===== Setup Environment =====
        self._setup_environment()
        self._setup_robots()
        self._setup_camera()
        
        # Initialize world
        self.reset()
        self._configure_robots()
        
        # ===== ROS State Variables =====
        self.run_mode = "realtime"  # 'realtime' or 'physical'
        self.gripper_factor = 0.35 / 0.688
        self.dof = 6
        
        # Latest positions
        self.latest_left_positions: List[float] = [0.0] * (self.dof + 1)
        self.latest_right_positions: List[float] = [0.0] * (self.dof + 1)
        
        # Data buffers for physical mode
        self.left_data_buffer: List[JointStateData] = []
        self.right_data_buffer: List[JointStateData] = []
        
        # Interpolation trackers
        self.time_trackers_idx: Dict[str, int] = {'left': 0, 'right': 0}
        self.master_time_offset: float = 0.0
        self.latest_msg_time: float = 0.0
        
        # Initialize default joint states
        self._init_default_positions()
        
        print("=" * 50)
        print("IsaacSim-ROS Controller Initialized")
        print(f"  Mode: {self.run_mode}")
        print("  Press 'SPACE' to toggle physical/realtime mode")
        print("  Press 'q' to quit")
        print("=" * 50)

    def _setup_environment(self):
        """Setup ground and cloth"""
        self.ground = Real_Ground(self.scene, visual_material_usd=None)
        
        cloth_usd_path = self.cfg.cloth_params.cloth_model_usda
        visual_material_usd = "assets/Assets/Material/Garment/linen_White.usd"
        
        self.garment = Particle_Garment(
            self.world,
            pos=np.array([0.1, 0, 0.01]),
            ori=np.array([0.0, 0.0, 0.0]),
            scale=np.array([1.0, 1.0, 1.0]),
            usd_path=cloth_usd_path,
            visual_material_usd=visual_material_usd,
            friction=100,
        )

    def _setup_robots(self):
        """Setup dual Piper robots"""
        left_arm_path = self.cfg.env.get('piper_usda', 
            "/home/sivan/cloth/piper_usd/piper_urdf.usda")
        right_arm_path = left_arm_path
        
        # Left arm
        add_reference_to_stage(usd_path=left_arm_path, prim_path="/World/PiperLeft")
        self.left_arm = Robot(
            prim_path="/World/PiperLeft",
            name="left_piper_arm",
            translation=np.array([0.5, 0, 0.05])
        )
        self.world.scene.add(self.left_arm)
        
        # Right arm
        add_reference_to_stage(usd_path=right_arm_path, prim_path="/World/PiperRight")
        self.right_arm = Robot(
            prim_path="/World/PiperRight",
            name="right_piper_arm",
            translation=np.array([-0.5, 0, 0.05])
        )
        self.world.scene.add(self.right_arm)

    def _setup_camera(self):
        """Setup recording camera"""
        self.env_camera = Recording_Camera(
            camera_position=np.array([2, 0, 1.5]),
            prim_path="/World/env_camera",
        )
        set_camera_view(
            eye=[4, 0, 3],
            target=[0.1, 0.0, 0.0],
            camera_prim_path="/World/env_camera",
        )

    def _configure_robots(self):
        """Configure robot initial poses and joint indices"""
        # Set base poses
        for arm, pos in [(self.left_arm, [0, -0.25, 0.0]), 
                         (self.right_arm, [0, 0.25, 0.0])]:
            ori = np.array([0.0, 0.0, 0.0])
            quat = euler_angles_to_quat(ori, degrees=True)
            arm.set_world_pose(position=np.array(pos), orientation=quat)
            arm.set_default_state(position=np.array(pos), orientation=quat)
            arm.set_joints_default_state(np.zeros(8))
            arm.post_reset()
        
        # Define joint names and indices
        joint_names = [f"joint{i}" for i in range(1, 9)]
        for arm in [self.left_arm, self.right_arm]:
            arm.arm_dof_names = joint_names
            arm.arm_dof_indices = [arm.get_dof_index(name) for name in joint_names]
        
        # Warm up simulation
        for _ in range(100):
            self.world.step()
            simulation_app.update()

    def _init_default_positions(self):
        """Initialize default joint positions from robot state"""
        for i in range(self.dof):
            self.latest_left_positions[i] = 0.0
            self.latest_right_positions[i] = 0.0

    # ===== ROS Callbacks =====
    def left_arm_callback(self, msg: JointState):
        """Left arm joint state callback"""
        positions = list(msg.position)[:self.dof + 1]
        if len(positions) < self.dof + 1:
            return
        
        msg_time = msg.header.stamp.to_sec()
        self.latest_msg_time = msg_time
        
        with self.lock:
            self.latest_left_positions = positions
            self.left_data_buffer.append(JointStateData(msg_time, positions))

    def right_arm_callback(self, msg: JointState):
        """Right arm joint state callback"""
        positions = list(msg.position)[:self.dof + 1]
        if len(positions) < self.dof + 1:
            return
        
        msg_time = msg.header.stamp.to_sec()
        self.latest_msg_time = msg_time
        
        with self.lock:
            self.latest_right_positions = positions
            self.right_data_buffer.append(JointStateData(msg_time, positions))

    # ===== Mode Control =====
    def toggle_mode(self):
        """Toggle between realtime and physical mode"""
        with self.lock:
            current_ros_time = rospy.get_time()
            if self.run_mode == "realtime":
                self.run_mode = "physical"
                self.master_time_offset = self.latest_msg_time - self.world.current_time
                self._reset_trackers_to_time(current_ros_time)
                print("\n--- Switched to [PHYSICAL MODE] ---")
            else:
                self.run_mode = "realtime"
                print("\n--- Switched to [REALTIME MODE] ---")

    def _reset_trackers_to_time(self, current_wall_time: float):
        """Reset interpolation trackers to current time"""
        for arm_name, buffer in [('left', self.left_data_buffer), 
                                  ('right', self.right_data_buffer)]:
            idx = 0
            while idx < len(buffer) - 2 and buffer[idx + 1].time < current_wall_time:
                idx += 1
            self.time_trackers_idx[arm_name] = idx

    # ===== Interpolation =====
    def _get_interpolated_positions(
        self,
        target_time: float,
        arm_data: List[JointStateData],
        current_idx: int
    ) -> tuple[Optional[List[float]], int]:
        """Get interpolated joint positions for target time"""
        if not arm_data:
            return None, current_idx
        
        # Advance index
        new_idx = current_idx
        while new_idx < len(arm_data) - 2 and arm_data[new_idx + 1].time < target_time:
            new_idx += 1
        
        # Boundary cases
        if target_time <= arm_data[0].time:
            return arm_data[0].positions, new_idx
        if target_time >= arm_data[-1].time:
            return arm_data[-1].positions, new_idx
        
        # Linear interpolation
        p0, p1 = arm_data[new_idx], arm_data[new_idx + 1]
        interval = p1.time - p0.time
        
        if interval <= 0:
            return p0.positions, new_idx
        
        alpha = (target_time - p0.time) / interval
        interpolated = np.array(p0.positions) + alpha * (np.array(p1.positions) - np.array(p0.positions))
        
        return interpolated.tolist(), new_idx

    def _apply_arm_state(self, arm: Robot, positions: List[float]):
        """Apply joint positions to robot arm"""
        joint_positions = positions.copy()
        # Add symmetric gripper joint
        joint_positions.append(joint_positions[-1] * -1)
        
        return ArticulationAction(
            joint_positions=np.array(joint_positions, dtype=np.float32),
            joint_indices=np.array(arm.arm_dof_indices)
        )

    # ===== Main Loop =====
    def start_ros_listener(self):
        """Start ROS subscriber threads"""
        rospy.init_node('isaacsim_ros_listener', anonymous=True)
        
        rospy.Subscriber('/left_arm/joint_states', JointState, self.left_arm_callback)
        rospy.Subscriber('/right_arm/joint_states', JointState, self.right_arm_callback)
        
        print("\n✓ ROS Listener Started")
        print("  Subscribed to:")
        print("    - /left_arm/joint_states")
        print("    - /right_arm/joint_states\n")
        
        ros_thread = threading.Thread(target=rospy.spin, daemon=True)
        ros_thread.start()

    def start_keyboard_listener(self):
        """Start keyboard listener for mode switching"""
        def on_press(key):
            try:
                if key.char == 'q':
                    print("[Keyboard] 'q' pressed. Quitting...")
                    simulation_app.close()
                    return False
            except AttributeError:
                if key == keyboard.Key.space:
                    print("[Keyboard] SPACE pressed. Toggling mode...")
                    self.toggle_mode()
        
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        print("✓ Keyboard listener started\n")

    def run_simulation(self):
        """Main simulation loop"""
        print("Starting IsaacSim simulation...\n")
        
        while simulation_app.is_running() and not rospy.is_shutdown():
            step_start_time = time.time()
            
            # ===== MODE 1: Realtime =====
            if self.run_mode == "realtime":
                with self.lock:
                    left_action = self._apply_arm_state(self.left_arm, self.latest_left_positions)
                    right_action = self._apply_arm_state(self.right_arm, self.latest_right_positions)
                
                self.left_arm.apply_action(left_action)
                self.right_arm.apply_action(right_action)
                
                self.world.step()
                simulation_app.update()
                
                # Realtime sync
                elapsed = time.time() - step_start_time
                sleep_time = self.world.get_physics_dt() - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # ===== MODE 2: Physical =====
            else:
                target_time = self.world.current_time + self.master_time_offset
                
                with self.lock:
                    # Interpolate left arm
                    left_pos, new_left_idx = self._get_interpolated_positions(
                        target_time,
                        self.left_data_buffer,
                        self.time_trackers_idx['left']
                    )
                    self.time_trackers_idx['left'] = new_left_idx
                    
                    # Interpolate right arm
                    right_pos, new_right_idx = self._get_interpolated_positions(
                        target_time,
                        self.right_data_buffer,
                        self.time_trackers_idx['right']
                    )
                    self.time_trackers_idx['right'] = new_right_idx
                    
                    # Apply positions
                    if left_pos:
                        self.left_arm.apply_action(self._apply_arm_state(self.left_arm, left_pos))
                    if right_pos:
                        self.right_arm.apply_action(self._apply_arm_state(self.right_arm, right_pos))
                
                self.world.step()
                simulation_app.update()


if __name__ == "__main__":
    try:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    except NameError:
        PROJECT_ROOT = os.getcwd()
    
    print(f"Project Root: {PROJECT_ROOT}\n")
    
    # Configuration
    config = {
        'cloth_params': {
            'cloth_model_usda': f'{os.environ["HOME"]}/cloth/DexGarmentLab/Assets/Garment/Tops/test/5k.usda',
        },
        'env': {
            'physics_timestep': 1/500,
            'rendering_timestep': 1/500,
            'piper_usda': "/home/sivan/cloth/piper_usd/piper_urdf.usda",
        }
    }
    
    cfg = OmegaConf.create(config)
    
    try:
        controller = IsaacSimRosController(cfg)
        controller.start_ros_listener()
        controller.start_keyboard_listener()
        controller.run_simulation()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()