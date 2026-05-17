#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from piper_sdk import C_PiperInterface
import numpy as np
from tf.transformations import quaternion_from_euler

class PiperArmNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('piper_arm_node', anonymous=True)

        # Initialize publishers for both arms
        self.left_joint_pub = rospy.Publisher('left_arm/joint_states', JointState, queue_size=10)
        self.right_joint_pub = rospy.Publisher('right_arm/joint_states', JointState, queue_size=10)
        self.left_pose_pub = rospy.Publisher('left_arm/end_pose', PoseStamped, queue_size=10)
        self.right_pose_pub = rospy.Publisher('right_arm/end_pose', PoseStamped, queue_size=10)
        self.left_end_rotation_pub = rospy.Publisher('left_arm/end_rotation_pose', PoseStamped, queue_size=10)
        self.right_end_rotation_pub = rospy.Publisher('right_arm/end_rotation_pose', PoseStamped, queue_size=10)
        
        # Connect hardware
        self.pi_per_left = C_PiperInterface("can0")
        self.pi_per_right = C_PiperInterface("can1")
        self.pi_per_left.ConnectPort()
        self.pi_per_right.ConnectPort()

        # Set loop rate (~200Hz)
        self.rate = rospy.Rate(200)
        self.joint_factor = round(1000*180/np.pi,8)
        self.gripper_factor = 1000*1000
        print("joint_factor", self.joint_factor)
        print("gripper_factor",self.gripper_factor)

        # Mutation detection
        self.last_left_joints = None
        self.last_right_joints = None
        # Joint angle change threshold (radians); tune as needed
        self.joint_change_threshold = 0.2  # ~28.6 degrees

    def convert_joint_state(self, raw_data, is_gripper=False):
        """Convert raw joint state data."""
        if is_gripper:
            return round(raw_data/ self.gripper_factor, 8)
        return round(raw_data / self.joint_factor, 8)

    def check_joint_mutation(self, current_joints, last_joints, arm_name):
        """
        Detect joint-angle mutations for a single arm.
        :param current_joints: joint angles for the current frame
        :param last_joints: joint angles for the previous frame
        :param arm_name: arm name ("left" or "right")
        """
        # Skip comparison on the first sample
        if last_joints is None:
            return

        for i in range(len(current_joints)):
            # Absolute difference between current and previous angle
            change = abs(current_joints[i] - last_joints[i])
            if change > self.joint_change_threshold:
                # Use rospy.logwarn so the warning follows ROS conventions
                rospy.logwarn(f"Sudden change detected on {arm_name} arm, joint {i + 1}!")
                rospy.logwarn(
                    f"  Previous: {last_joints[i]:.4f}, Current: {current_joints[i]:.4f}, Change: {change:.4f}")

    def publish_left_arm_data(self):
        # Read raw data
        joint_data = self.pi_per_left.GetArmJointMsgs()
        pose_data = self.pi_per_left.GetArmEndPoseMsgs()
        gripper_data = self.pi_per_left.GetArmGripperMsgs().gripper_state.grippers_angle

        # Build the JointState message
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']
        joint_msg.position = [
            self.convert_joint_state(joint_data.joint_state.joint_1),
            self.convert_joint_state(joint_data.joint_state.joint_2),
            self.convert_joint_state(joint_data.joint_state.joint_3),
            self.convert_joint_state(joint_data.joint_state.joint_4),
            self.convert_joint_state(joint_data.joint_state.joint_5),
            self.convert_joint_state(joint_data.joint_state.joint_6),
            self.convert_joint_state(gripper_data, is_gripper=True)
        ]

        # Build the PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = joint_msg.header.stamp
        pose_msg.header.frame_id = "base_link"
        pose_msg.pose.position.x = pose_data.end_pose.X_axis /1000.0/1000.0 # m
        pose_msg.pose.position.y = pose_data.end_pose.Y_axis /1000.0/1000.0
        pose_msg.pose.position.z = pose_data.end_pose.Z_axis /1000.0/1000.0
        
        # Convert Euler angles to quaternion (raw data assumed to be degrees)
        roll_rad = np.radians(pose_data.end_pose.RX_axis /1000.0)
        pitch_rad = np.radians(pose_data.end_pose.RY_axis /1000.0)
        yaw_rad = np.radians(pose_data.end_pose.RZ_axis /1000.0)
        
        quaternion = quaternion_from_euler(roll_rad, pitch_rad, yaw_rad, axes='rxyz')
        # quaternion = quaternion_from_euler(yaw_rad, pitch_rad, roll_rad, axes='szyx')
        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]

        

        end_rotation_msg = PoseStamped()
        end_rotation_msg.header.stamp = joint_msg.header.stamp
        end_rotation_msg.header.frame_id = "base_link"
        end_rotation_msg.pose.position.x = pose_data.end_pose.RX_axis
        end_rotation_msg.pose.position.y = pose_data.end_pose.RY_axis
        end_rotation_msg.pose.position.z = pose_data.end_pose.RZ_axis


        # Publish messages
        self.left_joint_pub.publish(joint_msg)
        self.left_pose_pub.publish(pose_msg)
        self.left_end_rotation_pub.publish(end_rotation_msg)

        # Detect joint mutations
        self.check_joint_mutation(joint_msg.position, self.last_left_joints, "left")
        self.last_left_joints = joint_msg.position

    def publish_right_arm_data(self):
        # Read raw data
        joint_data = self.pi_per_right.GetArmJointMsgs()
        pose_data = self.pi_per_left.GetArmEndPoseMsgs()
        gripper_data = self.pi_per_right.GetArmGripperMsgs().gripper_state.grippers_angle

        # Build the JointState message
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']
        joint_msg.position = [
            self.convert_joint_state(joint_data.joint_state.joint_1),
            self.convert_joint_state(joint_data.joint_state.joint_2),
            self.convert_joint_state(joint_data.joint_state.joint_3),
            self.convert_joint_state(joint_data.joint_state.joint_4),
            self.convert_joint_state(joint_data.joint_state.joint_5),
            self.convert_joint_state(joint_data.joint_state.joint_6),
            self.convert_joint_state(gripper_data, is_gripper=True)
        ]

        # Build the PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = joint_msg.header.stamp
        pose_msg.header.frame_id = "base_link"
        pose_msg.pose.position.x = pose_data.end_pose.X_axis /1000.0/1000.0 # m
        pose_msg.pose.position.y = pose_data.end_pose.Y_axis /1000.0/1000.0
        pose_msg.pose.position.z = pose_data.end_pose.Z_axis /1000.0/1000.0
        
        # Convert Euler angles to quaternion (raw data assumed to be degrees)
        roll_rad = np.radians(pose_data.end_pose.RX_axis /1000.0)
        pitch_rad = np.radians(pose_data.end_pose.RY_axis /1000.0)
        yaw_rad = np.radians(pose_data.end_pose.RZ_axis /1000.0)
        
        quaternion = quaternion_from_euler(roll_rad, pitch_rad, yaw_rad, axes='rxyz')
        # quaternion = quaternion_from_euler(yaw_rad, pitch_rad, roll_rad, axes='szyx')
        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]

        end_rotation_msg = PoseStamped()
        end_rotation_msg.header.stamp = joint_msg.header.stamp
        end_rotation_msg.header.frame_id = "base_link"
        end_rotation_msg.pose.position.x = pose_data.end_pose.RX_axis
        end_rotation_msg.pose.position.y = pose_data.end_pose.RY_axis
        end_rotation_msg.pose.position.z = pose_data.end_pose.RZ_axis

        # Publish messages
        self.right_joint_pub.publish(joint_msg)
        self.right_pose_pub.publish(pose_msg)
        self.right_end_rotation_pub.publish(end_rotation_msg)

        # Detect joint mutations
        self.check_joint_mutation(joint_msg.position, self.last_right_joints, "right")
        self.last_right_joints = joint_msg.position

    def run(self):
        while not rospy.is_shutdown():
            try:
                self.publish_left_arm_data()
                self.publish_right_arm_data()
                self.rate.sleep()
            except Exception as e:
                rospy.logerr(f"Error: {str(e)}")
                break

        # Close connection
        self.pi_per_left.DisconnectPort()
        self.pi_per_right.DisconnectPort()

if __name__ == '__main__':
    try:
        node = PiperArmNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
