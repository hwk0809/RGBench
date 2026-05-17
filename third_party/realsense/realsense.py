import os
import os.path as osp
import sys
from typing import Tuple, List
from datetime import datetime
from loguru import logger
import yaml, json
import pyrealsense2 as rs

import numpy as np
import open3d as o3d
import cv2 # mostly used for visualization or BGR <-> RGB conversion
import time
from sympy import false

sys.path.append(osp.join(osp.dirname(__file__), "..", '..'))

class RealSenseCamera:
    def __init__(self,camera_name="realsense_d455",
                 width=640, height=480, fps=30, pcd_trunc_dis = 3.0,
                 align_to_color=True, save_dir=None,
                 vis=False):
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.fps = fps
        self.vis = vis
        self.align_to_color = align_to_color

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure depth and color flow
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps) # bgr8 for OpenCV

        self.profile = None
        self.depth_scale = 0.0
        self.pcd_trunc_dis = pcd_trunc_dis
        self.align = None

        # Open3D align intrinsics
        self.o3d_intrinsics = None
        self.distortion_coeffs_cv = np.array([0, 0, 0, 0, 0])

        # Save properties
        current_dir = osp.dirname(osp.abspath(__file__))
        project_root = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_counter =0

        if save_dir:
            if not osp.isabs(save_dir):
                save_dir = osp.abspath(osp.join(current_dir, save_dir))
            self.save_dir = osp.join(save_dir, self.timestamp)
        else: # no input, default save dir
            self.save_dir = osp.join(project_root, "data",self.camera_name+ "_images", self.timestamp)


        print("--> RealSenseCamera initialized")

    def start(self):
        self.profile = self.pipeline.start(self.config)

        # Read the depth sensor's depth scale factor
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"--> Depth Scale: {self.depth_scale}")

        # Create alignment object
        if self.align_to_color:
            self.align = rs.align(rs.stream.color)
            aligned_stream_profile = self.profile.get_stream(rs.stream.color)
        else:
            self.align = rs.align(rs.stream.depth)
            aligned_stream_profile = self.profile.get_stream(rs.stream.depth)

        # Cache aligned-stream intrinsics for Open3D
        intr = aligned_stream_profile.as_video_stream_profile().get_intrinsics()
        self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
        )

        # Skip a few frames so auto-exposure stabilizes
        for _ in range(15):
            self.pipeline.wait_for_frames()
        print("--> RealSense camera started and auto-exposure settled.")

    def stop(self):
        if self.profile: # ensure the camera was started
            self.pipeline.stop()
            self.profile = None # reset profile
            print("--> RealSense camera stopped.")

    def _get_aligned_frames(self):
        frames = self.pipeline.wait_for_frames()
        if self.align:
            aligned_frames = self.align.process(frames)
        else:
            aligned_frames = frames # used when no align object was created (e.g., align_to_color=False)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("Could not acquire depth or color frames.")
        return depth_frame, color_frame

    def capture_rgb(self, save=False, prefix=None) -> np.ndarray:
        """Capture an RGB image (BGR format)."""
        _, color_frame = self._get_aligned_frames()
        color_image = np.asanyarray(color_frame.get_data())
        if self.vis:
            cv2.imshow('RealSense RGB', color_image)
            cv2.waitKey(1000)
        if save:
            self.save_data(color_image, "rgb", prefix)
        return color_image

    def capture_pcd(self, save=False, prefix=None) -> o3d.geometry.PointCloud:
        """
        Capture a point cloud expressed in the coordinate frame of self.o3d_intrinsics
        (the alignment target stream). Colors come from the aligned color image.
        """
        depth_frame, color_frame = self._get_aligned_frames()

        depth_image_o3d = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
        # RealSense outputs BGR; Open3D typically expects RGB for point cloud colors
        color_image_bgr = np.asanyarray(color_frame.get_data())
        color_image_rgb_o3d = o3d.geometry.Image(cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB))

        # Build the RGBD image
        # Note: Open3D's depth_scale multiplies raw depth to obtain meters
        # RealSense's depth_scale also multiplies raw depth to obtain meters
        # We use 1.0 here because the depth frame is already Z16 (units = depth_scale * millimeters)
        # Open3D's create_from_rgbd_image uses depth_scale such that meters = raw_depth * depth_scale
        # depth_frame.get_data() from RealSense returns scaled values (use self.depth_scale otherwise)
        # Typical Z16 depth values are in millimeters; divide by 1000.0 (or multiply by self.depth_scale if it is 0.001)
        # For clarity, use depth_scale=1.0/self.depth_scale when the depth frame is unscaled
        # When pulled from get_depth_frame() the values are already in millimeters, so use 1000.0
        # The safest approach is to verify depth_frame.get_units() or rely on self.depth_scale directly
        # In o3d.geometry.RGBDImage.create_from_color_and_depth, depth_scale satisfies raw_depth_value * depth_scale = meters
        # RealSense depth_frame.get_data() returns D where D * self.depth_scale = meters
        # Therefore o3d_depth_scale = self.depth_scale

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image_rgb_o3d,
            depth_image_o3d,
            depth_scale=1.0 / self.depth_scale, # correct: Open3D's scale converts raw depth to meters
                                                # while RealSense's depth_scale is the depth unit (e.g., 0.001 for mm)
                                                # So for a depth pixel D, D * depth_scale_realsense = meters
                                                # Open3D expects raw_depth_o3d * depth_scale_o3d = meters
                                                # When raw_depth_o3d == D, depth_scale_o3d = depth_scale_realsense
                                                # Verified: o3d's depth_scale converts depth pixel values to meters
                                                # RealSense depth_frame values are already scaled to millimeters
                                                # So if depth_image_o3d values are in millimeters, depth_scale should be 1000.0
                                                # Or more precisely, depth_scale=1.0 and depth_trunc in meters
                                                # In practice, o3d.geometry.PointCloud.create_from_rgbd_image's depth_scale is 1.0/depth_camera_scale_factor
                                                # And self.depth_scale is depth_camera_scale_factor, so 1.0 / self.depth_scale is correct
            depth_trunc=self.pcd_trunc_dis,  # truncation distance in meters (drop points beyond ~3m)
            convert_rgb_to_intensity=False
        )


        # Create the point cloud from the RGBD image and intrinsics
        # Intrinsics belong to the aligned target stream (color in this case, since align_to_color=True)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            self.o3d_intrinsics # use the aligned stream intrinsics cached earlier
        )

        # RealSense's coordinate frame is typically Z forward, Y down, X right
        # Many robotics and graphics applications expect Y up with Z in or out
        # Apply a transform here or downstream if needed
        # E.g., flip Y and Z so Y is up and Z is forward (when original Z is forward):
        # pcd.transform([[1, 0, 0, 0],
        #                [0, -1, 0, 0],
        #                [0, 0, -1, 0],
        #                [0, 0, 0, 1]])

        if self.vis:
            # Add a coordinate frame to visualize orientation
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd, coord_frame], window_name="RealSense PCD")
        if save:
            self.save_data(pcd, "pcd", prefix)

        return pcd

    def get_intrinsics_matrix(self):
        """Return the aligned stream's 3x3 intrinsic matrix (numpy array)."""
        if self.o3d_intrinsics:
            return self.o3d_intrinsics.intrinsic_matrix
        return None

    def get_realsense_intrinsics(self):
        """Return the aligned stream's pyrealsense2.intrinsics object."""
        if self.profile:
            if self.align_to_color:
                return self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            else:
                return self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        return None

    def save_data(self, data, data_type="rgb", prefix=None, **kwargs):
        """
        Generic data save method.
        Args:
            data: data to save (RGB image / point cloud / depth image)
            data_type: data type ('rgb' / 'pcd' / 'depth')
            prefix: optional filename prefix
            kwargs: extra arguments (e.g., depth_scale for depth images)

        Returns: saved file path
        """
        # Ensure the directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"--> Save directory set to: {self.save_dir}")

        # Generate a unique filename
        self.save_counter += 1
        prefix = prefix or self.timestamp
        filename = f"{prefix}_{data_type}_{self.save_counter:04d}"

        # Ensure the directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"--> Save directory set to: {self.save_dir}")

        try:
            if data_type == "rgb":
                # Save RGB image
                file_path = os.path.join(self.save_dir, f"{filename}.png")
                cv2.imwrite(file_path, data)
                print(f"--> RGB saved to: {file_path}")

            elif data_type == "pcd":
                # Save point cloud
                file_path = os.path.join(self.save_dir, f"{filename}.ply")
                o3d.io.write_point_cloud(file_path, data)
                print(f"--> Point cloud saved to: {file_path}")

            elif data_type == "depth":
                # Save depth image
                depth_scale = kwargs.get('depth_scale', 1.0)
                depth_image = np.asanyarray(data.get_data())

                # Convert to an 8-bit visualization or save the raw 16-bit data
                if kwargs.get('visualize', False):
                    # 8-bit visualization depth image
                    file_path = os.path.join(self.save_dir, f"{filename}_vis.png")
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03),
                        cv2.COLORMAP_JET
                    )
                    cv2.imwrite(file_path, depth_colormap)
                else:
                    # Raw 16-bit depth data
                    file_path = os.path.join(self.save_dir, f"{filename}.png")
                    cv2.imwrite(file_path, depth_image)

                print(f"--> Depth image saved to: {file_path}")

            elif data_type=="calibrate":
                # calibration
                file_path = os.path.join(self.save_dir, "realsense_d455_intrinsics.yaml")
                self.save_calibration(file_path, overwrite=False)
                print(f"--> calibration result saved to: {file_path}")

            else:
                raise ValueError(f"Unsupported data type: {data_type}")

            return file_path

        except Exception as e:
            print(f"!!! Failed to save data: {e}")
            return None

    def save_calibration(self, file_path: str, overwrite: bool = False, format: str = 'yaml') -> bool:
        """
        Save camera calibration parameters to a YAML or JSON file.
        :param file_path: destination file path
        :param overwrite: whether to overwrite an existing file
        :param format: 'yaml' or 'json'
        :return: success flag
        """
        # Skip if the file exists and overwriting is disabled
        if os.path.exists(file_path) and not overwrite:
            print(f"Calibration file already exists at {file_path}, skipping save")
            return False

        # Make Sure have intrinsics
        if self.o3d_intrinsics is None:
            # Try to get intrinsics if not available (e.g. if start() wasn't called but we want to save defaults)
            # This part might need adjustment if used purely offline without 'start'
            if self.profile:  # if camera was started
                self.get_realsense_intrinsics()  # This sets self.o3d_intrinsics
            else:
                logger.warning("Cannot save calibration: o3d_intrinsics not available and camera not started.")
                return False

        rs_intr = self.get_realsense_intrinsics()  # Get full rs intrinsics

        calibration_data = {
            'camera_name': self.camera_name,
            'width': self.o3d_intrinsics.width,
            'height': self.o3d_intrinsics.height,
            'K': self.o3d_intrinsics.intrinsic_matrix.tolist(),  # 3x3 matrix
            'fx': float(self.o3d_intrinsics.intrinsic_matrix[0, 0]),
            'fy': float(self.o3d_intrinsics.intrinsic_matrix[1, 1]),
            'cx': float(self.o3d_intrinsics.intrinsic_matrix[0, 2]),
            'cy': float(self.o3d_intrinsics.intrinsic_matrix[1, 2]),
            'distortion_model': rs_intr.model.name if rs_intr else 'none',  # e.g., 'BrownConrady', 'None'
            'D': list(rs_intr.coeffs) if rs_intr else [],  # Distortion coefficients
            'depth_scale': self.depth_scale
            # 'P' and 'R' from ROS CameraInfo are not directly here unless calculated
        }

        try:
            if format.lower() == 'yaml':
                if not file_path.endswith(".yaml"): file_path += ".yaml"
                with open(file_path, 'w') as f:
                    yaml.dump(calibration_data, f, sort_keys=False)
            elif format.lower() == 'json':
                if not file_path.endswith(".json"): file_path += ".json"
                with open(file_path, 'w') as f:
                    json.dump(calibration_data, f, indent=4)
            else:
                raise ValueError("Unsupported format. Choose 'yaml' or 'json'.")
            logger.info(f"Saved calibration to {file_path} (format: {format})")
            return True
        except Exception as e:
            logger.error(f"Failed to save calibration: {str(e)}")
            return False

    def load_calibration(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            logger.error(f"Calibration file not found: {file_path}")
            return False
        try:
            if file_path.endswith(".yaml") or file_path.endswith(".yml"):
                with open(file_path, 'r') as f:
                    calib_data = yaml.safe_load(f)
            elif file_path.endswith(".json"):
                with open(file_path, 'r') as f:
                    calib_data = json.load(f)
            else:
                logger.error(f"Unsupported calibration file extension: {file_path}. Use .yaml or .json.")
                return False

            self.width = calib_data['width']
            self.height = calib_data['height']

            # Prefer K if available, otherwise fx,fy,cx,cy
            if 'K' in calib_data:
                K_matrix = np.array(calib_data['K']).reshape((3,3))
                fx, fy = K_matrix[0, 0], K_matrix[1, 1]
                cx, cy = K_matrix[0, 2], K_matrix[1, 2]
            elif all(k in calib_data for k in ['fx', 'fy', 'cx', 'cy']):
                fx, fy, cx, cy = calib_data['fx'], calib_data['fy'], calib_data['cx'], calib_data['cy']
            else:
                raise ValueError("Intrinsics matrix (K) or parameters (fx,fy,cx,cy) not found in calibration file.")

            self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                self.width, self.height, fx, fy, cx, cy
            )

            if 'depth_scale' in calib_data:
                self.depth_scale = calib_data['depth_scale']

            # Store distortion coefficients if available (for project_image_to_point_cloud if it uses cv2.projectPoints)
            self.distortion_coeffs_cv = np.array(calib_data.get('D', [0, 0, 0, 0, 0])[:5])  # Get D or default to zeros

            logger.info(f"Loaded calibration from {file_path}")
            logger.info(f"  Dimensions: {self.width}x{self.height}")
            logger.info(f"  Intrinsics (fx,fy,cx,cy): {fx:.2f}, {fy:.2f}, {cx:.2f}, {cy:.2f}")
            if np.any(self.distortion_coeffs_cv):
                logger.info(f"  Distortion coeffs: {self.distortion_coeffs_cv}")

            return True
        except Exception as e:
            logger.error(f"Failed to load calibration: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def get_color_point_cloud_uv(self, pc: np.ndarray) -> np.ndarray:
        """
        Project the point cloud onto the color image plane and return UV coordinates.
        """
        intrinsics = self.get_intrinsics_matrix()
        assert intrinsics is not None

        # Project the point cloud onto the color camera plane
        pc_camera = np.hstack((pc, np.ones((pc.shape[0], 1), dtype=np.float32)))
        pc_image = (intrinsics @ pc_camera[:, :3].T).T
        pc_uv = pc_image[:, :2] / pc_camera[:, 2][:, np.newaxis]

        return pc_uv

    def project_image_to_point_cloud(self,
                                     image: np.ndarray,
                                     pc: np.ndarray,
                                     dtype=np.uint8) -> np.ndarray:
        """
        Sample image pixel values onto each point in the point cloud.
        image: 2D mask or color image (H, W, C) or (H, W).
        pc_xyz: (N,3) numpy array of point coordinates.
        """
        if self.o3d_intrinsics is None:  # Ensure intrinsics are loaded
            raise RuntimeError("Camera intrinsics not loaded. Call load_calibration() first.")

        pc_uv = self.get_color_point_cloud_uv(pc)
        width = image.shape[1]
        height = image.shape[0]

        projected = np.zeros((pc.shape[0], image.shape[2]), dtype=dtype)
        valid_idxs = ((pc_uv[:, 0] >= 0) & (pc_uv[:, 0] < width) &
                      (pc_uv[:, 1] >= 0) & (pc_uv[:, 1] < height))

        if not np.any(valid_idxs):
            # logger.warning("No points projected within image boundaries.")
            return valid_idxs  # All zeros

        valid_uv = np.floor(pc_uv[valid_idxs, :]).astype(np.int32)
        projected[valid_idxs, :] = image[valid_uv[:, 1], valid_uv[:, 0], :]

        return projected


if __name__ == '__main__':

    # set save path
    current_path = osp.dirname(osp.abspath(__file__))
    save_path = os.path.join(current_path, 'realsense_data')

    # flag
    save_flag = False # if you want to save file, set to True
    vis_flag = True


    # initial
    camera = RealSenseCamera(vis=vis_flag)

    # if you want to load calibrate file:
    REPO_ROOT = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
    DEFAULT_SAMPLE_ROOT = os.path.join(REPO_ROOT, "data", "sample")
    DATA_ROOT = os.environ.get("RGBENCH_DATA_ROOT", DEFAULT_SAMPLE_ROOT)
    calibration_path = os.environ.get(
        "RGBENCH_REALSENSE_INTRINSICS",
        os.path.join(DATA_ROOT, "realsense_data", "intrinsics", "camera_intrinsics.json"),
    )
    calibrate = camera.load_calibration(calibration_path)
    intr = camera.o3d_intrinsics
    print(intr)

    # if you want to get data
    try:
        # camera.start()
        # save calibrate
        # camera.save_data(camera.o3d_intrinsics, "calibrate")
        # It is worth mentioning that every realsense initialization the intrinsic parameter is automatically calibrated
        # calibration_path = "../../data/calibration/realsense_d455_intrinsics.yaml"
        # camera.save_calibration(calibration_path, overwrite=false) #  you can set calibration too

        # get data, you can those
        # rgb = camera.capture_rgb(save=save_flag)
        # pcd = camera.capture_pcd(save=save_flag)

        # read calibrate
        # calibration_path =osp.join(camera.save_dir, f"{camera.camera_name}_intrinsics.yaml")
        calibrate = camera.load_calibration(calibration_path)
        intr = camera.o3d_intrinsics
        print(intr)

        # for i in range(5):
        #     print(f'Tring to capture {i}-th color image!')
            # rgb = camera.capture_rgb()
            # print(f'Tring to capture {i}-th point cloud!')
            # pcd = camera.capture_pcd()
    finally:
        camera.stop()

