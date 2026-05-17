import sys
import os
import os.path as osp

import numpy as np
import open3d as o3d
import cv2

sys.path.append(osp.join(osp.dirname(__file__),".."))

from third_party.grounded_sam.grounded_sam import GroundedSAM
from third_party.realsense.realsense import RealSenseCamera

from rgbench.config import config_completion

from omegaconf import OmegaConf, DictConfig, open_dict, ListConfig
from typing import Union,Dict
from easydict import EasyDict


class PointCloudProcessor:
    def __init__(self, camera: RealSenseCamera, config: Union[Dict, EasyDict, str]=None):
        self.camera = camera

        if config is not None:
            self.option: DictConfig = config_completion(config)
            self.segmentation_model = GroundedSAM(**self.option.segmentation)
        else:
            self.segmentation_model = GroundedSAM()

        # Coordinate-transform parameters can be added if needed
        # self.transforms = ...

    def capture_pcd_with_mask(self, rgb_img_bgr=None, camera_pcd=None):
        """
        Capture a point cloud and extract the target object via mask segmentation.
        """
        # 1. Capture RGB image and point cloud
        if rgb_img_bgr is None or camera_pcd is None:
            # Pull from the camera if either argument is missing
            rgb_img_bgr = self.camera.capture_rgb()
            camera_pcd = self.camera.capture_pcd()


        rgb_img = cv2.cvtColor(rgb_img_bgr, cv2.COLOR_BGR2RGB)  # convert to RGB

        # Extract point cloud coordinates
        pc_xyz = np.asarray(camera_pcd.points).copy()

        # 2. Run the segmentation model to obtain masks
        masks = self.segmentation_model.predict(rgb_img)  # model prediction (k, h, w)

        # 3. Mask post-processing - pick the target object
        mask_sum = masks.sum(axis=-1).sum(axis=-1)  # area of each mask
        h, w = masks.shape[1:]

        # Drop overly large masks (e.g., the table)
        max_mask_ratio = 0.8  # tune as needed
        mask_sum[mask_sum > h * w * max_mask_ratio] = 0

        # Pick the largest remaining valid mask
        max_mask_idx = np.argmax(mask_sum)

        # Build a 3-channel mask image
        mask_img = np.zeros((h, w, 3), dtype=np.uint8)
        mask_img[masks[max_mask_idx] > 0] = [255, 255, 255]  # white marks the target region

        # 4. Project the mask into the point cloud space
        mask_values = self.camera.project_image_to_point_cloud(
            mask_img, pc_xyz, dtype=np.uint8
        )

        # 5. Extract the target object's point cloud via the mask
        # Red channel > 0 indicates the target object
        valid_idxs = mask_values[:, 0] > 0

        # Build a point cloud containing only the target object
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(pc_xyz[valid_idxs, :])

        # Read and set the color information
        point_colors = np.asarray(camera_pcd.colors)[valid_idxs]
        target_pcd.colors = o3d.utility.Vector3dVector(point_colors)

        # 6. Optional: transform to the world frame
        # if hasattr(self, 'transforms'):
        #     target_pcd = target_pcd.transform(self.transforms.camera_to_world_transform)

        return target_pcd, mask_img, rgb_img


# Usage example
if __name__ == "__main__":

    # 1. Pull the camera point cloud
    camera = RealSenseCamera(width=640, height=480, fps=30, align_to_color=True)
    rgb_img_bgr = cv2.imread("../.png")


    # 2. Optionally load configuration
    # cfg = OmegaConf.load("../config/segment_realsense_pcd.yaml")


    # 3. Build the processor
    processor = PointCloudProcessor(camera)

    # 4. Capture and process the point cloud
    target_pcd, mask_img, rgb_img = processor.capture_pcd_with_mask()

    # 5. Visualize
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([coord, target_pcd])

    # Display the original image and mask
    cv2.imshow("RGB Image", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    cv2.imshow("Object Mask", mask_img)
    cv2.waitKey(0)

    # 6. Cleanup
    camera.stop()