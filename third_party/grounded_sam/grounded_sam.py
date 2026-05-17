import cv2
import numpy as np
import supervision as sv

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

import os
import os.path as osp

from typing import Tuple

class GroundedSAM:
    def __init__(self, 
                 grounding_dino_config_path: str = "checkpoints/GroundingDINO_SwinT_OGC.py",
                 grounding_dino_checkpoint_path: str = "checkpoints/groundingdino_swint_ogc.pth",
                 sam_encoder_version: str = "vit_h",
                 sam_checkpoint_path: str = "checkpoints/sam_vit_h_4b8939.pth",
                 classes: list = ["brown cloth"],
                 box_threshold: float = 0.25,
                 text_threshold: float  = 0.25,
                 nms_threshold: float = 0.8,
                 vis: bool = False,
                 **kwargs) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(model_config_path=grounding_dino_config_path, 
                                          model_checkpoint_path=grounding_dino_checkpoint_path,
                                          device=device)
        
        # self.grounding_dino_model = Model(model_config_path=grounding_dino_config_path, 
        #                                   model_checkpoint_path=grounding_dino_checkpoint_path,
        #                                   device=device)

        # Building SAM Model and SAM Predictor
        sam = sam_model_registry[sam_encoder_version](checkpoint=sam_checkpoint_path).to(torch.device(device))
        self.sam_predictor = SamPredictor(sam)

        # params
        self.classes = list(classes)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        self.vis = vis
        self.save = kwargs.get("save", False)

    def predict(self, image: np.ndarray) -> np.ndarray:
        # detect objects
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=self.classes,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{self.classes[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _
            in detections]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)

        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            self.nms_threshold
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        print(f"After NMS: {len(detections.xyxy)} boxes")

        # convert detections to masks
        detections.mask = self.segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{self.classes[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _
            in detections]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)

        if self.vis:
            # visualize the annotated grounding dino image
            cv2.imshow("groundingdino_annotated_image.jpg", annotated_frame)
            cv2.waitKey(0)
            # visualize the annotated grounded-sam image
            cv2.imshow("grounded_sam_annotated_image.jpg", annotated_image)
            cv2.waitKey(0)

        if self.save:
            project_root = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
            prefix = "cloth1"
            grounding_dino_path = osp.join(project_root, "data/images/grounded_sam_output",
                                           prefix + "_groundingdino_annotated_image.jpg")
            grounded_sam_path = osp.join(project_root, "data/images/grounded_sam_output",
                                         prefix + "_grounded_sam_annotated_image.jpg")
            cv2.imwrite(grounding_dino_path, annotated_frame)
            cv2.imwrite(grounded_sam_path, annotated_image)

        return detections.mask

    def predict_and_get_annotated_images(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs prediction and returns masks along with annotated images.
        """
        # detect objects
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=self.classes,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{self.classes[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _
            in detections]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)

        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            self.nms_threshold
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        print(f"After NMS: {len(detections.xyxy)} boxes")

        # convert detections to masks
        detections.mask = self.segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{self.classes[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _
            in detections]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)

        return detections.mask, annotated_frame, annotated_image

    @staticmethod
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        # Prompting SAM with detected boxes
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)
    
if __name__ == '__main__':
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    DEFAULT_SAMPLE_ROOT = os.path.join(REPO_ROOT, "data", "sample")
    SOURCE_IMAGE_PATH = os.environ.get(
        "RGBENCH_GROUNDED_SAM_IMAGE",
        os.path.join(DEFAULT_SAMPLE_ROOT, "images", "cloth_rgb_0001.png"),
    )

    # load image
    image = cv2.imread(SOURCE_IMAGE_PATH)

    grounded_sam_model = GroundedSAM(vis=True)

    masks = grounded_sam_model.predict(image)
