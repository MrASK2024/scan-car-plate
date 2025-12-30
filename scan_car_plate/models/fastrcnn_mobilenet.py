"""Module to create Faster R-CNN model with MobileNetV3 backbone."""

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_fasterrcnn_mobilenet(
    num_classes: int, max_detections_per_image: int
) -> FasterRCNN:
    """Create Faster R-CNN model with MobileNetV3 backbone."""
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights="DEFAULT",
        box_detections_per_img=max_detections_per_image,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
