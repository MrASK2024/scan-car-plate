"""Lightning module for car plate detection using object detection model."""

import lightning as L
import torch
import torch.nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from scan_car_plate.utilites.metrics_utils import _prepare_for_metric


class CarPlateDetectModule(L.LightningModule):
    """Lightning module for car plate detection using object detection model."""

    def __init__(
        self, model: torch.nn.Module, lr: float, weight_decay: float, iou_thresholds
    ):
        """Initialize the detection module with model and training parameters."""
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.map_metric = MeanAveragePrecision(iou_thresholds=iou_thresholds)

    def forward(self, images):
        """Forward pass through the model."""
        return self.model(images)

    def training_step(self, batch):
        """Perform a training step."""
        images, target = batch
        loss_dict = self.model(images, target)
        loss = sum(loss_dict.values())
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=len(images),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        images, target = batch
        preds = self.model(images)
        self.map_metric.update(
            _prepare_for_metric(preds, is_prediction=True),
            _prepare_for_metric(target, is_prediction=False),
        )

    def on_validation_epoch_end(self):
        """Log validation metrics."""
        res = self.map_metric.compute()
        self.log("val_mAP@0.5", res["map_50"], prog_bar=True, logger=True)
        self.log("val_MAR@100", res["mar_100"], prog_bar=False, logger=True)
        self.map_metric.reset()

    def test_step(self, batch, batch_idx):
        """Perform a test step."""
        images, target = batch
        preds = self.model(images)
        self.map_metric.update(
            _prepare_for_metric(preds, is_prediction=True),
            _prepare_for_metric(target, is_prediction=False),
        )

    def on_test_epoch_end(self):
        """Log test metrics."""
        res = self.map_metric.compute()
        self.log("test_mAP@0.5", res["map_50"], prog_bar=True, logger=True)
        self.log("test_MAR@100", res["mar_100"], prog_bar=False, logger=True)
        self.map_metric.reset()

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
