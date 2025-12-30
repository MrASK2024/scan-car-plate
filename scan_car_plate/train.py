"""Train the car plate detection model."""

import subprocess

import hydra
import mlflow
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import MLFlowLogger
from module import CarPlateDetectModule
from omegaconf import DictConfig

from scan_car_plate.data_modules.car_numbers_data import PlatesDataModule
from scan_car_plate.models.fastrcnn_mobilenet import create_fasterrcnn_mobilenet
from scan_car_plate.utilites.dvc_utils import dvc_pull_dataset


def get_git_commit() -> str:
    """Get the current git commit hash."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    """Train the car plate detection model."""
    dvc_pull_dataset(cfg)

    datamodule = PlatesDataModule(
        train_batch_size=cfg.data.train_batch_size,
        val_batch_size=cfg.data.predict_batch_size,
        train_images_dir=cfg.data.train.images_dir,
        train_labels_dir=cfg.data.train.labels_dir,
        val_images_dir=cfg.data.val.images_dir,
        val_labels_dir=cfg.data.val.labels_dir,
        test_images_dir=cfg.data.test.images_dir,
        test_labels_dir=cfg.data.test.labels_dir,
        transforms=cfg.data.transforms,
    )

    datamodule.setup("fit")

    model = create_fasterrcnn_mobilenet(
        num_classes=cfg.model.num_classes,
        max_detections_per_image=cfg.model.max_detections_per_image,
    )
    module = CarPlateDetectModule(
        model=model,
        lr=cfg.model.optimizer.lr,
        weight_decay=cfg.model.optimizer.weight_decay,
        iou_thresholds=list(cfg.model.metrics.iou_thresholds),
    )

    mlflow_cfg = cfg.logging.mlflow
    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    mlflow_logger = MLFlowLogger(
        experiment_name=mlflow_cfg.experiment_name,
        run_name=mlflow_cfg.run_name,
        tracking_uri=mlflow_cfg.tracking_uri,
    )

    git_commit = get_git_commit()
    mlflow_logger.log_hyperparams(
        {
            "model.type": cfg.model.type,
            "model.num_classes": cfg.model.num_classes,
            "optimizer.lr": cfg.model.optimizer.lr,
            "optimizer.weight_decay": cfg.model.optimizer.weight_decay,
            "metrics.iou_thresholds": list(cfg.model.metrics.iou_thresholds),
            "data.train_batch_size": cfg.data.train_batch_size,
            "data.predict_batch_size": cfg.data.predict_batch_size,
            "git.commit": git_commit,
        }
    )

    trainer = Trainer(
        max_epochs=cfg.num_epochs,
        logger=mlflow_logger,
        callbacks=[TQDMProgressBar(refresh_rate=1, leave=True)],
        enable_progress_bar=True,
    )
    trainer.fit(module, datamodule=datamodule)
    torch.save(module.model.state_dict(), cfg.output_file)


if __name__ == "__main__":
    train()
