"""Data module for car plate detection using YOLO formatted datasets."""

import lightning as L
from data_modules.plates import YoloPlatesDataset
from data_modules.transforms import ResizeAndNorm

from scan_car_plate.utilites.data_utils import init_dataloader


class PlatesDataModule(L.LightningDataModule):
    """Data module for car plate detection using YOLO formatted datasets."""

    def __init__(
        self,
        train_batch_size: int,
        val_batch_size: int,
        train_images_dir: str,
        train_labels_dir: str,
        val_images_dir: str,
        val_labels_dir: str,
        test_images_dir: str,
        test_labels_dir: str,
        transforms,
    ):
        """Initialize data module with dataset paths and parameters."""
        super().__init__()
        self.train_images_dir = train_images_dir
        self.train_labels_dir = train_labels_dir
        self.val_images_dir = val_images_dir
        self.val_labels_dir = val_labels_dir
        self.test_images_dir = test_images_dir
        self.test_labels_dir = test_labels_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_transforms = ResizeAndNorm(
            size=transforms.size,
            mean=transforms.mean,
            std=transforms.std,
        )
        self.val_transforms = ResizeAndNorm(
            size=transforms.size,
            mean=transforms.mean,
            std=transforms.std,
        )

    def setup(self, stage=None):
        """Create datasets for different stages."""
        if stage in ("fit", None):
            self.train_dataset = YoloPlatesDataset(
                images_dir=self.train_images_dir,
                labels_dir=self.train_labels_dir,
                transforms=self.train_transforms,
            )
            self.val_dataset = YoloPlatesDataset(
                images_dir=self.val_images_dir,
                labels_dir=self.val_labels_dir,
                transforms=self.val_transforms,
            )
        if stage in ("test", None):
            self.test_dataset = YoloPlatesDataset(
                images_dir=self.test_images_dir,
                labels_dir=self.test_labels_dir,
                transforms=self.val_transforms,
            )

    def train_dataloader(self):
        """Create training dataloader."""
        return init_dataloader(self.train_dataset, self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        """Create validation dataloader."""
        return init_dataloader(self.val_dataset, self.val_batch_size, shuffle=False)

    def test_dataloader(self):
        """Create test dataloader."""
        return init_dataloader(self.test_dataset, self.val_batch_size, shuffle=False)
