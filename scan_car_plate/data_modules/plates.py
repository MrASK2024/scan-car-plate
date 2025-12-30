"""Dataset for YOLO formatted car plate data."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class YoloPlatesDataset(Dataset):
    """Dataset for YOLO formatted car plate data."""

    def __init__(self, images_dir, labels_dir, transforms=None):
        """Initialize dataset."""
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.img_ids = sorted([path.stem for path in Path(images_dir).glob("*.jpg")])

    def __getitem__(self, idx):
        """Get image and target by index."""
        img_id = self.img_ids[idx]
        img_path = f"{self.images_dir}/{img_id}.jpg"
        label_path = f"{self.labels_dir}/{img_id}.txt"

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes = []
        labels = []
        with open(label_path) as f:
            for line in f.readlines():
                c, cx, cy, bw, bh = map(float, line.split())

                cx *= w
                cy *= h
                bw *= w
                bh *= h

                x_min = cx - bw / 2
                y_min = cy - bh / 2
                x_max = cx + bw / 2
                y_max = cy + bh / 2

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(c) + 1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": 0,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        """Return length of dataset."""
        return len(self.img_ids)
