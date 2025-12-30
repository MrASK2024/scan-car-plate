"""Transforms for image resizing and normalization."""

import torchvision.transforms.functional as F


class ResizeAndNorm:
    """Resize and normalize images, adjust bounding boxes accordingly."""

    def __init__(self, size, mean, std):
        """Initialize resize and normalize transform."""
        self.size = tuple(size)
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        """Resize image and normalize it, adjust bounding boxes accordingly."""
        w0, h0 = image.size
        image = F.resize(image, self.size)
        w1, h1 = self.size

        scale_x = w1 / w0
        scale_y = h1 / h0
        boxes = target["boxes"]
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        target["boxes"] = boxes

        image = F.to_tensor(image)
        image = F.normalize(image, self.mean, self.std)

        return image, target
