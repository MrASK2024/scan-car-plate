"""Utility functions for data loading."""

from typing import Any

import torch


def init_dataloader(
    dataset: Any, batch_size: int, shuffle: bool = True, num_workers: int = 0
):
    """Initialize dataloader with given parameters."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: tuple(zip(*batch, strict=False)),
    )
