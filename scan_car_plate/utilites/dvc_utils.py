"""DVC utilities."""

from dvc.repo import Repo
from omegaconf import DictConfig


def dvc_pull_dataset(cfg: DictConfig):
    """Pull dataset from DVC remote storage."""
    with Repo(cfg.data.repo_root) as repo:
        repo.pull(targets=[cfg.data.dvc_dataset_dir])
