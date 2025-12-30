"""Utility functions for metric computations."""


def _prepare_for_metric(data, is_prediction=False):
    """Prepare data for metric computation by moving tensors to CPU."""
    required_keys = ["boxes", "labels"]
    if is_prediction:
        required_keys.append("scores")

    return [{k: item[k].cpu() for k in required_keys if k in item} for item in data]
