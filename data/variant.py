from .offline_dataset import OfflineDataset


def get_coverage_variants(
    dataset: OfflineDataset, coverage_percent: float
) -> OfflineDataset:
    """
    Args:
        dataset: Offline dataset
        coverage_percent: Proportion of trajectories that approach the boundary to be kept
    Returns:
        A new OfflineDataset object with only the kept trajectories
    """
    raise NotImplementedError()
