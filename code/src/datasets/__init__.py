from .base import AQABaseDataset, Sample
from .transforms import VideoTransform


class AQA7Dataset(AQABaseDataset):
    name = "aqa7"


class MTLAQADataset(AQABaseDataset):
    name = "mtl_aqa"


class JIGSAWSDataset(AQABaseDataset):
    name = "jigsaws"


DATASET_REGISTRY = {
    "aqa7": AQA7Dataset,
    "mtl_aqa": MTLAQADataset,
    "jigsaws": JIGSAWSDataset,
}


def build_dataset(name: str, split: str, **kwargs) -> AQABaseDataset:
    """Factory. split ∈ {train, val, test}."""
    from pathlib import Path
    cls = DATASET_REGISTRY[name]
    root = Path(__file__).resolve().parents[2]  # code/
    split_path = root / "data" / "splits" / f"{name}_{split}.json"
    return cls(split_path=split_path, root=root, **kwargs)


__all__ = [
    "AQABaseDataset",
    "Sample",
    "VideoTransform",
    "AQA7Dataset",
    "MTLAQADataset",
    "JIGSAWSDataset",
    "DATASET_REGISTRY",
    "build_dataset",
]
