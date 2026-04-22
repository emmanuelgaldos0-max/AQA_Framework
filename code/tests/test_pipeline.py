"""Test end-to-end del pipeline: dataset sintético → splits → DataLoader."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(*args):
    subprocess.run([PY, *args], cwd=ROOT, check=True, capture_output=True)


@pytest.fixture(scope="module")
def synth_splits(tmp_path_factory):
    """Genera dataset sintético chico y sus splits una sola vez por módulo."""
    _run("scripts/make_synthetic_dataset.py", "--n_clips", "30", "--dataset", "synth_test")
    _run("scripts/make_splits.py", "--dataset", "synth_test", "--seed", "42")
    return {
        "train": ROOT / "data" / "splits" / "synth_test_train.json",
        "val":   ROOT / "data" / "splits" / "synth_test_val.json",
        "test":  ROOT / "data" / "splits" / "synth_test_test.json",
    }


def test_splits_exist(synth_splits):
    for path in synth_splits.values():
        assert path.exists()


def test_splits_sizes_add_up(synth_splits):
    sizes = []
    for path in synth_splits.values():
        with open(path) as f:
            sizes.append(len(json.load(f)))
    assert sum(sizes) == 30


def test_dataset_loads(synth_splits):
    from src.datasets import AQABaseDataset, VideoTransform

    t = VideoTransform(clip_length=64, frame_size=224, is_train=False)
    ds = AQABaseDataset(
        split_path=synth_splits["train"],
        transform=t,
        root=ROOT,
    )
    assert len(ds) > 0
    sample = ds[0]
    assert sample["clip"].shape == (3, 64, 224, 224)
    assert 0.0 <= sample["score"].item() <= 1.0


def test_dataloader_batch(synth_splits):
    from src.datasets import AQABaseDataset, VideoTransform

    t = VideoTransform(clip_length=64, frame_size=224, is_train=True)
    ds = AQABaseDataset(
        split_path=synth_splits["train"],
        transform=t,
        root=ROOT,
    )
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(dl))
    assert batch["clip"].shape == (4, 3, 64, 224, 224)
    assert batch["score"].shape == (4,)
    assert batch["clip"].dtype == torch.float32
