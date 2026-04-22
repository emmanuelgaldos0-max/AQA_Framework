from pathlib import Path

from src.utils.config import load_config


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_base_loads():
    cfg = load_config(CONFIGS / "base.yaml")
    assert cfg["seed"] == 42
    assert cfg["data"]["clip_length"] == 64


def test_teacher_extends_base():
    cfg = load_config(CONFIGS / "teacher_i3d.yaml")
    assert cfg["seed"] == 42                    # heredado de base
    assert cfg["model"]["name"] == "i3d"        # propio
    assert cfg["train"]["batch_size"] == 2      # override
    assert cfg["train"]["epochs"] == 50         # heredado


def test_kd_config():
    cfg = load_config(CONFIGS / "kd.yaml")
    assert cfg["kd"]["alpha_reg"] == 1.0
    assert cfg["kd"]["beta_att"] == 0.5
    assert cfg["kd"]["warmup_epochs"] == 5
