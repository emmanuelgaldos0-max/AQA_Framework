"""Carga de configs YAML con soporte de `extends`."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "extends" in cfg:
        parent_rel = cfg.pop("extends")
        parent_path = path.parent / parent_rel
        parent_cfg = load_config(parent_path)
        cfg = _deep_merge(parent_cfg, cfg)

    return cfg
