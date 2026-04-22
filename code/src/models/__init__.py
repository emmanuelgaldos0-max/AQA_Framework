from .heads import RegressionHead
from .i3d import I3DRegressor, build_i3d
from .mobilenetv3_video import MobileNetV3Video, build_mobilenetv3
from .tsm import TemporalShift
from .tsm_mobilenetv2 import TSMMobileNetV2, build_tsm_mobilenetv2


def build_model(name: str, clip_length: int, pretrained: bool = True, **kwargs):
    """Factory de modelos. `name` según configs YAML."""
    name = name.lower()
    if name == "i3d":
        return build_i3d(pretrained=pretrained, **kwargs)
    if name == "tsm_mobilenetv2":
        return build_tsm_mobilenetv2(clip_length=clip_length, pretrained=pretrained, **kwargs)
    if name in {"mobilenetv3_large", "mobilenetv3"}:
        return build_mobilenetv3(clip_length=clip_length, pretrained=pretrained, **kwargs)
    raise ValueError(f"Modelo desconocido: {name}")


__all__ = [
    "RegressionHead",
    "TemporalShift",
    "I3DRegressor",
    "TSMMobileNetV2",
    "MobileNetV3Video",
    "build_i3d",
    "build_tsm_mobilenetv2",
    "build_mobilenetv3",
    "build_model",
]
