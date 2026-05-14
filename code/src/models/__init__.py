from .heads import RegressionHead
from .i3d import I3DRegressor, build_i3d
from .mobilenetv2_video import MobileNetV2Video, build_mobilenetv2_video
from .mobilenetv3_video import MobileNetV3Video, build_mobilenetv3
from .slowfast import SlowFastRegressor, build_slowfast
from .tsm import TemporalShift
from .tsm_mobilenetv2 import TSMMobileNetV2, build_tsm_mobilenetv2
from .x3d import X3DRegressor, build_x3d


def build_model(name: str, clip_length: int, pretrained: bool = True, **kwargs):
    """Factory de modelos. `name` según configs YAML."""
    name = name.lower()
    if name == "i3d":
        return build_i3d(pretrained=pretrained, **kwargs)
    if name == "slowfast" or name == "slowfast_r50":
        return build_slowfast(pretrained=pretrained, **kwargs)
    if name in {"x3d", "x3d_m"}:
        return build_x3d(pretrained=pretrained, variant="x3d_m", **kwargs)
    if name == "tsm_mobilenetv2":
        return build_tsm_mobilenetv2(clip_length=clip_length, pretrained=pretrained, **kwargs)
    if name == "mobilenetv2_video":  # E7: TSM-MBv2 sin TSM
        return build_mobilenetv2_video(clip_length=clip_length, pretrained=pretrained, **kwargs)
    if name in {"mobilenetv3_large", "mobilenetv3"}:
        return build_mobilenetv3(clip_length=clip_length, pretrained=pretrained, **kwargs)
    raise ValueError(f"Modelo desconocido: {name}")


__all__ = [
    "RegressionHead",
    "TemporalShift",
    "I3DRegressor",
    "SlowFastRegressor",
    "TSMMobileNetV2",
    "MobileNetV2Video",
    "MobileNetV3Video",
    "build_i3d",
    "build_slowfast",
    "build_tsm_mobilenetv2",
    "build_mobilenetv2_video",
    "build_mobilenetv3",
    "build_model",
]
