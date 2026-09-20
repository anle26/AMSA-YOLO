from src.amsa.scale_aware import ScaleAwareModule
from src.amsa.spatial_attention import AdaptiveSpatialAttention
from src.amsa.channel_attention import AdaptiveChannelAttention
from src.amsa.fusion import FeatureFusion
from src.amsa.amsa import AMSAModule
from src.amsa.pretrained import (
    WeightTransferReport,
    remap_yolov8s_key,
    transfer_yolov8s_weights,
)


def register_amsa() -> None:
    """Register AMSAModule into Ultralytics tasks and module namespaces."""
    try:
        import ultralytics.nn.tasks as tasks
        tasks.AMSAModule = AMSAModule
    except (ImportError, AttributeError):
        pass

    try:
        import ultralytics.nn.modules as modules
        modules.AMSAModule = AMSAModule
    except (ImportError, AttributeError):
        pass

    try:
        import ultralytics.nn.modules.block as block
        block.AMSAModule = AMSAModule
    except (ImportError, AttributeError):
        pass


# Auto-register on import
register_amsa()

__all__ = [
    "ScaleAwareModule",
    "AdaptiveSpatialAttention",
    "AdaptiveChannelAttention",
    "FeatureFusion",
    "AMSAModule",
    "WeightTransferReport",
    "remap_yolov8s_key",
    "transfer_yolov8s_weights",
    "register_amsa",
]


