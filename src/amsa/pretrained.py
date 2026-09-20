"""
Pretrained Weight Transfer Utility for YOLOv8s-AMSA.

Architecture Relationship:
    Stock YOLOv8s:
        - Layers 0..9:   Backbone
        - Layers 10..21: PAN-FPN Neck
        - Layer 22:      Detect Head

    AMSA-YOLOv8s:
        - Layers 0..9:   Identical Backbone
        - Layers 10..12: Lateral AMSA Nodes (AMSA-P3, AMSA-P4, AMSA-P5)
        - Layers 13..24: Shifted PAN-FPN Neck (stock layers 10..21 shifted by +3)
        - Layer 25:      Shifted Detect Head (stock layer 22 shifted by +3)

Remapping Rule:
    For any stock key `model.{i}.{suffix}`:
        - If 0 <= i <= 9:   target = f"model.{i}.{suffix}"
        - If 10 <= i <= 22: target = f"model.{i + 3}.{suffix}"

    AMSA layers (10, 11, 12) are intentionally untouched and left fresh.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


@dataclass
class WeightTransferReport:
    """Detailed summary of pretrained weight remapping and transfer."""

    total_source_keys: int = 0
    total_target_keys: int = 0
    total_transferred: int = 0
    backbone_transferred: int = 0
    neck_transferred: int = 0
    head_transferred: int = 0
    transferred_keys: List[str] = field(default_factory=list)
    skipped_source_keys: List[Tuple[str, str]] = field(default_factory=list)
    missing_target_keys: List[str] = field(default_factory=list)
    shape_mismatches: List[Tuple[str, str, Tuple[int, ...], Tuple[int, ...]]] = field(default_factory=list)
    amsa_keys_left_fresh: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True if all source keys transferred with zero skips, missing keys, or mismatches."""
        return (
            len(self.skipped_source_keys) == 0
            and len(self.missing_target_keys) == 0
            and len(self.shape_mismatches) == 0
            and self.total_transferred == self.total_source_keys
        )

    def summary(self) -> str:
        """Return formatted human-readable summary string."""
        pct = (self.total_transferred / max(1, self.total_source_keys)) * 100.0
        lines = [
            "=" * 65,
            "             YOLOv8s -> AMSA-YOLO Weight Transfer Report        ",
            "=" * 65,
            f"Total Source Keys:       {self.total_source_keys}",
            f"Total Target Keys:       {self.total_target_keys}",
            f"Total Keys Transferred:  {self.total_transferred} / {self.total_source_keys} ({pct:.1f}%)",
            f"  - Backbone (0..9):     {self.backbone_transferred}",
            f"  - Neck (13..24):       {self.neck_transferred}",
            f"  - Head (Detect 25):    {self.head_transferred}",
            f"AMSA Keys Left Fresh:    {len(self.amsa_keys_left_fresh)} (layers 10, 11, 12)",
            f"Skipped Source Keys:     {len(self.skipped_source_keys)}",
            f"Missing Target Keys:     {len(self.missing_target_keys)}",
            f"Shape Mismatches:        {len(self.shape_mismatches)}",
            "=" * 65,
        ]
        return "\n".join(lines)


def remap_yolov8s_key(key: str, target_has_prefix: bool = True) -> Optional[str]:
    """
    Map a stock YOLOv8s state_dict key to its corresponding AMSA-YOLOv8s key.

    Handles keys with or without the leading 'model.' prefix, and outputs in the target's format.

    Rule:
        - layer index 0 <= i <= 9:   layer index remains i
        - layer index 10 <= i <= 22: layer index shifts to i + 3
        - Returns None if the key does not match the expected pattern or index is out of bounds.
    """
    parts = key.split(".")
    if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
        idx = int(parts[1])
        suffix = ".".join(parts[2:])
    elif len(parts) >= 2 and parts[0].isdigit():
        idx = int(parts[0])
        suffix = ".".join(parts[1:])
    else:
        return None

    if 0 <= idx <= 9:
        new_idx = idx
    elif 10 <= idx <= 22:
        new_idx = idx + 3
    else:
        return None

    return f"model.{new_idx}.{suffix}" if target_has_prefix else f"{new_idx}.{suffix}"


def transfer_yolov8s_weights(
    target_model: Union[nn.Module, Any],
    source_state_dict: Union[Dict[str, torch.Tensor], nn.Module, Any],
    strict_dtype: bool = False,
) -> WeightTransferReport:
    """
    Explicitly transfer weights from a stock YOLOv8s model or state_dict to AMSA-YOLOv8s.

    Validates key existence, exact tensor shape, and dtype compatibility before loading.
    AMSA lateral layers (10, 11, 12) are guaranteed untouched.

    Args:
        target_model: Target AMSA-YOLO model instance (nn.Module or YOLO wrapper).
        source_state_dict: Source state dict or model instance containing stock YOLOv8s weights.
        strict_dtype (bool): If True, requires source and target dtypes to match exactly.
                             If False, safely casts source tensor to target dtype before loading.

    Returns:
        WeightTransferReport containing detailed audit of transferred and skipped keys.
    """
    # Extract underlying nn.Module for target
    if hasattr(target_model, "predictor") or type(target_model).__name__ == "YOLO":
        raw_target = getattr(target_model, "model", target_model)
    elif isinstance(target_model, nn.Module):
        raw_target = target_model
    else:
        raise TypeError(
            f"Expected target_model to be an nn.Module or YOLO instance, got {type(target_model).__name__}"
        )

    # Extract source state dict
    if isinstance(source_state_dict, dict):
        if "model" in source_state_dict and isinstance(source_state_dict["model"], (dict, nn.Module)):
            inner = source_state_dict["model"]
            src_sd = inner.state_dict() if isinstance(inner, nn.Module) else inner
        else:
            src_sd = source_state_dict
    elif hasattr(source_state_dict, "predictor") or type(source_state_dict).__name__ == "YOLO":
        src_sd = source_state_dict.model.state_dict()
    elif isinstance(source_state_dict, nn.Module):
        src_sd = source_state_dict.state_dict()
    else:
        raise TypeError(
            f"Expected source_state_dict to be dict, nn.Module, or YOLO instance, got {type(source_state_dict).__name__}"
        )

    target_sd = raw_target.state_dict()
    target_has_prefix = any(k.startswith("model.") for k in target_sd.keys())

    # Identify AMSA keys in target that must remain untouched
    if target_has_prefix:
        amsa_fresh_keys = [
            k for k in target_sd.keys()
            if any(k.startswith(f"model.{i}.") for i in (10, 11, 12))
        ]
    else:
        amsa_fresh_keys = [
            k for k in target_sd.keys()
            if any(k.startswith(f"{i}.") for i in (10, 11, 12))
        ]

    remapped_dict: Dict[str, torch.Tensor] = {}
    report = WeightTransferReport(
        total_source_keys=len(src_sd),
        total_target_keys=len(target_sd),
        amsa_keys_left_fresh=amsa_fresh_keys,
    )

    for src_key, src_tensor in src_sd.items():
        if not isinstance(src_tensor, torch.Tensor):
            report.skipped_source_keys.append((src_key, f"Value is not a torch.Tensor ({type(src_tensor).__name__})"))
            continue

        target_key = remap_yolov8s_key(src_key, target_has_prefix=target_has_prefix)
        if target_key is None:
            report.skipped_source_keys.append(
                (src_key, "Key does not match expected 'model.{0..22}.*' stock pattern")
            )
            continue

        # Target key existence check
        if target_key not in target_sd:
            report.missing_target_keys.append(target_key)
            report.skipped_source_keys.append(
                (src_key, f"Mapped target key '{target_key}' does not exist in target model")
            )
            continue

        # Guard: never write into AMSA layers
        t_parts = target_key.split(".")
        target_layer_idx = int(t_parts[1]) if t_parts[0] == "model" else int(t_parts[0])
        if target_layer_idx in (10, 11, 12):
            report.skipped_source_keys.append(
                (src_key, f"Target key '{target_key}' maps into protected AMSA lateral layer")
            )
            continue

        target_tensor = target_sd[target_key]

        # Shape validation: strict equality, no silent reshaping or truncation
        if src_tensor.shape != target_tensor.shape:
            report.shape_mismatches.append(
                (src_key, target_key, tuple(src_tensor.shape), tuple(target_tensor.shape))
            )
            report.skipped_source_keys.append(
                (src_key, f"Shape mismatch: source {tuple(src_tensor.shape)} != target {tuple(target_tensor.shape)}")
            )
            continue

        # Dtype validation
        if strict_dtype and src_tensor.dtype != target_tensor.dtype:
            report.skipped_source_keys.append(
                (src_key, f"Dtype mismatch: source {src_tensor.dtype} != target {target_tensor.dtype}")
            )
            continue

        # Clone and ensure matching dtype
        tensor_to_load = src_tensor.detach().clone()
        if tensor_to_load.dtype != target_tensor.dtype:
            tensor_to_load = tensor_to_load.to(dtype=target_tensor.dtype)

        remapped_dict[target_key] = tensor_to_load
        report.transferred_keys.append(target_key)

        # Increment layer category counts
        if 0 <= target_layer_idx <= 9:
            report.backbone_transferred += 1
        elif 13 <= target_layer_idx <= 24:
            report.neck_transferred += 1
        elif target_layer_idx == 25:
            report.head_transferred += 1

    report.total_transferred = len(report.transferred_keys)

    # Load remapped tensors into target model
    raw_target.load_state_dict(remapped_dict, strict=False)

    return report
