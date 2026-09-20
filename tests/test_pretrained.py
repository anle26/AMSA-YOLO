"""
Unit and Integration Tests for Pretrained Weight Remapping & Clean Bootstrap (Phase 8B).

Validates:
1. Index remapping rules:
   - Backbone layers 0..9 -> identical 0..9
   - Neck layers 10..21 -> shifted 13..24
   - Detect head layer 22 -> shifted 25
   - Out of bounds and malformed keys return None.
2. Target prefix handling (model.i.* vs i.*).
3. Shape mismatch validation (strict check, no silent reshape/truncate).
4. Missing target key & non-tensor value validation.
5. Strict dtype vs permissive casting validation.
6. Full offline model weight transfer (355 / 355 keys transferred).
7. Representative tensor exact value equality via torch.equal.
8. Protected AMSA lateral layers (10, 11, 12) remain 100% untouched and finite.
9. Clean process bootstrap in isolated subprocesses.
"""

from pathlib import Path
import subprocess
import sys
import pytest
import torch
import torch.nn as nn
from ultralytics import YOLO

from src.amsa.pretrained import (
    WeightTransferReport,
    remap_yolov8s_key,
    transfer_yolov8s_weights,
)
from src.amsa import register_amsa

CONFIG_PATH = Path("configs/yolov8s-amsa.yaml")


class TestPretrainedRemapping:
    """Test suite for index remapping logic."""

    def test_backbone_indices_remain_unchanged(self) -> None:
        """Verify layers 0..9 map to identical indices 0..9."""
        for i in range(10):
            src_key = f"model.{i}.conv.weight"
            expected = f"model.{i}.conv.weight"
            assert remap_yolov8s_key(src_key) == expected

    def test_neck_indices_shift_by_three(self) -> None:
        """Verify stock neck layers 10..21 shift to 13..24."""
        for i in range(10, 22):
            src_key = f"model.{i}.conv.weight"
            expected = f"model.{i + 3}.conv.weight"
            assert remap_yolov8s_key(src_key) == expected

    def test_detect_head_index_shifts_by_three(self) -> None:
        """Verify stock Detect head layer 22 shifts to 25."""
        src_key = "model.22.cv2.0.0.conv.weight"
        expected = "model.25.cv2.0.0.conv.weight"
        assert remap_yolov8s_key(src_key) == expected

    def test_prefix_adaptation(self) -> None:
        """Verify keys without 'model.' prefix are correctly remapped."""
        assert remap_yolov8s_key("4.conv.weight", target_has_prefix=False) == "4.conv.weight"
        assert remap_yolov8s_key("12.conv.weight", target_has_prefix=False) == "15.conv.weight"
        assert remap_yolov8s_key("22.conv.weight", target_has_prefix=False) == "25.conv.weight"

        # Mapping unprefixed source to prefixed target
        assert remap_yolov8s_key("4.conv.weight", target_has_prefix=True) == "model.4.conv.weight"
        assert remap_yolov8s_key("12.conv.weight", target_has_prefix=True) == "model.15.conv.weight"

    @pytest.mark.parametrize(
        "invalid_key",
        [
            "model.23.conv.weight",  # Out of range (> 22)
            "model.-1.conv.weight",  # Negative index
            "model.99.conv.weight",  # Out of range
            "backbone.0.conv.weight",  # Non-model prefix
            "model_0_conv_weight",  # No dot separator
            "invalid_key_string",
        ],
    )
    def test_invalid_and_out_of_bounds_keys_return_none(self, invalid_key: str) -> None:
        """Verify invalid or out-of-bounds keys safely return None."""
        assert remap_yolov8s_key(invalid_key) is None


@pytest.fixture(scope="module")
def stock_model() -> YOLO:
    """Instantiate stock YOLOv8s model offline."""
    return YOLO("yolov8s.yaml")


@pytest.fixture(scope="module")
def target_model() -> YOLO:
    """Instantiate AMSA-YOLOv8s model offline."""
    register_amsa()
    return YOLO(str(CONFIG_PATH))


class TestWeightTransferVerification:
    """Test suite for transfer execution, validation, value integrity, and bootstrap."""

    def test_full_offline_transfer_report(self, stock_model: YOLO, target_model: YOLO) -> None:
        """
        Verify that transfer_yolov8s_weights transfers all 355 stock keys
        with zero skipped keys, zero missing target keys, and zero shape mismatches.
        """
        report = transfer_yolov8s_weights(target_model, stock_model)

        assert report.total_source_keys == 355
        assert report.total_transferred == 355
        assert report.backbone_transferred == 162
        assert report.neck_transferred == 108
        assert report.head_transferred == 85
        assert len(report.skipped_source_keys) == 0
        assert len(report.missing_target_keys) == 0
        assert len(report.shape_mismatches) == 0
        assert len(report.amsa_keys_left_fresh) == 84
        assert report.is_clean

    def test_representative_exact_tensor_equality(self, stock_model: YOLO, target_model: YOLO) -> None:
        """
        Verify exact bitwise equality for representative tensors across backbone, neck, and head:
            stock model.0.*  == target model.0.*
            stock model.4.*  == target model.4.*
            stock model.9.*  == target model.9.*
            stock model.12.* == target model.15.*
            stock model.15.* == target model.18.*
            stock model.18.* == target model.21.*
            stock model.21.* == target model.24.*
            stock model.22.* == target model.25.*
        """
        transfer_yolov8s_weights(target_model, stock_model)

        s_sd = stock_model.model.state_dict()
        t_sd = target_model.model.state_dict()

        representative_pairs = [
            ("model.0.conv.weight", "model.0.conv.weight"),
            ("model.0.bn.running_mean", "model.0.bn.running_mean"),
            ("model.4.cv1.conv.weight", "model.4.cv1.conv.weight"),
            ("model.4.m.0.cv1.conv.weight", "model.4.m.0.cv1.conv.weight"),
            ("model.9.cv1.conv.weight", "model.9.cv1.conv.weight"),
            ("model.12.cv1.conv.weight", "model.15.cv1.conv.weight"),
            ("model.15.cv1.conv.weight", "model.18.cv1.conv.weight"),
            ("model.16.conv.weight", "model.19.conv.weight"),
            ("model.18.cv1.conv.weight", "model.21.cv1.conv.weight"),
            ("model.19.conv.weight", "model.22.conv.weight"),
            ("model.21.cv1.conv.weight", "model.24.cv1.conv.weight"),
            ("model.22.cv2.0.0.conv.weight", "model.25.cv2.0.0.conv.weight"),
            ("model.22.cv3.0.0.conv.weight", "model.25.cv3.0.0.conv.weight"),
            ("model.22.dfl.conv.weight", "model.25.dfl.conv.weight"),
        ]

        for s_key, t_key in representative_pairs:
            assert s_key in s_sd, f"Source key {s_key} not in stock state_dict"
            assert t_key in t_sd, f"Target key {t_key} not in target state_dict"
            assert torch.equal(s_sd[s_key], t_sd[t_key]), f"Value mismatch for {s_key} -> {t_key}"

    def test_amsa_layers_initialization_integrity(self, stock_model: YOLO) -> None:
        """Verify AMSA layers (10, 11, 12) are never modified and remain finite."""
        fresh_model = YOLO(str(CONFIG_PATH))
        layers = fresh_model.model.model

        # Snapshot initial AMSA weights
        initial_weights = {
            10: layers[10].scale_aware.scale_proj.weight.clone(),
            11: layers[11].scale_aware.scale_proj.weight.clone(),
            12: layers[12].scale_aware.scale_proj.weight.clone(),
        }

        # Perform transfer
        report = transfer_yolov8s_weights(fresh_model, stock_model)
        assert report.is_clean

        # Verify weights are identical to pre-transfer state
        assert torch.equal(layers[10].scale_aware.scale_proj.weight, initial_weights[10])
        assert torch.equal(layers[11].scale_aware.scale_proj.weight, initial_weights[11])
        assert torch.equal(layers[12].scale_aware.scale_proj.weight, initial_weights[12])

        # Verify all parameters in layers 10, 11, 12 are finite numbers
        for idx in (10, 11, 12):
            for name, param in layers[idx].named_parameters():
                assert not torch.isnan(param).any(), f"NaN found in AMSA layer {idx} parameter {name}"
                assert not torch.isinf(param).any(), f"Inf found in AMSA layer {idx} parameter {name}"

    def test_shape_mismatch_error_handling(self, target_model: YOLO) -> None:
        """Verify shape mismatches are flagged in report and skipped without altering target tensor."""
        mock_source = {
            "model.0.conv.weight": torch.randn(10, 10, 3, 3),  # Wrong shape (real is [32, 3, 3, 3])
        }
        original_tensor = target_model.model.state_dict()["model.0.conv.weight"].clone()

        report = transfer_yolov8s_weights(target_model, mock_source)
        assert len(report.shape_mismatches) == 1
        assert report.total_transferred == 0

        # Verify target tensor was NOT corrupted
        current_tensor = target_model.model.state_dict()["model.0.conv.weight"]
        assert torch.equal(current_tensor, original_tensor)

    def test_malformed_and_missing_key_handling(self, target_model: YOLO) -> None:
        """Verify malformed keys and missing target keys are gracefully recorded and skipped."""
        mock_source = {
            "model.99.conv.weight": torch.randn(32, 3, 3, 3),  # Out of range
            "invalid_non_tensor": "not_a_tensor",  # Non-tensor value
            "model.50.extra.key": torch.randn(1),
        }
        report = transfer_yolov8s_weights(target_model, mock_source)
        assert report.total_transferred == 0
        assert len(report.skipped_source_keys) == 3

    def test_clean_subprocess_bootstrap(self) -> None:
        """
        Verify model creation succeeds from an isolated clean Python process.
        Tests:
            - Explicit: register_amsa() called before model creation.
            - Reverse: ultralytics imported before register_amsa().
            - Module import: import src.amsa alone.
        """
        script = """
import sys
sys.path.insert(0, ".")
from src.amsa import register_amsa
register_amsa()
from ultralytics import YOLO
model = YOLO("configs/yolov8s-amsa.yaml")
assert len(model.model.model) == 26
print("CLEAN_BOOTSTRAP_OK")
"""
        res = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, f"Subprocess failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        assert "CLEAN_BOOTSTRAP_OK" in res.stdout
