"""
Unit and Integration Tests for YOLOv8s-AMSA Integration (Phase 8A).

Validates:
1. Offline YAML parsing and model construction via YOLO("configs/yolov8s-amsa.yaml").
2. Graph topology: 26 layers, correct backbone main path preservation, and lateral AMSA routing.
3. Dummy forward pass at 640x640:
   - AMSA P3: [B, 128, 80, 80]
   - AMSA P4: [B, 256, 40, 40]
   - AMSA P5: [B, 512, 20, 20]
   - Detect receives three valid pyramid levels: [B, 128, 80, 80], [B, 256, 40, 40], [B, 512, 20, 20]
4. Numerical stability: zero NaNs and Infs in forward predictions.
5. Parameter counts:
   - AMSA P3: 85,098
   - AMSA P4: 315,282
   - AMSA P5: 1,230,306
   - AMSA Total addition: 1,630,686
   - Full model: 12,797,246
6. Pretrained checkpoint transfer audit via intersect_dicts.
7. Autocast compatibility (CPU bfloat16).
"""

from pathlib import Path
import pytest
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import intersect_dicts

import src.amsa  # Ensures AMSAModule is registered into ultralytics


CONFIG_PATH = Path("configs/yolov8s-amsa.yaml")


@pytest.fixture(scope="module")
def amsa_yolo() -> YOLO:
    """Instantiate YOLOv8s-AMSA model once for module test methods."""
    assert CONFIG_PATH.exists(), f"Configuration file {CONFIG_PATH} does not exist."
    return YOLO(str(CONFIG_PATH))


class TestYOLOv8sAMSA:
    """Test suite for YOLOv8s-AMSA model integration."""

    def test_model_construction_offline(self, amsa_yolo: YOLO) -> None:
        """Verify model constructs cleanly without network requests."""
        model = amsa_yolo.model
        assert model is not None
        assert len(model.model) == 26

    def test_layer_types_and_indices(self, amsa_yolo: YOLO) -> None:
        """Verify layer sequence, modules, and routing indices."""
        layers = amsa_yolo.model.model

        # Backbone check: layers 0-9
        assert layers[0]._get_name() == "Conv"
        assert layers[4]._get_name() == "C2f"  # Backbone P3
        assert layers[6]._get_name() == "C2f"  # Backbone P4
        assert layers[9]._get_name() == "SPPF"  # Backbone P5

        # Lateral AMSA nodes: layers 10, 11, 12
        assert layers[10]._get_name() == "AMSAModule"
        assert layers[11]._get_name() == "AMSAModule"
        assert layers[12]._get_name() == "AMSAModule"

        assert layers[10].channels == 128
        assert layers[10].scale_level == 3
        assert layers[11].channels == 256
        assert layers[11].scale_level == 4
        assert layers[12].channels == 512
        assert layers[12].scale_level == 5

        # Detect head: layer 25
        assert layers[25]._get_name() == "Detect"
        assert layers[25].f == [18, 21, 24]

    def test_dummy_forward_and_intermediate_shapes(self, amsa_yolo: YOLO) -> None:
        """
        Verify runtime tensor shapes on 640x640 input:
            AMSA P3: [B, 128, 80, 80]
            AMSA P4: [B, 256, 40, 40]
            AMSA P5: [B, 512, 20, 20]
            Detect receives: [B, 128, 80, 80], [B, 256, 40, 40], [B, 512, 20, 20]
        """
        B = 2
        x = torch.randn(B, 3, 640, 640)
        captured_shapes = {}

        def record_shape(name):
            def hook(module, inp, out):
                if isinstance(out, torch.Tensor):
                    captured_shapes[name] = tuple(out.shape)
            return hook

        model = amsa_yolo.model.eval()
        hooks = [
            model.model[4].register_forward_hook(record_shape("backbone_p3")),
            model.model[6].register_forward_hook(record_shape("backbone_p4")),
            model.model[9].register_forward_hook(record_shape("backbone_p5")),
            model.model[10].register_forward_hook(record_shape("amsa_p3")),
            model.model[11].register_forward_hook(record_shape("amsa_p4")),
            model.model[12].register_forward_hook(record_shape("amsa_p5")),
            model.model[18].register_forward_hook(record_shape("neck_p3")),
            model.model[21].register_forward_hook(record_shape("neck_p4")),
            model.model[24].register_forward_hook(record_shape("neck_p5")),
        ]

        try:
            with torch.no_grad():
                preds = model(x)
        finally:
            for h in hooks:
                h.remove()

        # Backbone outputs
        assert captured_shapes["backbone_p3"] == (B, 128, 80, 80)
        assert captured_shapes["backbone_p4"] == (B, 256, 40, 40)
        assert captured_shapes["backbone_p5"] == (B, 512, 20, 20)

        # AMSA calibrated lateral outputs
        assert captured_shapes["amsa_p3"] == (B, 128, 80, 80)
        assert captured_shapes["amsa_p4"] == (B, 256, 40, 40)
        assert captured_shapes["amsa_p5"] == (B, 512, 20, 20)

        # Neck outputs fed into Detect
        assert captured_shapes["neck_p3"] == (B, 128, 80, 80)
        assert captured_shapes["neck_p4"] == (B, 256, 40, 40)
        assert captured_shapes["neck_p5"] == (B, 512, 20, 20)

        # Numerical stability
        assert isinstance(preds, (tuple, list, torch.Tensor))
        if isinstance(preds, (tuple, list)):
            for p in preds:
                if isinstance(p, torch.Tensor):
                    assert not torch.isnan(p).any()
                    assert not torch.isinf(p).any()
        else:
            assert not torch.isnan(preds).any()
            assert not torch.isinf(preds).any()

    def test_parameter_counts(self, amsa_yolo: YOLO) -> None:
        """Verify analytical and measured parameter counts."""
        layers = amsa_yolo.model.model
        p3_params = sum(p.numel() for p in layers[10].parameters())
        p4_params = sum(p.numel() for p in layers[11].parameters())
        p5_params = sum(p.numel() for p in layers[12].parameters())
        amsa_total = p3_params + p4_params + p5_params

        assert p3_params == 85_098
        assert p4_params == 315_282
        assert p5_params == 1_230_306
        assert amsa_total == 1_630_686

        total_model_params = sum(p.numel() for p in amsa_yolo.model.parameters())
        assert total_model_params == 12_797_246

    def test_pretrained_transfer_audit(self, amsa_yolo: YOLO) -> None:
        """
        Verify pretrained weight transfer behavior:
            - 100% of backbone keys match and transfer directly.
            - Neck/head keys do not match directly due to index shift.
        """
        stock = YOLO("yolov8s.yaml")
        da = stock.model.state_dict()
        db = amsa_yolo.model.state_dict()

        csd = intersect_dicts(da, db)

        backbone_stock_keys = [k for k in da if any(k.startswith(f"model.{i}.") for i in range(10))]
        backbone_transferred = [k for k in csd if any(k.startswith(f"model.{i}.") for i in range(10))]

        # Backbone is 100% preserved
        assert len(backbone_stock_keys) == 162
        assert len(backbone_transferred) == 162

        # Non-backbone keys
        neck_head_transferred = [k for k in csd if not any(k.startswith(f"model.{i}.") for i in range(10))]
        # Only scalar num_batches_tracked buffers match by name coincidence
        for k in neck_head_transferred:
            assert "num_batches_tracked" in k

    def test_dummy_forward_1x3x640x640(self, amsa_yolo: YOLO) -> None:
        """
        Verify dummy forward pass with exact Task 8 specifications:
            Input: [1, 3, 640, 640]
            Verify:
                - no shape errors
                - no NaN/Inf
                - Detect receives three valid pyramid levels
        """
        x = torch.randn(1, 3, 640, 640)
        model = amsa_yolo.model.eval()

        detect_in_shapes = []

        def detect_pre_hook(module, inp):
            # module input to Detect is a tuple of list of the 3 pyramid feature maps
            if isinstance(inp, (list, tuple)) and len(inp) > 0 and isinstance(inp[0], (list, tuple)):
                detect_in_shapes.extend([tuple(t.shape) for t in inp[0]])
            elif isinstance(inp, (list, tuple)):
                detect_in_shapes.extend([tuple(t.shape) for t in inp if isinstance(t, torch.Tensor)])

        hook = model.model[25].register_forward_pre_hook(detect_pre_hook)
        try:
            with torch.no_grad():
                preds = model(x)
        finally:
            hook.remove()

        # Detect must receive three valid pyramid levels
        assert len(detect_in_shapes) == 3
        assert detect_in_shapes[0] == (1, 128, 80, 80)
        assert detect_in_shapes[1] == (1, 256, 40, 40)
        assert detect_in_shapes[2] == (1, 512, 20, 20)

        # Output predictions verification
        assert isinstance(preds, (tuple, list, torch.Tensor))
        if isinstance(preds, (tuple, list)):
            pred_tensor = preds[0] if isinstance(preds[0], torch.Tensor) else preds[1][0]
        else:
            pred_tensor = preds

        assert not torch.isnan(pred_tensor).any()
        assert not torch.isinf(pred_tensor).any()
        assert pred_tensor.shape == (1, 84, 8400)

    def test_bfloat16_forward(self) -> None:
        """Verify model forward pass under bfloat16 precision."""
        model = YOLO(str(CONFIG_PATH)).model.eval().to(torch.bfloat16)
        x = torch.randn(1, 3, 640, 640, dtype=torch.bfloat16)

        with torch.no_grad():
            out = model(x)

        if isinstance(out, (tuple, list)):
            pred = out[0] if isinstance(out[0], torch.Tensor) else out[1][0]
        else:
            pred = out

        assert not torch.isnan(pred).any()
        assert not torch.isinf(pred).any()
        assert pred.shape == (1, 84, 8400)

    def test_autocast_cpu_bfloat16_forward(self, amsa_yolo: YOLO) -> None:
        """Verify model forward pass under CPU bfloat16 autocast with FP32 master weights."""
        model = amsa_yolo.model.eval()
        # Verify master weights remain float32
        first_weight = next(model.parameters())
        assert first_weight.dtype == torch.float32

        x = torch.randn(1, 3, 640, 640, dtype=torch.float32)

        with torch.no_grad():
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                out = model(x)

        if isinstance(out, (tuple, list)):
            pred = out[0] if isinstance(out[0], torch.Tensor) else out[1][0]
        else:
            pred = out

        assert not torch.isnan(pred).any()
        assert not torch.isinf(pred).any()
        assert pred.shape == (1, 84, 8400)

    def test_autocast_cuda_float16_forward(self, amsa_yolo: YOLO) -> None:
        """Verify model forward pass under CUDA float16 autocast with FP32 master weights."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA device not available for float16 autocast testing")

        model = amsa_yolo.model.eval().cuda()
        first_weight = next(model.parameters())
        assert first_weight.dtype == torch.float32

        x = torch.randn(1, 3, 640, 640, dtype=torch.float32, device="cuda")

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(x)

        if isinstance(out, (tuple, list)):
            pred = out[0] if isinstance(out[0], torch.Tensor) else out[1][0]
        else:
            pred = out

        assert not torch.isnan(pred).any()
        assert not torch.isinf(pred).any()
        assert pred.shape == (1, 84, 8400)

