"""
Unit and Integration Tests for Top-Level AMSAModule Wrapper.

Validates:
1. Canonical YOLOv8s / AMSA-YOLO feature shapes:
   - P3: C=128, 80x80, scale_level=3
   - P4: C=256, 40x40, scale_level=4
   - P5: C=512, 20x20, scale_level=5
2. Output shape strictly preserves input shape X.
3. Intermediate tensor shapes when return_intermediates=True:
   - S         == [B, 64, H, W]
   - X_spatial == [B, C, H, W]
   - X_channel == [B, C, H, W]
   - X_output  == [B, C, H, W]
4. Exact composition test:
   AMSAModule forward strictly matches sequential manual invocation of:
   scale_aware -> spatial_attention -> channel_attention -> fusion.
5. Scale level routing across {3, 4, 5} and aliases.
6. Gradient propagation to input X, active scale embedding, spatial attention,
   channel attention, and fusion parameters.
7. Inactive scale embeddings remain untouched (None grad).
8. Numerical stability: zero NaNs and Infs during forward and backward.
9. Dynamic batch sizes and dynamic spatial dimensions.
10. Robust validation and error raising (scale_level, rank, channels, types).
11. Pure CPU execution.
12. Deterministic evaluation forward pass.
13. State dict cleanliness and distinct parameter addresses.
14. Wrapper parameter count strictly equals the sum of submodule counts,
    matching analytical totals for YOLOv8s P3/P4/P5.
15. Automatic Mixed Precision (AMP) / autocast compatibility (CPU bfloat16, CUDA float16).
16. Default forward returns pure torch.Tensor; return_intermediates=True returns tuple.
17. Serialization state_dict save/load roundtrip determinism.
"""

import io
from typing import Tuple
import pytest
import torch
import torch.nn as nn

from src.amsa.amsa import AMSAModule


class TestAMSAModule:
    """Test suite for AMSAModule wrapper."""

    # -------------------------------------------------------------------------
    # 1-3. Output and Intermediate Tensor Shapes
    # -------------------------------------------------------------------------
    def test_output_shape_equals_input_shape(self) -> None:
        """Verify that output tensor preserves the exact shape of input X."""
        module = AMSAModule(channels=64, scale_dim=64)
        x = torch.randn(3, 64, 32, 48)

        out = module(x, scale_level=3)
        assert isinstance(out, torch.Tensor)
        assert not isinstance(out, tuple)
        assert out.shape == x.shape

    @pytest.mark.parametrize(
        ("channels", "spatial_size", "scale_level", "label"),
        [
            (128, (80, 80), 3, "YOLOv8s P3 backbone output"),
            (256, (40, 40), 4, "YOLOv8s P4 backbone output"),
            (512, (20, 20), 5, "YOLOv8s P5 backbone output"),
        ],
    )
    def test_canonical_yolov8s_feature_levels(
        self, channels: int, spatial_size: Tuple[int, int], scale_level: int, label: str
    ) -> None:
        """Verify feature shape preservation across canonical P3, P4, and P5 levels."""
        H, W = spatial_size
        module = AMSAModule(channels=channels, scale_dim=64)
        x = torch.randn(2, channels, H, W)

        out = module(x, scale_level=scale_level)
        assert out.shape == (2, channels, H, W), f"Failed for {label}"

    def test_intermediate_tensor_shapes(self) -> None:
        """
        Verify intermediate tensor shapes when return_intermediates=True:
            S         == [B, 64, H, W]
            X_spatial == [B, C, H, W]
            X_channel == [B, C, H, W]
            X_output  == [B, C, H, W]
        """
        B, C, D, H, W = 4, 32, 64, 28, 28
        module = AMSAModule(channels=C, scale_dim=D)
        x = torch.randn(B, C, H, W)

        out, intermediates = module(x, scale_level=3, return_intermediates=True)

        assert out.shape == (B, C, H, W)
        assert intermediates["scale_encoding"].shape == (B, D, H, W)
        assert intermediates["x_spatial"].shape == (B, C, H, W)
        assert intermediates["x_channel"].shape == (B, C, H, W)

    # -------------------------------------------------------------------------
    # 4-5. Exact Composition & Scale Level Routing
    # -------------------------------------------------------------------------
    def test_exact_composition_matches_manual_invocation(self) -> None:
        """
        Verify that AMSAModule.forward exactly matches sequential manual invocation:
            S = module.scale_aware(x, scale_level)
            X_spatial = module.spatial_attention(x, S)
            X_channel = module.channel_attention(x, S)
            expected = module.fusion(x, X_channel, X_spatial)
        """
        module = AMSAModule(channels=32, scale_dim=64)
        module.eval()
        x = torch.randn(2, 32, 16, 16)
        scale_level = 3

        with torch.no_grad():
            s_manual = module.scale_aware(x, scale_level)
            x_spatial_manual = module.spatial_attention(x, s_manual)
            x_channel_manual = module.channel_attention(x, s_manual)
            expected = module.fusion(x, x_channel_manual, x_spatial_manual)

            actual = module(x, scale_level)

        assert torch.allclose(actual, expected, atol=1e-6)

    @pytest.mark.parametrize(
        ("scale_level", "active_key"),
        [
            (3, "3"),
            ("3", "3"),
            ("P3", "3"),
            (4, "4"),
            ("4", "4"),
            ("P4", "4"),
            (5, "5"),
            ("5", "5"),
            ("P5", "5"),
        ],
    )
    def test_scale_level_routing(self, scale_level: int, active_key: str) -> None:
        """Verify scale-level parameter selection and backward gradient routing."""
        module = AMSAModule(channels=32, scale_dim=64)
        x = (torch.randn(2, 32, 16, 16) + 1.0).requires_grad_(True)

        out = module(x, scale_level=scale_level)
        loss = out.sum()
        loss.backward()

        # Active embedding must have received gradients
        active_param = module.scale_aware.scale_embeddings[active_key]
        assert active_param.grad is not None
        assert not torch.all(active_param.grad == 0.0)

        # Other embeddings must remain None
        for key, param in module.scale_aware.scale_embeddings.items():
            if key != active_key:
                assert param.grad is None, f"Inactive embedding {key} received gradient"

    # -------------------------------------------------------------------------
    # 6-8. Gradient Flow & Numerical Stability
    # -------------------------------------------------------------------------
    def test_gradient_flow_for_p3(self) -> None:
        """
        Verify backward gradients reach all constituent components:
            - input X
            - scale_embeddings["3"]
            - scale_proj
            - spatial attention parameters
            - channel attention parameters
            - fusion Conv
            - fusion BatchNorm
        """
        module = AMSAModule(channels=128, scale_dim=64)
        x = (torch.randn(2, 128, 40, 40) + 1.0).requires_grad_(True)

        out = module(x, scale_level=3)
        loss = out.sum()
        loss.backward()

        # 1. Input X
        assert x.grad is not None
        assert not torch.all(x.grad == 0.0)

        # 2. ScaleAwareModule active embedding and projection
        assert module.scale_aware.embed_p3.grad is not None
        assert not torch.all(module.scale_aware.embed_p3.grad == 0.0)
        assert module.scale_aware.scale_proj.weight.grad is not None
        assert not torch.all(module.scale_aware.scale_proj.weight.grad == 0.0)

        # 3. Spatial attention parameters
        for name, p in module.spatial_attention.named_parameters():
            assert p.grad is not None, f"Spatial param {name} grad is None"
            assert not torch.all(p.grad == 0.0), f"Spatial param {name} grad is all zero"

        # 4. Channel attention parameters
        for name, p in module.channel_attention.named_parameters():
            assert p.grad is not None, f"Channel param {name} grad is None"
            assert not torch.all(p.grad == 0.0), f"Channel param {name} grad is all zero"

        # 5. Fusion Conv and BN parameters
        for name, p in module.fusion.named_parameters():
            assert p.grad is not None, f"Fusion param {name} grad is None"
            assert not torch.all(p.grad == 0.0), f"Fusion param {name} grad is all zero"

    def test_no_nan_or_inf_in_forward_and_backward(self) -> None:
        """Verify numerical stability with large inputs."""
        module = AMSAModule(channels=32, scale_dim=64)
        x = (torch.randn(2, 32, 16, 16) * 50.0).requires_grad_(True)

        out = module(x, scale_level=3)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

        loss = out.mean()
        loss.backward()

        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    # -------------------------------------------------------------------------
    # 9-10. Dynamic Dimensions
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("batch_size", [1, 2, 5, 8])
    def test_dynamic_batch_sizes(self, batch_size: int) -> None:
        """Verify module functions across varying batch sizes."""
        module = AMSAModule(channels=32, scale_dim=64).eval()
        x = torch.randn(batch_size, 32, 16, 16)

        out = module(x, scale_level=3)
        assert out.shape == (batch_size, 32, 16, 16)

    @pytest.mark.parametrize("spatial_size", [(17, 23), (33, 31), (80, 80), (7, 13)])
    def test_dynamic_spatial_dimensions(self, spatial_size: Tuple[int, int]) -> None:
        """Verify module handles arbitrary non-square and prime spatial dimensions."""
        H, W = spatial_size
        module = AMSAModule(channels=32, scale_dim=64)
        x = torch.randn(2, 32, H, W)

        out = module(x, scale_level=3)
        assert out.shape == (2, 32, H, W)

    # -------------------------------------------------------------------------
    # 11-14. Robust Input Validation and Error Raising
    # -------------------------------------------------------------------------
    def test_invalid_scale_level_raises_value_error(self) -> None:
        """Verify unsupported scale levels raise ValueError."""
        module = AMSAModule(channels=32, scale_dim=64)
        x = torch.randn(2, 32, 16, 16)

        for invalid in [1, 2, 6, "P2", "P6", "invalid", None, 3.5]:
            with pytest.raises(ValueError):
                module(x, scale_level=invalid)  # type: ignore

    def test_invalid_input_rank_raises_value_error(self) -> None:
        """Verify non-4D inputs raise ValueError."""
        module = AMSAModule(channels=32, scale_dim=64)

        with pytest.raises(ValueError, match="Expected 4D tensor for x"):
            module(torch.randn(32, 16, 16), scale_level=3)
        with pytest.raises(ValueError, match="Expected 4D tensor for x"):
            module(torch.randn(2, 32, 16, 16, 1), scale_level=3)

    def test_channel_mismatch_raises_value_error(self) -> None:
        """Verify mismatched channel dimension raises ValueError."""
        module = AMSAModule(channels=32, scale_dim=64)
        x = torch.randn(2, 64, 16, 16)  # 64 instead of 32

        with pytest.raises(ValueError, match="Channel mismatch: x has 64 channels"):
            module(x, scale_level=3)

    def test_non_tensor_raises_type_error(self) -> None:
        """Verify non-tensor inputs raise TypeError."""
        module = AMSAModule(channels=32, scale_dim=64)

        with pytest.raises(TypeError, match="Expected x to be a torch.Tensor"):
            module([1.0], scale_level=3)  # type: ignore

    def test_invalid_constructor_args_raise_value_error(self) -> None:
        """Verify invalid constructor arguments raise ValueError."""
        with pytest.raises(ValueError, match="channels must be a positive integer"):
            AMSAModule(channels=0)
        with pytest.raises(ValueError, match="scale_dim must be a positive integer"):
            AMSAModule(channels=32, scale_dim=-1)
        with pytest.raises(ValueError):
            AMSAModule(channels=32, scale_level=2)
        with pytest.raises(ValueError):
            AMSAModule(channels=32, scale_level="invalid")

    def test_constructor_bound_and_forward_scale_resolution(self) -> None:
        """Verify scale_level resolution between constructor and forward."""
        x = torch.randn(2, 32, 16, 16)

        # 1. Constructor-bound scale allows forward call with no scale_level: module(x)
        module_bound = AMSAModule(channels=32, scale_level=3)
        out_bound = module_bound(x)
        assert out_bound.shape == x.shape

        # 2. Old-style explicit forward call works on unbound module: module(x, scale_level=3)
        module_unbound = AMSAModule(channels=32)
        out_explicit = module_unbound(x, scale_level=3)
        assert out_explicit.shape == x.shape
        # When weights are identical, bound module(x) and unbound module(x, scale_level=3) match
        module_unbound.load_state_dict(module_bound.state_dict())
        assert torch.equal(module_bound(x), module_unbound(x, scale_level=3))

        # 3. Forward override: forward scale_level wins over constructor scale_level
        module_bound_p3 = AMSAModule(channels=32, scale_level=3)
        module_bound_p4 = AMSAModule(channels=32, scale_level=4)
        module_bound_p4.load_state_dict(module_bound_p3.state_dict())
        # Overriding p3 module with scale_level=4 must match p4 module called with default
        out_override = module_bound_p3(x, scale_level=4)
        out_p4_default = module_bound_p4(x)
        assert torch.equal(out_override, out_p4_default)

        # 4. Missing scale level raises ValueError when neither constructor nor forward provides it
        module_missing = AMSAModule(channels=32)
        with pytest.raises(ValueError, match="scale_level must be specified"):
            module_missing(x)
        with pytest.raises(ValueError, match="scale_level must be specified"):
            module_missing(x, scale_level=None)

    def test_registration_in_ultralytics(self) -> None:
        """Verify AMSAModule is registered into ultralytics.nn.tasks and modules."""
        import ultralytics.nn.tasks as tasks
        import ultralytics.nn.modules as modules
        from src.amsa import register_amsa
        register_amsa()
        assert getattr(tasks, "AMSAModule", None) is AMSAModule
        assert getattr(modules, "AMSAModule", None) is AMSAModule


    # -------------------------------------------------------------------------
    # 15-17. Hardware Contracts, Evaluation Determinism & State Dict
    # -------------------------------------------------------------------------
    def test_pure_cpu_execution(self) -> None:
        """Verify clean CPU execution."""
        module = AMSAModule(channels=32, scale_dim=64)
        x = torch.randn(2, 32, 16, 16)

        out = module(x, scale_level=3)
        assert out.device.type == "cpu"

    def test_deterministic_eval_execution(self) -> None:
        """Verify deterministic forward evaluation."""
        module = AMSAModule(channels=32, scale_dim=64).eval()
        x = torch.randn(2, 32, 16, 16)

        with torch.no_grad():
            out1 = module(x, scale_level=3)
            out2 = module(x, scale_level=3)

        assert torch.equal(out1, out2)

    def test_state_dict_cleanliness_and_no_aliases(self) -> None:
        """Verify state dict contains all expected sub-module keys without rogue parameters."""
        module = AMSAModule(channels=32, scale_dim=64)
        keys = list(module.state_dict().keys())

        # Check prefixes
        assert any(k.startswith("scale_aware.") for k in keys)
        assert any(k.startswith("spatial_attention.") for k in keys)
        assert any(k.startswith("channel_attention.") for k in keys)
        assert any(k.startswith("fusion.") for k in keys)

        # Distinct memory addresses
        params = list(module.parameters())
        param_ptrs = {p.data_ptr() for p in params}
        assert len(params) == len(param_ptrs), "Duplicate memory addresses found among parameters"

    # -------------------------------------------------------------------------
    # 18. Parameter Count Verification Against Submodule Sums
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        ("channels", "scale_dim", "expected_total"),
        [
            # YOLOv8s canonical levels:
            # P3 (C=128): 4352 + 42194 + 5400 + 33152 = 85,098
            (128, 64, 85098),
            # P4 (C=256): 4352 + 165234 + 13856 + 131840 = 315,282
            (256, 64, 315282),
            # P5 (C=512): 4352 + 657074 + 43056 + 525824 = 1,230,306
            (512, 64, 1230306),
        ],
    )
    def test_parameter_count_matches_submodule_sum(
        self, channels: int, scale_dim: int, expected_total: int
    ) -> None:
        """Verify wrapper parameter count strictly equals the sum of its sub-modules."""
        module = AMSAModule(channels=channels, scale_dim=scale_dim)

        submodule_sum = (
            sum(p.numel() for p in module.scale_aware.parameters())
            + sum(p.numel() for p in module.spatial_attention.parameters())
            + sum(p.numel() for p in module.channel_attention.parameters())
            + sum(p.numel() for p in module.fusion.parameters())
        )
        total_params = sum(p.numel() for p in module.parameters())

        assert total_params == submodule_sum
        assert total_params == expected_total

    # -------------------------------------------------------------------------
    # 19. Mixed Precision (AMP) Autocast Compatibility
    # -------------------------------------------------------------------------
    def test_autocast_cpu_bfloat16(self) -> None:
        """Verify CPU autocast bfloat16 compatibility."""
        module = AMSAModule(channels=32, scale_dim=64)
        x = torch.randn(2, 32, 16, 16)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = module(x, scale_level=3)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_autocast_cuda_float16(self) -> None:
        """Verify CUDA float16 autocast compatibility if CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA device not available for float16 autocast testing")

        device = torch.device("cuda")
        module = AMSAModule(channels=32, scale_dim=64).to(device)
        x = torch.randn(2, 32, 16, 16, device=device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = module(x, scale_level=3)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    # -------------------------------------------------------------------------
    # 20-22. API Return Types & Serialization
    # -------------------------------------------------------------------------
    def test_api_return_types(self) -> None:
        """Verify return type behavior for return_intermediates flag."""
        module = AMSAModule(channels=32, scale_dim=64)
        x = torch.randn(2, 32, 16, 16)

        # Default: pure Tensor
        res_default = module(x, scale_level=3)
        assert isinstance(res_default, torch.Tensor)
        assert not isinstance(res_default, tuple)

        # return_intermediates=True: Tuple[Tensor, Dict[str, Tensor]]
        res_debug = module(x, scale_level=3, return_intermediates=True)
        assert isinstance(res_debug, tuple)
        assert len(res_debug) == 2
        out, intermediates = res_debug
        assert isinstance(out, torch.Tensor)
        assert isinstance(intermediates, dict)
        assert set(intermediates.keys()) == {"scale_encoding", "x_spatial", "x_channel"}

    def test_serialization_state_dict_roundtrip(self) -> None:
        """Verify state_dict save and load roundtrip produces bitwise identical eval output."""
        module1 = AMSAModule(channels=32, scale_dim=64).eval()
        module2 = AMSAModule(channels=32, scale_dim=64).eval()

        buffer = io.BytesIO()
        torch.save(module1.state_dict(), buffer)
        buffer.seek(0)
        module2.load_state_dict(torch.load(buffer, weights_only=True))

        x = torch.randn(2, 32, 16, 16)
        with torch.no_grad():
            out1 = module1(x, scale_level=3)
            out2 = module2(x, scale_level=3)

        assert torch.equal(out1, out2)
