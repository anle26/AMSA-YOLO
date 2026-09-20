"""
Unit and Integration Tests for Adaptive Spatial Attention Module.

Validates:
1. Canonical YOLOv8s / AMSA-YOLO feature shapes:
   - P3: C=128, 80x80 (Ultralytics 8.2.103 YOLOv8s Backbone Layer 4)
   - P4: C=256, 40x40 (Ultralytics 8.2.103 YOLOv8s Backbone Layer 6)
   - P5: C=512, 20x20 (Ultralytics 8.2.103 YOLOv8s Backbone Layer 9)
2. Internal tensor shapes (W_local, W_global, A_local, A_global, A_spatial).
3. Sigmoid value range bounds [0, 1].
4. Strict mathematical fusion equation fidelity:
   X_spatial = X * (W_local * A_local + W_global * A_global).
5. Dynamic batch sizes and spatial dimensions.
6. Gradient flow and numerical stability (X, S, module weights, no NaN/Inf).
7. Determinism in evaluation mode.
8. Robust input validation and error raising (rank, channels, scale_dim, batch/spatial mismatch).
9. Device contracts (CPU, CUDA if available, device mismatch raises RuntimeError).
10. Automatic Mixed Precision (AMP) / autocast compatibility (CPU bfloat16, CUDA float16).
11. State dict cleanliness (no duplicate module aliases or rogue parameters).
12. Parameter count exact formula sanity check across verified YOLOv8s configurations.
13. Integration test connecting ScaleAwareModule -> S -> AdaptiveSpatialAttention.
"""

from typing import Tuple
import pytest
import torch

from src.amsa.scale_aware import ScaleAwareModule
from src.amsa.spatial_attention import AdaptiveSpatialAttention


class TestAdaptiveSpatialAttention:
    """Test suite for AdaptiveSpatialAttention standalone module."""

    # -------------------------------------------------------------------------
    # 1-3. Mandatory Base Tests: Shapes and Range Bounds
    # -------------------------------------------------------------------------
    def test_output_shape_equals_input_shape(self) -> None:
        """Verify that output tensor preserves the exact shape of input X."""
        module = AdaptiveSpatialAttention(channels=64, scale_dim=64)
        x = torch.randn(3, 64, 32, 48)
        s = torch.randn(3, 64, 32, 48)

        out = module(x, s)
        assert out.shape == x.shape
        assert isinstance(out, torch.Tensor)

    def test_internal_tensor_shapes(self) -> None:
        """
        Verify internal attention and weight shapes:
            W_local   == [B, 1, 1, 1]
            W_global  == [B, 1, 1, 1]
            A_local   == [B, C, H, W]
            A_global  == [B, C, 1, 1]
            A_spatial == [B, C, H, W]
        """
        B, C, D, H, W = 4, 32, 64, 28, 28
        module = AdaptiveSpatialAttention(channels=C, scale_dim=D)
        x = torch.randn(B, C, H, W)
        s = torch.randn(B, D, H, W)

        out, maps = module(x, s, return_attention_maps=True)

        assert out.shape == (B, C, H, W)
        assert maps["w_local"].shape == (B, 1, 1, 1)
        assert maps["w_global"].shape == (B, 1, 1, 1)
        assert maps["a_local"].shape == (B, C, H, W)
        assert maps["a_global"].shape == (B, C, 1, 1)
        assert maps["a_spatial"].shape == (B, C, H, W)

    def test_all_sigmoid_outputs_within_zero_one(self) -> None:
        """Verify that all sigmoid-activated attention weights and maps lie in [0, 1]."""
        module = AdaptiveSpatialAttention(channels=48, scale_dim=64)
        x = torch.randn(2, 48, 24, 24) * 10.0  # Large inputs to stress test sigmoid saturation
        s = torch.randn(2, 64, 24, 24) * 10.0

        _, maps = module(x, s, return_attention_maps=True)

        for key in ("w_local", "w_global", "a_local", "a_global"):
            tensor = maps[key]
            assert torch.all(tensor >= 0.0), f"{key} contains values < 0.0: min={tensor.min()}"
            assert torch.all(tensor <= 1.0), f"{key} contains values > 1.0: max={tensor.max()}"

    # -------------------------------------------------------------------------
    # 4-6. Canonical Feature Pyramid Shapes (P3, P4, P5)
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        ("channels", "spatial_size", "label"),
        [
            (128, (80, 80), "YOLOv8s P3 backbone output"),
            (256, (40, 40), "YOLOv8s P4 backbone output"),
            (512, (20, 20), "YOLOv8s P5 backbone output"),
            (64, (80, 80), "P3 small configuration"),
            (256, (80, 80), "generic 256-channel P3-like configuration"),
            (1024, (20, 20), "generic 1024-channel P5-like configuration"),
        ],
    )
    def test_pyramid_scale_preservation(self, channels: int, spatial_size: Tuple[int, int], label: str) -> None:
        """Verify feature shape preservation across P3, P4, and P5 dimensions."""
        H, W = spatial_size
        scale_dim = 64
        module = AdaptiveSpatialAttention(channels=channels, scale_dim=scale_dim)

        x = torch.randn(2, channels, H, W)
        s = torch.randn(2, scale_dim, H, W)

        out = module(x, s)
        assert out.shape == (2, channels, H, W)

    # -------------------------------------------------------------------------
    # 7. Explicit Fusion Equation Fidelity
    # -------------------------------------------------------------------------
    def test_explicit_fusion_equation_fidelity(self) -> None:
        """
        Verify that module output strictly equals:
            A_spatial = (W_local * A_local) + (W_global * A_global)
            X_spatial = X * A_spatial
        """
        module = AdaptiveSpatialAttention(channels=32, scale_dim=64)
        x = torch.randn(2, 32, 16, 16)
        s = torch.randn(2, 64, 16, 16)

        # Compute via module
        out, maps = module(x, s, return_attention_maps=True)

        w_local = maps["w_local"]
        w_global = maps["w_global"]
        a_local = maps["a_local"]
        a_global = maps["a_global"]

        # Manual computation via paper formula
        expected_a_spatial = (w_local * a_local) + (w_global * a_global)
        expected_out = x * expected_a_spatial

        assert torch.allclose(maps["a_spatial"], expected_a_spatial, atol=1e-7, rtol=1e-6)
        assert torch.allclose(out, expected_out, atol=1e-7, rtol=1e-6)

    # -------------------------------------------------------------------------
    # 8-10. Gradient Flow & Numerical Stability
    # -------------------------------------------------------------------------
    def test_gradient_flow_through_x_and_s(self) -> None:
        """Verify non-zero gradients propagate back through both input X and scale tensor S."""
        module = AdaptiveSpatialAttention(channels=32, scale_dim=64)
        x = torch.randn(2, 32, 16, 16, requires_grad=True)
        s = torch.randn(2, 64, 16, 16, requires_grad=True)

        out = module(x, s)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.all(torch.isfinite(x.grad))
        assert x.grad.abs().sum() > 0.0

        assert s.grad is not None
        assert s.grad.shape == s.shape
        assert torch.all(torch.isfinite(s.grad))
        assert s.grad.abs().sum() > 0.0

    def test_gradient_flow_through_module_parameters(self) -> None:
        """Verify all learnable parameters in scale_mlp and local_conv receive valid gradients."""
        module = AdaptiveSpatialAttention(channels=32, scale_dim=64)
        x = torch.randn(2, 32, 16, 16)
        s = torch.randn(2, 64, 16, 16)

        out = module(x, s)
        loss = (out ** 2).sum()
        loss.backward()

        for name, param in module.named_parameters():
            assert param.grad is not None, f"Parameter {name} did not receive a gradient!"
            assert torch.all(torch.isfinite(param.grad)), f"Parameter {name} received non-finite gradient!"
            assert param.grad.abs().sum() > 0.0, f"Parameter {name} received zero gradient!"

    def test_no_nan_or_inf_in_forward_and_backward(self) -> None:
        """Verify stability with extreme input values (zeros, large positive/negative values)."""
        module = AdaptiveSpatialAttention(channels=16, scale_dim=32)

        # Extreme values with matching spatial shape (2, 16, 8, 8)
        x = (torch.randn(2, 16, 8, 8) * 1e4).requires_grad_(True)
        s = (torch.randn(2, 32, 8, 8) * 1e4).requires_grad_(True)

        out = module(x, s)
        assert torch.all(torch.isfinite(out)), "Forward output contains NaN or Inf!"

        loss = out.mean()
        loss.backward()
        assert torch.all(torch.isfinite(x.grad)), "x.grad contains NaN or Inf!"
        assert torch.all(torch.isfinite(s.grad)), "s.grad contains NaN or Inf!"

    # -------------------------------------------------------------------------
    # 11. Determinism in Eval Mode
    # -------------------------------------------------------------------------
    def test_deterministic_forward_in_eval_mode(self) -> None:
        """Verify identical outputs across multiple evaluations with eval() mode."""
        module = AdaptiveSpatialAttention(channels=32, scale_dim=64)
        module.eval()

        x = torch.randn(2, 32, 20, 20)
        s = torch.randn(2, 64, 20, 20)

        with torch.no_grad():
            out1 = module(x, s)
            out2 = module(x, s)

        assert torch.equal(out1, out2)

    # -------------------------------------------------------------------------
    # 12-15. Robust Input Validation and Error Raising
    # -------------------------------------------------------------------------
    def test_invalid_input_rank_raises_value_error(self) -> None:
        """Verify 3D or 5D inputs raise informative ValueError."""
        module = AdaptiveSpatialAttention(channels=32, scale_dim=64)

        # 3D x
        with pytest.raises(ValueError, match="Expected 4D tensor for x"):
            module(torch.randn(32, 16, 16), torch.randn(1, 64, 16, 16))

        # 5D s
        with pytest.raises(ValueError, match="Expected 4D tensor for s"):
            module(torch.randn(1, 32, 16, 16), torch.randn(1, 1, 64, 16, 16))

    def test_channel_mismatch_raises_value_error(self) -> None:
        """Verify channel mismatch between x and configured module raises ValueError."""
        module = AdaptiveSpatialAttention(channels=32, scale_dim=64)
        x_wrong_c = torch.randn(1, 64, 16, 16)
        s = torch.randn(1, 64, 16, 16)

        with pytest.raises(ValueError, match="Channel mismatch: x has 64 channels, but module was configured with channels=32"):
            module(x_wrong_c, s)

    def test_scale_dim_mismatch_raises_value_error(self) -> None:
        """Verify scale_dim mismatch between s and configured module raises ValueError."""
        module = AdaptiveSpatialAttention(channels=32, scale_dim=64)
        x = torch.randn(1, 32, 16, 16)
        s_wrong_d = torch.randn(1, 32, 16, 16)

        with pytest.raises(ValueError, match="Scale dim mismatch: s has 32 scale dimension, but module was configured with scale_dim=64"):
            module(x, s_wrong_d)

    def test_batch_and_spatial_mismatch_raises_value_error(self) -> None:
        """Verify batch size or spatial resolution mismatch between X and S raises ValueError."""
        module = AdaptiveSpatialAttention(channels=32, scale_dim=64)

        # Batch mismatch
        with pytest.raises(ValueError, match="Batch size mismatch"):
            module(torch.randn(2, 32, 16, 16), torch.randn(4, 64, 16, 16))

        # Spatial mismatch
        with pytest.raises(ValueError, match="Spatial resolution mismatch"):
            module(torch.randn(2, 32, 16, 16), torch.randn(2, 64, 20, 20))

    def test_non_tensor_raises_type_error(self) -> None:
        """Verify non-tensor inputs raise TypeError."""
        module = AdaptiveSpatialAttention(channels=32, scale_dim=64)
        s = torch.randn(1, 64, 16, 16)

        with pytest.raises(TypeError, match="Expected x to be a torch.Tensor"):
            module([1, 2, 3], s)

        with pytest.raises(TypeError, match="Expected s to be a torch.Tensor"):
            module(s, [1, 2, 3])

    def test_invalid_constructor_args_raise_value_error(self) -> None:
        """Verify non-positive dimensions raise ValueError in __init__."""
        with pytest.raises(ValueError, match="channels must be a positive integer"):
            AdaptiveSpatialAttention(channels=0)

        with pytest.raises(ValueError, match="scale_dim must be a positive integer"):
            AdaptiveSpatialAttention(channels=32, scale_dim=-1)

        with pytest.raises(ValueError, match="local_reduction must be a positive integer"):
            AdaptiveSpatialAttention(channels=32, local_reduction=0)

        with pytest.raises(ValueError, match="scale_reduction must be a positive integer"):
            AdaptiveSpatialAttention(channels=32, scale_reduction=-2)

    # -------------------------------------------------------------------------
    # 16-17. Device Contracts & Multi-Precision Execution
    # -------------------------------------------------------------------------
    def test_pure_cpu_execution(self) -> None:
        """Verify module executes on CPU with clean tensor returns."""
        module = AdaptiveSpatialAttention(channels=16, scale_dim=32)
        x = torch.randn(2, 16, 10, 10, device="cpu")
        s = torch.randn(2, 32, 10, 10, device="cpu")

        out = module(x, s)
        assert out.device.type == "cpu"

    def test_device_contract_and_float64(self) -> None:
        """Verify device matching and float64 execution."""
        # Double precision module execution
        module_double = AdaptiveSpatialAttention(channels=16, scale_dim=32).double()
        x_double = torch.randn(2, 16, 8, 8, dtype=torch.float64)
        s_double = torch.randn(2, 32, 8, 8, dtype=torch.float64)

        out_double = module_double(x_double, s_double)
        assert out_double.dtype == torch.float64

        # Cross-device mismatch verification (if CUDA available)
        if torch.cuda.is_available():
            module_cuda = AdaptiveSpatialAttention(channels=16, scale_dim=32).cuda()
            x_cpu = torch.randn(2, 16, 8, 8, device="cpu")
            s_cuda = torch.randn(2, 32, 8, 8, device="cuda")

            with pytest.raises(RuntimeError, match="Device mismatch: x is on cpu"):
                module_cuda(x_cpu, s_cuda)

            with pytest.raises(RuntimeError, match="Device mismatch: s is on cpu"):
                module_cuda(s_cuda, x_cpu)

    # -------------------------------------------------------------------------
    # 18. AMP (Automatic Mixed Precision) / Autocast Compatibility
    # -------------------------------------------------------------------------
    def test_autocast_cpu_bfloat16(self) -> None:
        """
        Verify that module runs safely under torch.autocast on CPU with bfloat16.
        Ensures float32 master weights work seamlessly with bfloat16 activations.
        """
        module = AdaptiveSpatialAttention(channels=128, scale_dim=64)
        x_bf16 = torch.randn(2, 128, 40, 40, dtype=torch.bfloat16, requires_grad=True)
        s_bf16 = torch.randn(2, 64, 40, 40, dtype=torch.bfloat16, requires_grad=True)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = module(x_bf16, s_bf16)

        assert out.shape == (2, 128, 40, 40)
        assert out.dtype == torch.bfloat16

        loss = out.sum()
        loss.backward()

        assert x_bf16.grad is not None
        assert s_bf16.grad is not None
        assert torch.all(torch.isfinite(x_bf16.grad))
        assert torch.all(torch.isfinite(s_bf16.grad))

    def test_autocast_chained_path_cpu(self) -> None:
        """
        Verify chained ScaleAwareModule -> S -> AdaptiveSpatialAttention under CPU autocast.
        ScaleAwareModule conv projection outputs bfloat16 S, which flows directly into AdaptiveSpatialAttention.
        """
        sam = ScaleAwareModule(embed_dim=64)
        asa = AdaptiveSpatialAttention(channels=128, scale_dim=64)
        x = torch.randn(2, 128, 40, 40, requires_grad=True)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            s = sam(x, scale_level=3)
            out = asa(x, s)

        assert out.shape == (2, 128, 40, 40)
        loss = out.sum()
        loss.backward()

        assert sam.scale_embeddings["3"].grad is not None
        assert torch.all(torch.isfinite(sam.scale_embeddings["3"].grad))
        assert asa.local_conv1.weight.grad is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for CUDA autocast test")
    def test_autocast_cuda_float16(self) -> None:
        """
        Verify module and chained execution under torch.autocast on CUDA with float16.
        Skips cleanly when CUDA is not present.
        """
        sam = ScaleAwareModule(embed_dim=64).cuda()
        asa = AdaptiveSpatialAttention(channels=128, scale_dim=64).cuda()
        x = torch.randn(2, 128, 40, 40, device="cuda", requires_grad=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            s = sam(x, scale_level=3)
            out = asa(x, s)

        assert out.shape == (2, 128, 40, 40)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert sam.scale_embeddings["3"].grad is not None
        assert asa.local_conv1.weight.grad is not None

    # -------------------------------------------------------------------------
    # 19. State Dict Cleanliness (No duplicate aliases or rogue parameters)
    # -------------------------------------------------------------------------
    def test_state_dict_cleanliness_and_no_aliases(self) -> None:
        """
        Verify state_dict contains exactly the expected parameter keys:
            scale_mlp.0.weight, scale_mlp.0.bias
            scale_mlp.2.weight, scale_mlp.2.bias
            local_conv1.weight, local_conv1.bias
            local_conv2.weight, local_conv2.bias
        """
        module = AdaptiveSpatialAttention(channels=64, scale_dim=64)
        state_keys = set(module.state_dict().keys())

        expected_keys = {
            "scale_mlp.0.weight",
            "scale_mlp.0.bias",
            "scale_mlp.2.weight",
            "scale_mlp.2.bias",
            "local_conv1.weight",
            "local_conv1.bias",
            "local_conv2.weight",
            "local_conv2.bias",
        }
        assert state_keys == expected_keys, f"Discrepancy in state_dict keys: {state_keys ^ expected_keys}"

    # -------------------------------------------------------------------------
    # 20. Parameter Count Exact Sanity Formula (Including YOLOv8s Channels)
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        ("channels", "scale_dim", "local_reduction", "scale_reduction", "bias", "expected_count"),
        [
            # Verified YOLOv8s channels:
            (128, 64, 4, 4, True, 42194),    # YOLOv8s P3 (C=128, D=64)
            (256, 64, 4, 4, True, 165234),   # YOLOv8s P4 (C=256, D=64)
            (512, 64, 4, 4, True, 657074),   # YOLOv8s P5 (C=512, D=64)
            # Other configurations:
            (64, 64, 4, 4, True, 11394),     # Canonical baseline
            (1024, 64, 8, 4, True, 1312946), # Large width with lr=8
            (32, 16, 2, 2, False, 5264),     # Bias=False configuration
        ],
    )
    def test_parameter_count_matches_theoretical_formula(
        self,
        channels: int,
        scale_dim: int,
        local_reduction: int,
        scale_reduction: int,
        bias: bool,
        expected_count: int,
    ) -> None:
        """
        Verify exact parameter count formula:
            D_hidden = max(1, scale_dim // scale_reduction)
            C_reduced = max(1, channels // local_reduction)
            Scale branch:
                Conv1 (D -> D_hidden, 1x1): D_hidden * scale_dim + (D_hidden if bias else 0)
                Conv2 (D_hidden -> 2, 1x1): 2 * D_hidden + (2 if bias else 0)
            Local branch:
                Conv1 (C -> C_reduced, 1x1): C_reduced * channels + (C_reduced if bias else 0)
                Conv2 (C_reduced -> C, 3x3): channels * C_reduced * 9 + (channels if bias else 0)
            Global branch: 0 params
        """
        module = AdaptiveSpatialAttention(
            channels=channels,
            scale_dim=scale_dim,
            local_reduction=local_reduction,
            scale_reduction=scale_reduction,
            bias=bias,
        )

        d_hidden = max(1, scale_dim // scale_reduction)
        c_reduced = max(1, channels // local_reduction)

        # Scale branch params
        scale_p1 = d_hidden * scale_dim + (d_hidden if bias else 0)
        scale_p2 = 2 * d_hidden + (2 if bias else 0)
        scale_params = scale_p1 + scale_p2

        # Local branch params
        local_p1 = c_reduced * channels + (c_reduced if bias else 0)
        local_p2 = channels * c_reduced * 9 + (channels if bias else 0)
        local_params = local_p1 + local_p2

        expected_total = scale_params + local_params
        actual_total = sum(p.numel() for p in module.parameters())

        assert expected_total == expected_count, (
            f"Param formula discrepancy: formula gives {expected_total}, expected {expected_count}"
        )
        assert actual_total == expected_total, (
            f"Param mismatch: expected {expected_total}, got {actual_total}"
        )

    # -------------------------------------------------------------------------
    # 21. End-to-End Integration Test: ScaleAwareModule -> S -> AdaptiveSpatialAttention
    # -------------------------------------------------------------------------
    def test_integration_scale_aware_to_spatial_attention(self) -> None:
        """
        Integration test verifying pipeline connection with YOLOv8s P3 dimensions:
            ScaleAwareModule(embed_dim=64)
                |
                v
                S in R^(2 x 64 x 80 x 80)
                |
                v
            AdaptiveSpatialAttention(channels=128, scale_dim=64)
                |
                v
            X_spatial in R^(2 x 128 x 80 x 80)
        """
        B, C, D, H, W = 2, 128, 64, 80, 80
        scale_aware = ScaleAwareModule(embed_dim=D)
        spatial_att = AdaptiveSpatialAttention(channels=C, scale_dim=D)

        x = torch.randn(B, C, H, W, requires_grad=True)

        # 1. Scale-aware forward pass at level P3 (s=3)
        s = scale_aware(x, scale_level=3)
        assert s.shape == (B, D, H, W)

        # 2. Adaptive spatial attention forward pass
        out = spatial_att(x, s)
        assert out.shape == (B, C, H, W)

        # 3. Backward pass
        loss = out.sum()
        loss.backward()

        # Verify input X gradient
        assert x.grad is not None
        assert x.grad.shape == (B, C, H, W)
        assert x.grad.abs().sum() > 0.0

        # Verify ScaleAwareModule gradients: level '3' must have gradient, '4' and '5' must not
        assert scale_aware.scale_embeddings["3"].grad is not None
        assert scale_aware.scale_embeddings["3"].grad.abs().sum() > 0.0
        assert scale_aware.scale_embeddings["4"].grad is None
        assert scale_aware.scale_embeddings["5"].grad is None
        assert scale_aware.scale_proj.weight.grad is not None

        # Verify AdaptiveSpatialAttention gradients
        assert spatial_att.local_conv1.weight.grad is not None
        assert spatial_att.local_conv2.weight.grad is not None
        assert spatial_att.scale_mlp[0].weight.grad is not None
        assert spatial_att.scale_mlp[2].weight.grad is not None
