"""
Unit and Integration Tests for Feature Fusion Module.

Validates:
1. Canonical YOLOv8s / AMSA-YOLO feature shapes:
   - P3: C=128, 80x80 (Ultralytics 8.2.103 YOLOv8s Backbone Layer 4)
   - P4: C=256, 40x40 (Ultralytics 8.2.103 YOLOv8s Backbone Layer 6)
   - P5: C=512, 20x20 (Ultralytics 8.2.103 YOLOv8s Backbone Layer 9)
2. Internal tensor shapes:
   - X_concat == [B, 2C, H, W]
   - X_fused  == [B, C, H, W]
   - X_bn     == [B, C, H, W]
   - X_output == [B, C, H, W]
3. Strict mathematical paper equation fidelity:
   X_output = ReLU(BN(Conv1x1(Concat(X_channel, X_spatial))) + X)
4. Concatenation order verification: [X_channel, X_spatial]
5. Residual addition verification: original X added after BN
6. Non-negativity constraint from ReLU (X_output >= 0)
7. Dynamic batch sizes and spatial dimensions
8. Gradient flow through X, X_channel, X_spatial, Conv1x1, and BatchNorm without NaN/Inf
9. Deterministic evaluation forward pass
10. Robust input validation and error raising (rank, channels, batch, spatial mismatch, non-tensor)
11. Device contracts (CPU, CUDA if available, device mismatch raises RuntimeError)
12. Automatic Mixed Precision (AMP) / autocast compatibility (CPU bfloat16, CUDA float16)
13. State dict cleanliness
14. Exact parameter count formula verification across verified YOLOv8s configurations
15. Multi-module integration test connecting:
    ScaleAwareModule -> S
    AdaptiveSpatialAttention(X, S) -> X_spatial
    AdaptiveChannelAttention(X, S) -> X_channel
    FeatureFusion(X, X_channel, X_spatial) -> X_output
"""

from typing import Tuple
import pytest
import torch
import torch.nn as nn

from src.amsa.channel_attention import AdaptiveChannelAttention
from src.amsa.fusion import FeatureFusion
from src.amsa.scale_aware import ScaleAwareModule
from src.amsa.spatial_attention import AdaptiveSpatialAttention


def compute_expected_params(channels: int, bias: bool = True) -> int:
    """Analytical parameter count formula for FeatureFusion."""
    # Conv1x1: weight = channels * (2 * channels), bias = channels if bias else 0
    conv_w = channels * (2 * channels)
    conv_b = channels if bias else 0
    # BatchNorm2d: weight = channels, bias = channels
    bn_w = channels
    bn_b = channels
    return conv_w + conv_b + bn_w + bn_b


class TestFeatureFusion:
    """Test suite for FeatureFusion standalone module."""

    # -------------------------------------------------------------------------
    # 1-3. Output and Internal Tensor Shapes
    # -------------------------------------------------------------------------
    def test_output_shape_equals_input_shape(self) -> None:
        """Verify that output tensor preserves the exact shape of input X."""
        module = FeatureFusion(channels=64)
        x = torch.randn(3, 64, 32, 48)
        x_channel = torch.randn(3, 64, 32, 48)
        x_spatial = torch.randn(3, 64, 32, 48)

        out = module(x, x_channel, x_spatial)
        assert out.shape == x.shape
        assert isinstance(out, torch.Tensor)

    def test_internal_tensor_shapes(self) -> None:
        """
        Verify internal tensor shapes:
            X_concat == [B, 2C, H, W]
            X_fused  == [B, C, H, W]
            X_bn     == [B, C, H, W]
            X_output == [B, C, H, W]
        """
        B, C, H, W = 4, 32, 28, 28
        module = FeatureFusion(channels=C)
        x = torch.randn(B, C, H, W)
        x_channel = torch.randn(B, C, H, W)
        x_spatial = torch.randn(B, C, H, W)

        out, intermediates = module(x, x_channel, x_spatial, return_intermediates=True)

        assert out.shape == (B, C, H, W)
        assert intermediates["x_concat"].shape == (B, 2 * C, H, W)
        assert intermediates["x_fused"].shape == (B, C, H, W)
        assert intermediates["x_bn"].shape == (B, C, H, W)
        assert intermediates["x_residual"].shape == (B, C, H, W)
        assert intermediates["x_output"].shape == (B, C, H, W)

    # -------------------------------------------------------------------------
    # 4-6. Mathematical Paper Equation & Residual Fidelity
    # -------------------------------------------------------------------------
    def test_explicit_paper_equation_fidelity(self) -> None:
        """
        Verify paper equation:
            X_output = ReLU(BN(Conv1x1(Concat(X_channel, X_spatial))) + X)
        """
        module = FeatureFusion(channels=32, bias=True)
        x = torch.randn(2, 32, 16, 16)
        x_channel = torch.randn(2, 32, 16, 16)
        x_spatial = torch.randn(2, 32, 16, 16)

        out, intermediates = module(x, x_channel, x_spatial, return_intermediates=True)

        # Recompute step-by-step
        concat_ref = torch.cat([x_channel, x_spatial], dim=1)
        fused_ref = module.conv(concat_ref)
        bn_ref = module.bn(fused_ref)
        residual_ref = bn_ref + x
        expected_out = module.relu(residual_ref)

        assert torch.allclose(intermediates["x_concat"], concat_ref, atol=1e-6)
        assert torch.allclose(intermediates["x_fused"], fused_ref, atol=1e-6)
        assert torch.allclose(intermediates["x_bn"], bn_ref, atol=1e-6)
        assert torch.allclose(intermediates["x_residual"], residual_ref, atol=1e-6)
        assert torch.allclose(out, expected_out, atol=1e-6)

    def test_concat_order_verification(self) -> None:
        """
        Verify concatenation order is [X_channel, X_spatial] along dim=1.
        First C channels of X_concat must correspond to X_channel,
        and last C channels must correspond to X_spatial.
        """
        module = FeatureFusion(channels=16)
        x = torch.randn(2, 16, 8, 8)
        x_channel = torch.randn(2, 16, 8, 8)
        x_spatial = torch.randn(2, 16, 8, 8)

        _, intermediates = module(x, x_channel, x_spatial, return_intermediates=True)
        concat = intermediates["x_concat"]

        assert torch.equal(concat[:, :16, :, :], x_channel)
        assert torch.equal(concat[:, 16:, :, :], x_spatial)

    def test_residual_path_incorporates_original_x(self) -> None:
        """Verify original X is directly added after batch normalization."""
        module = FeatureFusion(channels=16)
        x = torch.randn(2, 16, 8, 8)
        x_channel = torch.randn(2, 16, 8, 8)
        x_spatial = torch.randn(2, 16, 8, 8)

        _, intermediates = module(x, x_channel, x_spatial, return_intermediates=True)

        assert torch.allclose(intermediates["x_residual"], intermediates["x_bn"] + x, atol=1e-6)

    def test_relu_non_negativity_constraint(self) -> None:
        """Verify output tensor satisfies ReLU non-negativity constraint (X_output >= 0)."""
        module = FeatureFusion(channels=32)
        # Large negative inputs to ensure ReLU activation clamping is tested
        x = torch.randn(2, 32, 16, 16) * -50.0
        x_channel = torch.randn(2, 32, 16, 16) * -50.0
        x_spatial = torch.randn(2, 32, 16, 16) * -50.0

        out = module(x, x_channel, x_spatial)
        assert torch.all(out >= 0.0), f"Found negative values in ReLU output: min={out.min()}"

    # -------------------------------------------------------------------------
    # 7-9. Feature Pyramid Canonical Shapes (P3, P4, P5)
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
    def test_pyramid_scale_preservation(
        self, channels: int, spatial_size: Tuple[int, int], label: str
    ) -> None:
        """Verify feature shape preservation across canonical P3, P4, and P5 dimensions."""
        H, W = spatial_size
        module = FeatureFusion(channels=channels)
        x = torch.randn(2, channels, H, W)
        x_channel = torch.randn(2, channels, H, W)
        x_spatial = torch.randn(2, channels, H, W)

        out = module(x, x_channel, x_spatial)
        assert out.shape == (2, channels, H, W), f"Failed for {label}"

    # -------------------------------------------------------------------------
    # 10-11. Dynamic Batch Sizes and Spatial Dimensions
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("batch_size", [1, 2, 5, 8])
    def test_dynamic_batch_sizes(self, batch_size: int) -> None:
        """Verify module functions across varying batch sizes."""
        module = FeatureFusion(channels=32)
        # In eval mode or with spatial > 1, batch_size=1 executes cleanly through BatchNorm
        module.eval()
        x = torch.randn(batch_size, 32, 16, 16)
        x_channel = torch.randn(batch_size, 32, 16, 16)
        x_spatial = torch.randn(batch_size, 32, 16, 16)

        out = module(x, x_channel, x_spatial)
        assert out.shape == (batch_size, 32, 16, 16)

    @pytest.mark.parametrize("spatial_size", [(17, 23), (33, 31), (80, 80), (7, 13)])
    def test_dynamic_spatial_dimensions(self, spatial_size: Tuple[int, int]) -> None:
        """Verify module handles arbitrary non-square and prime spatial dimensions."""
        H, W = spatial_size
        module = FeatureFusion(channels=32)
        x = torch.randn(2, 32, H, W)
        x_channel = torch.randn(2, 32, H, W)
        x_spatial = torch.randn(2, 32, H, W)

        out = module(x, x_channel, x_spatial)
        assert out.shape == (2, 32, H, W)

    # -------------------------------------------------------------------------
    # 12-16. Gradient Flow & Numerical Stability
    # -------------------------------------------------------------------------
    def test_gradient_flow_through_all_inputs(self) -> None:
        """Verify gradients propagate back to X, X_channel, and X_spatial."""
        module = FeatureFusion(channels=32)
        x = (torch.randn(2, 32, 16, 16) + 2.0).requires_grad_(True)
        x_channel = (torch.randn(2, 32, 16, 16) + 2.0).requires_grad_(True)
        x_spatial = (torch.randn(2, 32, 16, 16) + 2.0).requires_grad_(True)

        out = module(x, x_channel, x_spatial)
        loss = out.sum()
        loss.backward()

        for tensor_name, tensor in [("x", x), ("x_channel", x_channel), ("x_spatial", x_spatial)]:
            assert tensor.grad is not None, f"Gradient is None for {tensor_name}"
            assert not torch.all(tensor.grad == 0.0), f"Gradient is all zeros for {tensor_name}"
            assert not torch.isnan(tensor.grad).any(), f"Gradient has NaN for {tensor_name}"
            assert not torch.isinf(tensor.grad).any(), f"Gradient has Inf for {tensor_name}"

    def test_gradient_flow_through_conv_and_bn_parameters(self) -> None:
        """Verify parameters in Conv1x1 and BatchNorm receive valid gradients."""
        module = FeatureFusion(channels=32, bias=True)
        x = torch.randn(2, 32, 16, 16) + 2.0
        x_channel = torch.randn(2, 32, 16, 16) + 2.0
        x_spatial = torch.randn(2, 32, 16, 16) + 2.0

        out = module(x, x_channel, x_spatial)
        loss = out.sum()
        loss.backward()

        for name, param in module.named_parameters():
            assert param.grad is not None, f"Parameter {name} has None grad"
            assert not torch.all(param.grad == 0.0), f"Parameter {name} has all zero grad"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN grad"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf grad"

    def test_no_nan_or_inf_in_forward_and_backward(self) -> None:
        """Verify numerical stability with large inputs."""
        module = FeatureFusion(channels=32)
        x = (torch.randn(2, 32, 16, 16) * 50.0).requires_grad_(True)
        x_channel = (torch.randn(2, 32, 16, 16) * 50.0).requires_grad_(True)
        x_spatial = (torch.randn(2, 32, 16, 16) * 50.0).requires_grad_(True)

        out = module(x, x_channel, x_spatial)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

        loss = out.mean()
        loss.backward()

        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        assert not torch.isnan(x_channel.grad).any()
        assert not torch.isinf(x_channel.grad).any()
        assert not torch.isnan(x_spatial.grad).any()
        assert not torch.isinf(x_spatial.grad).any()

    def test_deterministic_forward_in_eval_mode(self) -> None:
        """Verify deterministic forward evaluation."""
        module = FeatureFusion(channels=32).eval()
        x = torch.randn(2, 32, 16, 16)
        x_channel = torch.randn(2, 32, 16, 16)
        x_spatial = torch.randn(2, 32, 16, 16)

        with torch.no_grad():
            out1 = module(x, x_channel, x_spatial)
            out2 = module(x, x_channel, x_spatial)

        assert torch.equal(out1, out2)

    # -------------------------------------------------------------------------
    # 17-22. Robust Input Validation and Error Raising
    # -------------------------------------------------------------------------
    def test_invalid_input_rank_raises_value_error(self) -> None:
        """Verify that non-4D inputs raise ValueError."""
        module = FeatureFusion(channels=32)
        x_4d = torch.randn(2, 32, 16, 16)

        # 3D tensor
        with pytest.raises(ValueError, match="Expected 4D tensor for x"):
            module(torch.randn(32, 16, 16), x_4d, x_4d)
        with pytest.raises(ValueError, match="Expected 4D tensor for x_channel"):
            module(x_4d, torch.randn(32, 16, 16), x_4d)
        with pytest.raises(ValueError, match="Expected 4D tensor for x_spatial"):
            module(x_4d, x_4d, torch.randn(32, 16, 16))

    def test_channel_mismatch_raises_value_error(self) -> None:
        """Verify mismatched channel dimension raises ValueError."""
        module = FeatureFusion(channels=32)
        x_32 = torch.randn(2, 32, 16, 16)
        x_64 = torch.randn(2, 64, 16, 16)

        with pytest.raises(ValueError, match="Channel mismatch: x has 64 channels"):
            module(x_64, x_32, x_32)
        with pytest.raises(ValueError, match="Channel mismatch: x_channel has 64 channels"):
            module(x_32, x_64, x_32)
        with pytest.raises(ValueError, match="Channel mismatch: x_spatial has 64 channels"):
            module(x_32, x_32, x_64)

    def test_batch_and_spatial_mismatch_raises_value_error(self) -> None:
        """Verify batch size and spatial resolution mismatch raise ValueError."""
        module = FeatureFusion(channels=32)

        # Batch mismatch
        with pytest.raises(ValueError, match="Batch size mismatch"):
            module(torch.randn(2, 32, 16, 16), torch.randn(3, 32, 16, 16), torch.randn(2, 32, 16, 16))

        # Spatial mismatch
        with pytest.raises(ValueError, match="Spatial resolution mismatch"):
            module(torch.randn(2, 32, 16, 16), torch.randn(2, 32, 16, 16), torch.randn(2, 32, 20, 20))

    def test_non_tensor_raises_type_error(self) -> None:
        """Verify passing non-tensor inputs raises TypeError."""
        module = FeatureFusion(channels=32)
        x_4d = torch.randn(2, 32, 16, 16)

        with pytest.raises(TypeError, match="Expected x to be a torch.Tensor"):
            module([1.0], x_4d, x_4d)  # type: ignore
        with pytest.raises(TypeError, match="Expected x_channel to be a torch.Tensor"):
            module(x_4d, "not_a_tensor", x_4d)  # type: ignore
        with pytest.raises(TypeError, match="Expected x_spatial to be a torch.Tensor"):
            module(x_4d, x_4d, None)  # type: ignore

    def test_invalid_constructor_args_raise_value_error(self) -> None:
        """Verify invalid constructor parameters raise ValueError."""
        with pytest.raises(ValueError, match="channels must be a positive integer"):
            FeatureFusion(channels=0)
        with pytest.raises(ValueError, match="channels must be a positive integer"):
            FeatureFusion(channels=-32)

    # -------------------------------------------------------------------------
    # 23-26. Hardware Contracts & Mixed Precision (AMP)
    # -------------------------------------------------------------------------
    def test_pure_cpu_execution(self) -> None:
        """Verify clean CPU execution."""
        module = FeatureFusion(channels=32)
        x = torch.randn(2, 32, 16, 16)
        x_c = torch.randn(2, 32, 16, 16)
        x_s = torch.randn(2, 32, 16, 16)

        out = module(x, x_c, x_s)
        assert out.device.type == "cpu"

    def test_device_contract_and_float64(self) -> None:
        """Verify device matching contracts and double precision support."""
        module = FeatureFusion(channels=16)

        # Double precision execution
        module.to(dtype=torch.float64)
        x_d = torch.randn(2, 16, 8, 8, dtype=torch.float64)
        x_cd = torch.randn(2, 16, 8, 8, dtype=torch.float64)
        x_sd = torch.randn(2, 16, 8, 8, dtype=torch.float64)
        out_d = module(x_d, x_cd, x_sd)
        assert out_d.dtype == torch.float64

        # Device mismatch contract check
        if torch.cuda.is_available():
            module_cpu = FeatureFusion(channels=16)
            x_cuda = torch.randn(2, 16, 8, 8, device="cuda")
            x_cpu = torch.randn(2, 16, 8, 8)
            with pytest.raises(RuntimeError, match="Device mismatch"):
                module_cpu(x_cuda, x_cpu, x_cpu)

    def test_autocast_cpu_bfloat16(self) -> None:
        """
        Verify PyTorch Automatic Mixed Precision (torch.autocast) execution on CPU with bfloat16.
        Ensures module does not perform rigid dtype checks that break AMP.
        """
        module = FeatureFusion(channels=32)
        x = torch.randn(2, 32, 16, 16)
        x_c = torch.randn(2, 32, 16, 16)
        x_s = torch.randn(2, 32, 16, 16)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = module(x, x_c, x_s)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_autocast_cuda_float16(self) -> None:
        """Verify CUDA float16 autocast compatibility if CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA device not available for float16 autocast testing")

        device = torch.device("cuda")
        module = FeatureFusion(channels=32).to(device)
        x = torch.randn(2, 32, 16, 16, device=device)
        x_c = torch.randn(2, 32, 16, 16, device=device)
        x_s = torch.randn(2, 32, 16, 16, device=device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = module(x, x_c, x_s)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    # -------------------------------------------------------------------------
    # 27. State Dict Cleanliness
    # -------------------------------------------------------------------------
    def test_state_dict_cleanliness_and_no_aliases(self) -> None:
        """Verify state dict keys match expected modules without rogue parameters."""
        module = FeatureFusion(channels=32, bias=True)
        state_keys = set(module.state_dict().keys())

        expected_keys = {
            "conv.weight",
            "conv.bias",
            "bn.weight",
            "bn.bias",
            "bn.running_mean",
            "bn.running_var",
            "bn.num_batches_tracked",
        }

        assert state_keys == expected_keys

        # Ensure parameters are distinct objects in memory
        params = list(module.parameters())
        assert len(params) == 4  # conv.weight, conv.bias, bn.weight, bn.bias
        param_ptrs = {p.data_ptr() for p in params}
        assert len(param_ptrs) == 4, "Duplicate memory addresses found among parameters"

    # -------------------------------------------------------------------------
    # 28. Exact Parameter Count Formula Sanity Check
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        ("channels", "bias", "expected_params"),
        [
            # YOLOv8s canonical feature levels:
            # P3: C=128 -> N = 2*(128)^2 + 3*(128) = 32768 + 384 = 33,152
            (128, True, 33152),
            # P4: C=256 -> N = 2*(256)^2 + 3*(256) = 131072 + 768 = 131,840
            (256, True, 131840),
            # P5: C=512 -> N = 2*(512)^2 + 3*(512) = 524288 + 1536 = 525,824
            (512, True, 525824),
            # Without bias:
            # P3: C=128 -> N = 2*(128)^2 + 2*(128) = 32768 + 256 = 33,024
            (128, False, 33024),
            # P4: C=256 -> N = 2*(256)^2 + 2*(256) = 131072 + 512 = 131,584
            (256, False, 131584),
            # P5: C=512 -> N = 2*(512)^2 + 2*(512) = 524288 + 1024 = 525,312
            (512, False, 525312),
            # Small configuration
            (32, True, 2144),
            (32, False, 2112),
        ],
    )
    def test_parameter_count_matches_theoretical_formula(
        self, channels: int, bias: bool, expected_params: int
    ) -> None:
        """Verify that total parameter count strictly matches analytical formula."""
        module = FeatureFusion(channels=channels, bias=bias)

        actual_params = sum(p.numel() for p in module.parameters())
        formula_params = compute_expected_params(channels=channels, bias=bias)

        assert actual_params == expected_params, (
            f"Param mismatch: expected {expected_params}, got {actual_params}"
        )
        assert actual_params == formula_params, (
            f"Formula mismatch: formula produced {formula_params}, but got {actual_params}"
        )

    # -------------------------------------------------------------------------
    # 29. Multi-Module Integration Test
    # -------------------------------------------------------------------------
    def test_integration_full_standalone_pipeline(self) -> None:
        """
        Standalone multi-module integration test connecting:
            ScaleAwareModule -> S
            AdaptiveSpatialAttention(X, S) -> X_spatial
            AdaptiveChannelAttention(X, S) -> X_channel
            FeatureFusion(X, X_channel, X_spatial) -> X_output

        Verifies:
        1. Correct end-to-end forward shape matching X.
        2. Non-zero gradients propagate back to:
           - active scale embedding (level 3)
           - spatial attention parameters
           - channel attention parameters
           - fusion Conv1x1 and BatchNorm parameters
        3. Inactive scale embeddings (level 4, 5) remain untouched.
        """
        C = 128
        D = 64
        scale_level = 3

        # Instantiate standalone modules
        scale_mod = ScaleAwareModule(embed_dim=D)
        spatial_mod = AdaptiveSpatialAttention(channels=C, scale_dim=D)
        channel_mod = AdaptiveChannelAttention(channels=C, scale_dim=D)
        fusion_mod = FeatureFusion(channels=C, bias=True)

        x = (torch.randn(2, C, 80, 80) + 1.0).requires_grad_(True)

        # 1. Scale encoding
        s = scale_mod(x, scale_level=scale_level)
        assert s.shape == (2, D, 80, 80)

        # 2. Spatial attention
        x_spatial = spatial_mod(x, s)
        assert x_spatial.shape == (2, C, 80, 80)

        # 3. Channel attention
        x_channel = channel_mod(x, s)
        assert x_channel.shape == (2, C, 80, 80)

        # 4. Feature fusion
        x_output = fusion_mod(x, x_channel, x_spatial)
        assert x_output.shape == (2, C, 80, 80)

        # Backward gradient flow
        loss = x_output.sum()
        loss.backward()

        # 1. Input x receives valid gradients
        assert x.grad is not None
        assert not torch.all(x.grad == 0.0)

        # 2. Active scale embedding (P3) receives gradient
        assert scale_mod.embed_p3.grad is not None
        assert not torch.all(scale_mod.embed_p3.grad == 0.0)

        # 3. Inactive scale embeddings remain untouched
        assert scale_mod.embed_p4.grad is None
        assert scale_mod.embed_p5.grad is None

        # 4. Spatial attention parameters receive gradients
        for name, p in spatial_mod.named_parameters():
            assert p.grad is not None, f"Spatial param {name} grad is None"
            assert not torch.all(p.grad == 0.0), f"Spatial param {name} grad is all zero"

        # 5. Channel attention parameters receive gradients
        for name, p in channel_mod.named_parameters():
            assert p.grad is not None, f"Channel param {name} grad is None"
            assert not torch.all(p.grad == 0.0), f"Channel param {name} grad is all zero"

        # 6. Fusion parameters receive gradients
        for name, p in fusion_mod.named_parameters():
            assert p.grad is not None, f"Fusion param {name} grad is None"
            assert not torch.all(p.grad == 0.0), f"Fusion param {name} grad is all zero"
