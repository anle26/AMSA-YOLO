"""
Unit and Integration Tests for Adaptive Channel Attention Module.

Validates:
1. Canonical YOLOv8s / AMSA-YOLO feature shapes:
   - P3: C=128, 80x80 (Ultralytics 8.2.103 YOLOv8s Backbone Layer 4)
   - P4: C=256, 40x40 (Ultralytics 8.2.103 YOLOv8s Backbone Layer 6)
   - P5: C=512, 20x20 (Ultralytics 8.2.103 YOLOv8s Backbone Layer 9)
2. Internal tensor shapes:
   - GAP(X)     == [B, C, 1, 1]
   - GMP(X)     == [B, C, 1, 1]
   - A_channel  == [B, C, 1, 1]
   - GAP(S)     == [B, D, 1, 1]
   - M_scale    == [B, C, 1, 1]
   - A_adaptive == [B, C, 1, 1]
   - X_channel  == [B, C, H, W]
3. Sigmoid value range bounds [0, 1] for A_channel, M_scale, and A_adaptive.
4. Strict mathematical paper equation fidelity:
   A_channel = Sigmoid(MLP(GAP(X)) + MLP(GMP(X)))
5. Adaptive channel attention fusion fidelity:
   A_adaptive = A_channel * M_scale
6. Output feature recalibration fidelity:
   X_channel = X * A_adaptive
7. Shared-MLP test:
   Confirm GAP(X) and GMP(X) use the SAME channel MLP parameters.
8. Dynamic batch sizes and dynamic spatial dimensions.
9. Gradient flow through X, S, and all module parameters without NaN/Inf.
10. Deterministic evaluation forward pass.
11. Robust input validation and error raising (rank, channels, scale_dim, batch/spatial mismatch, types).
12. Device contracts (CPU, CUDA if available, device mismatch raises RuntimeError).
13. Automatic Mixed Precision (AMP) / autocast compatibility (CPU bfloat16, CUDA float16).
14. State dict cleanliness and no duplicate module aliases.
15. Exact parameter count formula verification against analytical derivation.
16. Chained integration test: ScaleAwareModule -> S -> AdaptiveChannelAttention.
"""

from typing import Tuple
import pytest
import torch
import torch.nn as nn

from src.amsa.channel_attention import AdaptiveChannelAttention
from src.amsa.scale_aware import ScaleAwareModule


def compute_expected_params(
    channels: int,
    scale_dim: int = 64,
    channel_reduction: int = 16,
    scale_reduction: int = 4,
    bias: bool = True,
) -> int:
    """Analytical parameter count formula for AdaptiveChannelAttention."""
    c_red = max(1, channels // channel_reduction)
    d_hid = max(1, scale_dim // scale_reduction)

    n_chan_w = 2 * channels * c_red
    n_chan_b = (c_red + channels) if bias else 0

    n_scale_w = scale_dim * d_hid + d_hid * channels
    n_scale_b = (d_hid + channels) if bias else 0

    return n_chan_w + n_chan_b + n_scale_w + n_scale_b


class TestAdaptiveChannelAttention:
    """Test suite for AdaptiveChannelAttention standalone module."""

    # -------------------------------------------------------------------------
    # 1-3. Mandatory Base Tests: Shapes and Range Bounds
    # -------------------------------------------------------------------------
    def test_output_shape_equals_input_shape(self) -> None:
        """Verify that output tensor preserves the exact shape of input X."""
        module = AdaptiveChannelAttention(channels=64, scale_dim=64)
        x = torch.randn(3, 64, 32, 48)
        s = torch.randn(3, 64, 32, 48)

        out = module(x, s)
        assert out.shape == x.shape
        assert isinstance(out, torch.Tensor)

    def test_internal_tensor_shapes(self) -> None:
        """
        Verify internal attention and modulation shapes:
            GAP(X)      == [B, C, 1, 1]
            GMP(X)      == [B, C, 1, 1]
            A_channel   == [B, C, 1, 1]
            GAP(S)      == [B, D, 1, 1]
            M_scale     == [B, C, 1, 1]
            A_adaptive  == [B, C, 1, 1]
            X_channel   == [B, C, H, W]
        """
        B, C, D, H, W = 4, 32, 64, 28, 28
        module = AdaptiveChannelAttention(channels=C, scale_dim=D)
        x = torch.randn(B, C, H, W)
        s = torch.randn(B, D, H, W)

        out, maps = module(x, s, return_attention_maps=True)

        assert out.shape == (B, C, H, W)
        assert maps["gap_x"].shape == (B, C, 1, 1)
        assert maps["gmp_x"].shape == (B, C, 1, 1)
        assert maps["a_channel"].shape == (B, C, 1, 1)
        assert maps["gap_s"].shape == (B, D, 1, 1)
        assert maps["m_scale"].shape == (B, C, 1, 1)
        assert maps["a_adaptive"].shape == (B, C, 1, 1)

    def test_all_sigmoid_outputs_within_zero_one(self) -> None:
        """Verify that all sigmoid-activated attention weights lie in [0, 1]."""
        module = AdaptiveChannelAttention(channels=48, scale_dim=64)
        x = torch.randn(2, 48, 24, 24) * 10.0  # Large inputs to stress test sigmoid saturation
        s = torch.randn(2, 64, 24, 24) * 10.0

        _, maps = module(x, s, return_attention_maps=True)

        for key in ("a_channel", "m_scale", "a_adaptive"):
            tensor = maps[key]
            assert torch.all(tensor >= 0.0), f"{key} contains values < 0.0: min={tensor.min()}"
            assert torch.all(tensor <= 1.0), f"{key} contains values > 1.0: max={tensor.max()}"

    # -------------------------------------------------------------------------
    # 4-6. Mathematical Paper Equation Fidelity
    # -------------------------------------------------------------------------
    def test_explicit_paper_equation_fidelity(self) -> None:
        """
        Verify paper equations:
            1. A_channel = Sigmoid(MLP(GAP(X)) + MLP(GMP(X)))
            2. M_scale = Sigmoid(MLP(GAP(S)))
            3. A_adaptive = A_channel * M_scale
            4. X_channel = X * A_adaptive
        """
        module = AdaptiveChannelAttention(channels=32, scale_dim=64)
        x = torch.randn(2, 32, 16, 16)
        s = torch.randn(2, 64, 16, 16)

        x_channel, maps = module(x, s, return_attention_maps=True)

        # 1. Standard channel attention equation
        gap_x = torch.mean(x, dim=(-2, -1), keepdim=True)
        gmp_x = torch.amax(x, dim=(-2, -1), keepdim=True)
        expected_a_channel = module.sigmoid(module.channel_mlp(gap_x) + module.channel_mlp(gmp_x))
        assert torch.allclose(maps["a_channel"], expected_a_channel, atol=1e-6)

        # 2. Scale modulation equation
        gap_s = torch.mean(s, dim=(-2, -1), keepdim=True)
        expected_m_scale = module.sigmoid(module.scale_mlp(gap_s))
        assert torch.allclose(maps["m_scale"], expected_m_scale, atol=1e-6)

        # 3. Adaptive channel attention
        expected_a_adaptive = expected_a_channel * expected_m_scale
        assert torch.allclose(maps["a_adaptive"], expected_a_adaptive, atol=1e-6)

        # 4. Output recalibration
        expected_x_channel = x * expected_a_adaptive
        assert torch.allclose(x_channel, expected_x_channel, atol=1e-6)

    def test_shared_mlp_parameter_sharing(self) -> None:
        """
        Verify that GAP(X) and GMP(X) pass through the EXACT SAME channel MLP parameters.
        Confirm there is only one channel MLP in the module.
        """
        module = AdaptiveChannelAttention(channels=64, scale_dim=64)

        # Verify there are no duplicate/split channel MLPs
        conv_modules = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
        # Expected: 2 convs in channel_mlp, 2 convs in scale_mlp -> total 4 Conv2d modules
        assert len(conv_modules) == 4

        # Track execution calls to channel_mlp
        call_count = 0

        def hook_fn(m, inp, outp):
            nonlocal call_count
            call_count += 1

        hook = module.channel_mlp.register_forward_hook(hook_fn)

        x = torch.randn(2, 64, 16, 16)
        module.compute_channel_attention(x)
        hook.remove()

        # Both GAP(X) and GMP(X) must pass through module.channel_mlp -> exactly 2 calls
        assert call_count == 2

        # Verify gradient flow reaches the shared weights from both GAP and GMP
        gap_x = torch.mean(x, dim=(-2, -1), keepdim=True)
        gmp_x = torch.amax(x, dim=(-2, -1), keepdim=True)

        out_gap = module.channel_mlp(gap_x).sum()
        grad_gap = torch.autograd.grad(out_gap, module.channel_mlp[0].weight, retain_graph=True)[0]

        out_gmp = module.channel_mlp(gmp_x).sum()
        grad_gmp = torch.autograd.grad(out_gmp, module.channel_mlp[0].weight, retain_graph=True)[0]

        assert not torch.all(grad_gap == 0.0)
        assert not torch.all(grad_gmp == 0.0)

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
        module = AdaptiveChannelAttention(channels=channels, scale_dim=64)
        x = torch.randn(2, channels, H, W)
        s = torch.randn(2, 64, H, W)

        out = module(x, s)
        assert out.shape == (2, channels, H, W), f"Failed for {label}"

    # -------------------------------------------------------------------------
    # 10-11. Dynamic Batch Sizes and Spatial Dimensions
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("batch_size", [1, 2, 5, 8])
    def test_dynamic_batch_sizes(self, batch_size: int) -> None:
        """Verify module functions across varying batch sizes."""
        module = AdaptiveChannelAttention(channels=64, scale_dim=64)
        x = torch.randn(batch_size, 64, 32, 32)
        s = torch.randn(batch_size, 64, 32, 32)

        out = module(x, s)
        assert out.shape == (batch_size, 64, 32, 32)

    @pytest.mark.parametrize("spatial_size", [(17, 23), (33, 31), (80, 80), (7, 13)])
    def test_dynamic_spatial_dimensions(self, spatial_size: Tuple[int, int]) -> None:
        """Verify module handles arbitrary non-square and prime spatial dimensions."""
        H, W = spatial_size
        module = AdaptiveChannelAttention(channels=32, scale_dim=64)
        x = torch.randn(2, 32, H, W)
        s = torch.randn(2, 64, H, W)

        out = module(x, s)
        assert out.shape == (2, 32, H, W)

    # -------------------------------------------------------------------------
    # 12-15. Gradient Flow & Numerical Stability
    # -------------------------------------------------------------------------
    def test_gradient_flow_through_x_and_s(self) -> None:
        """Verify that gradients flow back to both inputs X and S."""
        module = AdaptiveChannelAttention(channels=32, scale_dim=64)
        x = torch.randn(2, 32, 16, 16, requires_grad=True)
        s = torch.randn(2, 64, 16, 16, requires_grad=True)

        out = module(x, s)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0.0)
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

        assert s.grad is not None
        assert not torch.all(s.grad == 0.0)
        assert not torch.isnan(s.grad).any()
        assert not torch.isinf(s.grad).any()

    def test_gradient_flow_through_all_module_parameters(self) -> None:
        """Verify all parameters in channel_mlp and scale_mlp receive valid gradients."""
        module = AdaptiveChannelAttention(channels=32, scale_dim=64)
        x = torch.randn(2, 32, 16, 16)
        s = torch.randn(2, 64, 16, 16)

        out = module(x, s)
        loss = out.sum()
        loss.backward()

        for name, param in module.named_parameters():
            assert param.grad is not None, f"Parameter {name} has None grad"
            assert not torch.all(param.grad == 0.0), f"Parameter {name} has all zero grad"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN grad"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf grad"

    def test_no_nan_or_inf_in_forward_and_backward(self) -> None:
        """Verify numerical stability with large and negative values."""
        module = AdaptiveChannelAttention(channels=32, scale_dim=64)
        x = (torch.randn(2, 32, 16, 16) * 50.0).requires_grad_(True)
        s = (torch.randn(2, 64, 16, 16) * 50.0).requires_grad_(True)

        out = module(x, s)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

        loss = out.mean()
        loss.backward()

        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        assert not torch.isnan(s.grad).any()
        assert not torch.isinf(s.grad).any()

    def test_deterministic_forward_in_eval_mode(self) -> None:
        """Verify deterministic forward evaluation."""
        module = AdaptiveChannelAttention(channels=32, scale_dim=64).eval()
        x = torch.randn(2, 32, 16, 16)
        s = torch.randn(2, 64, 16, 16)

        with torch.no_grad():
            out1 = module(x, s)
            out2 = module(x, s)

        assert torch.equal(out1, out2)

    # -------------------------------------------------------------------------
    # 16-22. Robust Input Validation and Error Raising
    # -------------------------------------------------------------------------
    def test_invalid_input_rank_raises_value_error(self) -> None:
        """Verify that non-4D inputs raise ValueError."""
        module = AdaptiveChannelAttention(channels=32, scale_dim=64)

        # 3D tensor for X
        with pytest.raises(ValueError, match="Expected 4D tensor for x"):
            module(torch.randn(32, 16, 16), torch.randn(2, 64, 16, 16))

        # 5D tensor for S
        with pytest.raises(ValueError, match="Expected 4D tensor for s"):
            module(torch.randn(2, 32, 16, 16), torch.randn(2, 64, 16, 16, 1))

    def test_channel_mismatch_raises_value_error(self) -> None:
        """Verify that mismatched channel dimension in X raises ValueError."""
        module = AdaptiveChannelAttention(channels=32, scale_dim=64)
        x = torch.randn(2, 64, 16, 16)  # 64 instead of 32
        s = torch.randn(2, 64, 16, 16)

        with pytest.raises(ValueError, match="Channel mismatch: x has 64 channels"):
            module(x, s)

    def test_scale_dim_mismatch_raises_value_error(self) -> None:
        """Verify that mismatched scale dimension in S raises ValueError."""
        module = AdaptiveChannelAttention(channels=32, scale_dim=64)
        x = torch.randn(2, 32, 16, 16)
        s = torch.randn(2, 32, 16, 16)  # 32 instead of 64

        with pytest.raises(ValueError, match="Scale dim mismatch: s has 32 scale dimension"):
            module(x, s)

    def test_batch_and_spatial_mismatch_raises_value_error(self) -> None:
        """Verify batch size and spatial resolution mismatch between X and S raises ValueError."""
        module = AdaptiveChannelAttention(channels=32, scale_dim=64)

        # Batch mismatch
        with pytest.raises(ValueError, match="Batch size mismatch"):
            module(torch.randn(2, 32, 16, 16), torch.randn(3, 64, 16, 16))

        # Spatial mismatch
        with pytest.raises(ValueError, match="Spatial resolution mismatch"):
            module(torch.randn(2, 32, 16, 16), torch.randn(2, 64, 20, 20))

    def test_non_tensor_raises_type_error(self) -> None:
        """Verify passing non-tensor inputs raises TypeError."""
        module = AdaptiveChannelAttention(channels=32, scale_dim=64)

        with pytest.raises(TypeError, match="Expected x to be a torch.Tensor"):
            module([1.0, 2.0], torch.randn(2, 64, 16, 16))  # type: ignore

        with pytest.raises(TypeError, match="Expected s to be a torch.Tensor"):
            module(torch.randn(2, 32, 16, 16), "not_a_tensor")  # type: ignore

    def test_invalid_constructor_args_raise_value_error(self) -> None:
        """Verify invalid constructor parameters raise ValueError."""
        with pytest.raises(ValueError, match="channels must be a positive integer"):
            AdaptiveChannelAttention(channels=0)

        with pytest.raises(ValueError, match="scale_dim must be a positive integer"):
            AdaptiveChannelAttention(channels=32, scale_dim=-1)

        with pytest.raises(ValueError, match="channel_reduction must be a positive integer"):
            AdaptiveChannelAttention(channels=32, channel_reduction=0)

        with pytest.raises(ValueError, match="scale_reduction must be a positive integer"):
            AdaptiveChannelAttention(channels=32, scale_reduction=-4)

    # -------------------------------------------------------------------------
    # 23-26. Hardware Contracts & Mixed Precision (AMP)
    # -------------------------------------------------------------------------
    def test_pure_cpu_execution(self) -> None:
        """Verify clean CPU execution."""
        module = AdaptiveChannelAttention(channels=32, scale_dim=64)
        x = torch.randn(2, 32, 16, 16)
        s = torch.randn(2, 64, 16, 16)

        out = module(x, s)
        assert out.device.type == "cpu"

    def test_device_contract_and_float64(self) -> None:
        """Verify device matching contracts and double precision support."""
        module = AdaptiveChannelAttention(channels=16, scale_dim=32)

        # Double precision (float64) execution
        module.to(dtype=torch.float64)
        x_d = torch.randn(2, 16, 8, 8, dtype=torch.float64)
        s_d = torch.randn(2, 32, 8, 8, dtype=torch.float64)
        out_d = module(x_d, s_d)
        assert out_d.dtype == torch.float64

        # Device mismatch contract check (simulated with dummy device string if multiple devices unavailable)
        if torch.cuda.is_available():
            module_cpu = AdaptiveChannelAttention(channels=16, scale_dim=32)
            x_cuda = torch.randn(2, 16, 8, 8, device="cuda")
            s_cpu = torch.randn(2, 32, 8, 8)
            with pytest.raises(RuntimeError, match="Device mismatch"):
                module_cpu(x_cuda, s_cpu)

    def test_autocast_cpu_bfloat16(self) -> None:
        """
        Verify PyTorch Automatic Mixed Precision (torch.autocast) execution on CPU with bfloat16.
        Ensures module does not perform rigid dtype checks that break AMP.
        """
        module = AdaptiveChannelAttention(channels=32, scale_dim=64)
        x = torch.randn(2, 32, 16, 16)
        s = torch.randn(2, 64, 16, 16)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = module(x, s)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_autocast_cuda_float16(self) -> None:
        """Verify CUDA float16 autocast compatibility if CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA device not available for float16 autocast testing")

        device = torch.device("cuda")
        module = AdaptiveChannelAttention(channels=32, scale_dim=64).to(device)
        x = torch.randn(2, 32, 16, 16, device=device)
        s = torch.randn(2, 64, 16, 16, device=device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = module(x, s)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    # -------------------------------------------------------------------------
    # 27. Chained Integration Test: ScaleAwareModule -> S -> AdaptiveChannelAttention
    # -------------------------------------------------------------------------
    def test_integration_scale_aware_to_channel_attention(self) -> None:
        """
        End-to-end chained integration test:
            ScaleAwareModule -> S -> AdaptiveChannelAttention -> X_channel
        Verifies:
        1. Correct forward shape.
        2. Backward gradients propagate through AdaptiveChannelAttention back to the
           active scale embedding in ScaleAwareModule (s=3).
        3. Inactive scale embeddings (s=4, s=5) remain untouched (None grad).
        4. All channel attention parameters receive non-zero gradients.
        """
        scale_module = ScaleAwareModule(embed_dim=64)
        channel_module = AdaptiveChannelAttention(channels=128, scale_dim=64)

        x = torch.randn(2, 128, 80, 80, requires_grad=True)

        # Scale encoding from scale level 3 (P3)
        s = scale_module(x, scale_level=3)
        assert s.shape == (2, 64, 80, 80)

        # Adaptive channel recalibration
        x_channel = channel_module(x, s)
        assert x_channel.shape == (2, 128, 80, 80)

        loss = x_channel.sum()
        loss.backward()

        # 1. Input x receives valid gradient
        assert x.grad is not None
        assert not torch.all(x.grad == 0.0)

        # 2. Active scale embedding (P3) receives gradient
        assert scale_module.embed_p3.grad is not None
        assert not torch.all(scale_module.embed_p3.grad == 0.0)

        # 3. Inactive scale embeddings remain untouched
        assert scale_module.embed_p4.grad is None
        assert scale_module.embed_p5.grad is None

        # 4. Scale projection conv receives gradient
        assert scale_module.scale_proj.weight.grad is not None
        assert not torch.all(scale_module.scale_proj.weight.grad == 0.0)

        # 5. All channel attention parameters receive gradients
        for name, param in channel_module.named_parameters():
            assert param.grad is not None, f"Parameter {name} has None grad"
            assert not torch.all(param.grad == 0.0), f"Parameter {name} has all zero grad"

    # -------------------------------------------------------------------------
    # 28-29. State Dict Cleanliness & Aliasing
    # -------------------------------------------------------------------------
    def test_state_dict_cleanliness_and_no_aliases(self) -> None:
        """Verify state dict keys match expected modules without duplicate aliases."""
        module = AdaptiveChannelAttention(channels=32, scale_dim=64)
        state_keys = set(module.state_dict().keys())

        expected_keys = {
            "channel_mlp.0.weight",
            "channel_mlp.0.bias",
            "channel_mlp.2.weight",
            "channel_mlp.2.bias",
            "scale_mlp.0.weight",
            "scale_mlp.0.bias",
            "scale_mlp.2.weight",
            "scale_mlp.2.bias",
        }

        assert state_keys == expected_keys

        # Ensure parameters are distinct objects in memory
        params = list(module.parameters())
        assert len(params) == 8
        param_ptrs = {p.data_ptr() for p in params}
        assert len(param_ptrs) == 8, "Duplicate memory addresses found among parameters"

    # -------------------------------------------------------------------------
    # 30. Exact Parameter Count Formula Sanity Check
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        ("channels", "scale_dim", "channel_reduction", "scale_reduction", "bias", "expected_params"),
        [
            # YOLOv8s canonical feature levels (D=64, r_c=16, r_s=4, bias=True):
            # P3: C=128 -> N = 5,400
            (128, 64, 16, 4, True, 5400),
            # P4: C=256 -> N = 13,856
            (256, 64, 16, 4, True, 13856),
            # P5: C=512 -> N = 43,056
            (512, 64, 16, 4, True, 43056),
            # Small configuration
            (64, 64, 16, 4, True, 2708),
            # Generic 1024-channel configuration (r_c=16, r_s=4)
            (1024, 64, 16, 4, True, 150608),
            # Without bias (P3)
            (128, 64, 16, 4, False, 5120),
            # Custom reductions
            (32, 16, 8, 2, True, 716),
        ],
    )
    def test_parameter_count_matches_theoretical_formula(
        self,
        channels: int,
        scale_dim: int,
        channel_reduction: int,
        scale_reduction: int,
        bias: bool,
        expected_params: int,
    ) -> None:
        """Verify that total parameter count strictly matches analytical formula."""
        module = AdaptiveChannelAttention(
            channels=channels,
            scale_dim=scale_dim,
            channel_reduction=channel_reduction,
            scale_reduction=scale_reduction,
            bias=bias,
        )

        actual_params = sum(p.numel() for p in module.parameters())
        formula_params = compute_expected_params(
            channels=channels,
            scale_dim=scale_dim,
            channel_reduction=channel_reduction,
            scale_reduction=scale_reduction,
            bias=bias,
        )

        assert actual_params == expected_params, (
            f"Param mismatch: expected {expected_params}, got {actual_params}"
        )
        assert actual_params == formula_params, (
            f"Formula mismatch: formula produced {formula_params}, but got {actual_params}"
        )
