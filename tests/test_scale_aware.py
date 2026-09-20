"""
Unit tests for ScaleAwareModule in AMSA-YOLO.

Requirements covered:
A. Shape tests:
   - P3: [2, 256, 80, 80] -> S [2, 64, 80, 80]
   - P4: [2, 512, 40, 40] -> S [2, 64, 40, 40]
   - P5: [2, 1024, 20, 20] -> S [2, 64, 20, 20]
B. Learnability:
   - verify scale embeddings are nn.Parameter with requires_grad == True
   - verify scale_proj is nn.Conv2d with requires_grad == True
C. Gradient flow:
   - forward -> scalar loss -> backward -> verify embedding & Conv1x1 gradients exist without NaN/Inf
D. Scale distinction:
   - verify P3, P4, P5 embeddings are distinct, separate learnable parameters
E. Invalid input:
   - invalid scale levels raise clear ValueError
F. Determinism:
   - identical seeds produce identical initialization and forward outputs
G. CPU execution & Dtype/Device Contract:
   - all tests execute completely without CUDA
   - float32 CPU and float64 CPU (after module.double())
   - module does not silently move or cast parameters inside forward()
H. State Dict Namespace Hygiene:
   - state_dict contains scale_proj.weight and scale_proj.bias
   - state_dict does NOT contain proj.weight or proj.bias
"""

import pytest
import torch
import torch.nn as nn
from src.amsa.scale_aware import ScaleAwareModule


class TestScaleAwareModule:
    """Test suite verifying paper-faithful ScaleAwareModule implementation."""

    # --------------------------------------------------------------------------
    # A. Shape Tests
    # --------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "batch_size, in_channels, height, width, scale_level",
        [
            (2, 256, 80, 80, 3),   # P3 scale
            (2, 512, 40, 40, 4),   # P4 scale
            (2, 1024, 20, 20, 5),  # P5 scale
        ],
    )
    def test_paper_canonical_shapes(
        self,
        batch_size: int,
        in_channels: int,
        height: int,
        width: int,
        scale_level: int,
    ) -> None:
        """Verify output tensor preserves (B, D, H, W) for P3, P4, and P5 canonical dimensions."""
        embed_dim = 64
        module = ScaleAwareModule(embed_dim=embed_dim)

        x = torch.randn(batch_size, in_channels, height, width, device="cpu")
        out = module(x, scale_level=scale_level)

        assert isinstance(out, torch.Tensor), "Output must be a torch.Tensor"
        assert out.shape == (batch_size, embed_dim, height, width), (
            f"Expected shape ({batch_size}, {embed_dim}, {height}, {width}), got {out.shape}"
        )
        assert out.device == x.device, "Output device must match input device"
        assert out.dtype == x.dtype, "Output dtype must match input dtype"

    @pytest.mark.parametrize("scale_level", [3, 4, 5, "P3", "P4", "P5", "3", "4", "5"])
    def test_flexible_scale_identifiers(self, scale_level: int | str) -> None:
        """Verify both integer and string scale representations are supported."""
        module = ScaleAwareModule(embed_dim=64)
        x = torch.randn(1, 128, 40, 40, device="cpu")
        out = module(x, scale_level)
        assert out.shape == (1, 64, 40, 40)

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_dynamic_batch_sizes(self, batch_size: int) -> None:
        """Verify module does not hardcode batch size."""
        module = ScaleAwareModule(embed_dim=64)
        x = torch.randn(batch_size, 256, 80, 80, device="cpu")
        out = module(x, scale_level=3)
        assert out.shape == (batch_size, 64, 80, 80)

    # --------------------------------------------------------------------------
    # B. Learnability Tests
    # --------------------------------------------------------------------------
    def test_scale_embeddings_are_learnable_parameters(self) -> None:
        """Verify P3/P4/P5 scale embeddings are nn.Parameter instances with requires_grad=True."""
        module = ScaleAwareModule(embed_dim=64)

        for level in [3, 4, 5]:
            embed = module.get_embedding(level)
            assert isinstance(embed, nn.Parameter), f"Embedding for scale {level} must be an nn.Parameter"
            assert embed.requires_grad is True, f"Embedding for scale {level} must have requires_grad=True"
            assert embed.shape == (64,), f"Embedding for scale {level} must have shape (64,), got {embed.shape}"

        # Test direct property access
        assert isinstance(module.embed_p3, nn.Parameter) and module.embed_p3.requires_grad is True
        assert isinstance(module.embed_p4, nn.Parameter) and module.embed_p4.requires_grad is True
        assert isinstance(module.embed_p5, nn.Parameter) and module.embed_p5.requires_grad is True

    def test_conv1x1_is_learnable(self) -> None:
        """Verify 1x1 projection layer is an nn.Conv2d with trainable parameters."""
        module = ScaleAwareModule(embed_dim=64, bias=True)
        assert isinstance(module.scale_proj, nn.Conv2d)
        assert module.scale_proj.kernel_size == (1, 1)
        assert module.scale_proj.weight.requires_grad is True
        assert module.scale_proj.bias is not None and module.scale_proj.bias.requires_grad is True

    # --------------------------------------------------------------------------
    # C. Gradient Flow Tests
    # --------------------------------------------------------------------------
    @pytest.mark.parametrize("scale_level", [3, 4, 5])
    def test_gradient_flow_and_numerical_stability(self, scale_level: int) -> None:
        """Verify gradient propagation to the active embedding and projection weights without NaN/Inf."""
        module = ScaleAwareModule(embed_dim=64)
        module.zero_grad()

        x = torch.randn(2, 256, 40, 40, device="cpu")
        out = module(x, scale_level=scale_level)

        # Compute scalar loss and backpropagate
        loss = out.sum()
        loss.backward()

        active_embed = module.get_embedding(scale_level)
        assert active_embed.grad is not None, f"Active embedding for scale {scale_level} must receive gradients"
        assert not torch.isnan(active_embed.grad).any(), "Embedding gradient must not contain NaNs"
        assert not torch.isinf(active_embed.grad).any(), "Embedding gradient must not contain Infs"
        assert (active_embed.grad != 0).any(), "Embedding gradient must not be all zeros"

        # Verify 1x1 conv layer gradients
        assert module.scale_proj.weight.grad is not None, "Projection weight must receive gradients"
        assert not torch.isnan(module.scale_proj.weight.grad).any(), "Projection weight gradient must not contain NaNs"
        assert not torch.isinf(module.scale_proj.weight.grad).any(), "Projection weight gradient must not contain Infs"
        assert (module.scale_proj.weight.grad != 0).any(), "Projection weight gradient must not be all zeros"

        if module.scale_proj.bias is not None:
            assert module.scale_proj.bias.grad is not None
            assert not torch.isnan(module.scale_proj.bias.grad).any()

        # Verify inactive scale embeddings do NOT receive gradients
        for other_level in [3, 4, 5]:
            if other_level != scale_level:
                inactive_embed = module.get_embedding(other_level)
                assert inactive_embed.grad is None, (
                    f"Inactive embedding for scale {other_level} should not receive gradients when scale_level={scale_level}"
                )

    # --------------------------------------------------------------------------
    # D. Scale Distinction Tests
    # --------------------------------------------------------------------------
    def test_scale_embeddings_are_distinct_parameters(self) -> None:
        """Verify P3, P4, P5 embeddings are separate, distinct objects and initialized differently."""
        module = ScaleAwareModule(embed_dim=64)

        e3 = module.get_embedding(3)
        e4 = module.get_embedding(4)
        e5 = module.get_embedding(5)

        # Distinct parameter references
        assert e3 is not e4, "P3 and P4 embeddings must be distinct parameter instances"
        assert e4 is not e5, "P4 and P5 embeddings must be distinct parameter instances"
        assert e3 is not e5, "P3 and P5 embeddings must be distinct parameter instances"

        # Distinct numerical initializations
        assert not torch.allclose(e3, e4), "P3 and P4 embeddings must not have identical values"
        assert not torch.allclose(e4, e5), "P4 and P5 embeddings must not have identical values"
        assert not torch.allclose(e3, e5), "P3 and P5 embeddings must not have identical values"

    # --------------------------------------------------------------------------
    # E. Invalid Input Validation Tests
    # --------------------------------------------------------------------------
    @pytest.mark.parametrize("invalid_level", [1, 2, 6, 7, "P2", "P6", "invalid", None, 3.5])
    def test_invalid_scale_level_raises_value_error(self, invalid_level) -> None:
        """Verify invalid scale levels raise an explicit ValueError."""
        module = ScaleAwareModule(embed_dim=64)
        x = torch.randn(1, 64, 40, 40, device="cpu")

        with pytest.raises(ValueError, match="Invalid scale_level"):
            module(x, scale_level=invalid_level)

    def test_non_4d_tensor_raises_value_error(self) -> None:
        """Verify non-4D inputs raise a clear ValueError."""
        module = ScaleAwareModule(embed_dim=64)

        x_3d = torch.randn(64, 40, 40, device="cpu")
        with pytest.raises(ValueError, match="Expected 4D input tensor"):
            module(x_3d, scale_level=3)

        x_5d = torch.randn(1, 1, 64, 40, 40, device="cpu")
        with pytest.raises(ValueError, match="Expected 4D input tensor"):
            module(x_5d, scale_level=3)

    def test_non_tensor_raises_type_error(self) -> None:
        """Verify passing non-tensor raises a TypeError."""
        module = ScaleAwareModule(embed_dim=64)
        with pytest.raises(TypeError, match="Expected input x to be a torch.Tensor"):
            module([1, 2, 3], scale_level=3)

    # --------------------------------------------------------------------------
    # F. Determinism Tests
    # --------------------------------------------------------------------------
    def test_initialization_and_forward_determinism(self) -> None:
        """Verify with fixed seed, repeated initialization and forward produce identical outputs."""
        torch.manual_seed(42)
        m1 = ScaleAwareModule(embed_dim=64)
        x1 = torch.randn(2, 128, 40, 40, device="cpu")
        out1 = m1(x1, scale_level=4)

        torch.manual_seed(42)
        m2 = ScaleAwareModule(embed_dim=64)
        x2 = torch.randn(2, 128, 40, 40, device="cpu")
        out2 = m2(x2, scale_level=4)

        assert torch.allclose(m1.embed_p4, m2.embed_p4), "Embeddings must be identical with same seed"
        assert torch.allclose(m1.scale_proj.weight, m2.scale_proj.weight), "Conv weights must be identical"
        assert torch.allclose(out1, out2), "Forward outputs must be bitwise close with same seed"

    # --------------------------------------------------------------------------
    # G. CPU Execution, Parameter Count & Dtype/Device Contract Tests
    # --------------------------------------------------------------------------
    def test_pure_cpu_execution(self) -> None:
        """Verify module functions fully on CPU without requiring CUDA accelerator."""
        module = ScaleAwareModule(embed_dim=64)
        x = torch.randn(2, 64, 20, 20, device="cpu")
        out = module(x, scale_level=5)
        assert out.device.type == "cpu"

    def test_parameter_count(self) -> None:
        """
        Verify the parameter count of ScaleAwareModule:
        - 3 scale embeddings of size 64 = 192
        - Conv2d(64, 64, 1): weight = 64*64 = 4096, bias = 64 -> 4160
        - Total: 4352 parameters
        """
        module = ScaleAwareModule(embed_dim=64, bias=True)
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

        expected_count = (3 * 64) + (64 * 64 * 1 * 1) + 64
        assert expected_count == 4352
        assert total_params == 4352, f"Expected 4352 parameters, got {total_params}"
        assert trainable_params == 4352, f"Expected 4352 trainable parameters, got {trainable_params}"

    def test_dtype_device_contract(self) -> None:
        """
        Verify the dtype and device contract:
        - Module is expected to be moved/cast together with the model (e.g. module.double()).
        - Output dtype matches input dtype.
        - Parameters are not silently moved or cast inside forward().
        - Dtype and device mismatches raise explicit errors.
        """
        module = ScaleAwareModule(embed_dim=64)

        # 1. Default float32 on CPU
        x_f32 = torch.randn(2, 256, 40, 40, dtype=torch.float32, device="cpu")
        out_f32 = module(x_f32, scale_level=4)
        assert out_f32.dtype == torch.float32
        assert out_f32.dtype == x_f32.dtype
        assert out_f32.device == x_f32.device

        # 2. float64 on CPU after calling module.double()
        module.double()
        x_f64 = torch.randn(2, 256, 40, 40, dtype=torch.float64, device="cpu")
        out_f64 = module(x_f64, scale_level=4)
        assert out_f64.dtype == torch.float64
        assert out_f64.dtype == x_f64.dtype
        assert out_f64.device == x_f64.device

        # 3. Dtype mismatch should raise TypeError rather than silently casting
        module.float()  # Reset module to float32
        with pytest.raises(TypeError, match="Dtype mismatch"):
            module(x_f64, scale_level=4)

        # 4. Optional CUDA execution (guarded; not required locally)
        if torch.cuda.is_available():
            module.cuda()
            x_cuda = torch.randn(2, 256, 40, 40, device="cuda")
            out_cuda = module(x_cuda, scale_level=4)
            assert out_cuda.device.type == "cuda"
            assert out_cuda.dtype == x_cuda.dtype

    # --------------------------------------------------------------------------
    # H. State Dict Namespace Hygiene Tests
    # --------------------------------------------------------------------------
    def test_state_dict_keys_and_no_alias(self) -> None:
        """
        Verify state_dict contains only the canonical scale_proj.* keys and
        does NOT contain duplicate alias keys like proj.weight or proj.bias.
        """
        module = ScaleAwareModule(embed_dim=64, bias=True)
        keys = set(module.state_dict().keys())

        expected_keys = {
            "scale_embeddings.3",
            "scale_embeddings.4",
            "scale_embeddings.5",
            "scale_proj.weight",
            "scale_proj.bias",
        }

        # Exact key set match
        assert keys == expected_keys, f"state_dict keys mismatch: {keys} vs {expected_keys}"

        # Explicit positive checks
        assert "scale_proj.weight" in keys
        assert "scale_proj.bias" in keys

        # Explicit negative checks for legacy/redundant alias keys
        assert "proj.weight" not in keys
        assert "proj.bias" not in keys

        # Verify no self.proj attribute exists on the module instance
        assert not hasattr(module, "proj"), (
            "ScaleAwareModule should not define self.proj alias to avoid duplicate state_dict registration"
        )
