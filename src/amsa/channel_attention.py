"""
Adaptive Channel Attention Module for AMSA-YOLO.

Paper Reference:
    AMSA-YOLO: Real-time object detection with adaptive multi-scale attention mechanism
    Neural Networks, Volume 197, Article 108545 (January 2026), Section 3.4.

Mathematical Formulation:
    1. Standard Channel Attention:
       A_channel = Sigmoid(MLP(GAP(X)) + MLP(GMP(X)))
       where X in R^(B x C x H x W),
       GAP and GMP perform global average and max pooling over (H, W),
       MLP is a two-layer shared MLP (C -> C_reduced -> C),
       A_channel in R^(B x C x 1 x 1).

    2. Scale Modulation:
       M_scale = Sigmoid(MLP(GAP(S)))
       where S in R^(B x D x H x W) is the scale encoding tensor from ScaleAwareModule,
       GAP pools S over (H, W) to R^(B x D x 1 x 1),
       MLP maps D -> D_hidden -> C,
       M_scale in R^(B x C x 1 x 1).

    3. Adaptive Channel Attention:
       A_adaptive = A_channel (x) M_scale
       where (x) is element-wise multiplication with automatic broadcasting,
       A_adaptive in R^(B x C x 1 x 1).

    4. Output:
       X_channel = X (x) A_adaptive
       where X_channel in R^(B x C x H x W).

Implementation Interpretations & Assumptions (Unresolved Paper Ambiguities Handled):
    - Shared Channel MLP: The paper expresses standard channel attention as
      A_channel = Sigmoid(MLP(GAP(X)) + MLP(GMP(X))). Passing both pooled features through a single
      shared module instance is an implementation interpretation consistent with the equation's
      reuse of MLP notation.
    - Channel MLP Reduction Ratio: The paper explicitly states a two-layer MLP with reduction and
      expansion layers, but does not specify the reduction ratio. We parameterize this via
      `channel_reduction` (default: 16), giving C_reduced = max(1, channels // channel_reduction).
    - Scale Modulation MLP: The paper specifies M_scale = Sigmoid(MLP(GAP(S))), mapping D -> C.
      It does NOT specify the number of hidden layers, hidden dimension, reduction ratio, or hidden
      activation. The two-layer D -> D_hidden -> C architecture is an implementation assumption,
      parameterized via `scale_reduction` (default: 4), giving D_hidden = max(1, scale_dim // scale_reduction).
    - Hidden Activations: Standard ReLU activation is used in the intermediate layer of both MLPs as
      an implementation assumption.
    - Bias: Configurable via `bias` (default: True) as an implementation assumption.
    - Precision & AMP Compatibility: We do not enforce strict parameter-dtype equality on activations,
      preserving compatibility with PyTorch Automatic Mixed Precision (torch.autocast) where master
      weights remain float32 while activations are float16/bfloat16. No manual tensor casting is performed.
"""

from typing import Dict, Tuple, Union
import torch
import torch.nn as nn


class AdaptiveChannelAttention(nn.Module):
    """
    Adaptive Channel Attention module for AMSA-YOLO.

    Recalibrates channel responses by combining standard channel attention (GAP + GMP through
    a shared two-layer MLP) with scale modulation conditioned on scale context S.

    Args:
        channels (int): Number of feature channels C in input tensor X.
        scale_dim (int): Embedding dimension D of scale tensor S. Default: 64.
        channel_reduction (int): Channel reduction factor for the channel attention MLP. Default: 16.
        scale_reduction (int): Dimension reduction factor for the scale modulation MLP. Default: 4.
        bias (bool): Whether to include bias terms in convolution/linear layers. Default: True.
    """

    def __init__(
        self,
        channels: int,
        scale_dim: int = 64,
        channel_reduction: int = 16,
        scale_reduction: int = 4,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be a positive integer, got {channels}")
        if scale_dim <= 0:
            raise ValueError(f"scale_dim must be a positive integer, got {scale_dim}")
        if channel_reduction <= 0:
            raise ValueError(f"channel_reduction must be a positive integer, got {channel_reduction}")
        if scale_reduction <= 0:
            raise ValueError(f"scale_reduction must be a positive integer, got {scale_reduction}")

        self.channels = channels
        self.scale_dim = scale_dim
        self.channel_reduction = channel_reduction
        self.scale_reduction = scale_reduction
        self.bias = bias

        # Compute reduced dimensions (ensuring at least 1)
        self.c_reduced = max(1, channels // channel_reduction)
        self.scale_hidden = max(1, scale_dim // scale_reduction)

        # ---------------------------------------------------------------------
        # 1. Shared Standard Channel MLP: C -> C_reduced -> C
        # Implemented with 1x1 convolutions operating natively on 4D pooled tensors (B, C, 1, 1)
        # ---------------------------------------------------------------------
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, self.c_reduced, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.c_reduced, channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        # ---------------------------------------------------------------------
        # 2. Scale Modulation MLP: D -> D_hidden -> C
        # Implemented with 1x1 convolutions operating natively on 4D pooled tensors (B, D, 1, 1)
        # ---------------------------------------------------------------------
        self.scale_mlp = nn.Sequential(
            nn.Conv2d(scale_dim, self.scale_hidden, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.scale_hidden, channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        # Sigmoid activation shared across attention gating
        self.sigmoid = nn.Sigmoid()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize convolution layer weights and biases."""
        for m in self.channel_mlp.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.scale_mlp.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compute_channel_attention(
        self, x: torch.Tensor, return_intermediates: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Compute standard channel attention map A_channel from input tensor X.

        Equation:
            A_channel = Sigmoid(MLP(GAP(X)) + MLP(GMP(X)))

        Args:
            x (torch.Tensor): Input feature tensor X of shape (B, C, H, W).
            return_intermediates (bool): If True, returns (a_channel, gap_x, gmp_x).

        Returns:
            torch.Tensor: A_channel of shape (B, C, 1, 1), range [0, 1].
            (Optional) Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: If return_intermediates is True.
        """
        gap_x = torch.mean(x, dim=(-2, -1), keepdim=True)
        gmp_x = torch.amax(x, dim=(-2, -1), keepdim=True)
        a_channel = self.sigmoid(self.channel_mlp(gap_x) + self.channel_mlp(gmp_x))
        if return_intermediates:
            return a_channel, gap_x, gmp_x
        return a_channel

    def compute_scale_modulation(
        self, s: torch.Tensor, return_intermediates: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute scale modulation map M_scale from scale tensor S.

        Equation:
            M_scale = Sigmoid(MLP(GAP(S)))

        Args:
            s (torch.Tensor): Scale tensor S of shape (B, D, H, W).
            return_intermediates (bool): If True, returns (m_scale, gap_s).

        Returns:
            torch.Tensor: M_scale of shape (B, C, 1, 1), range [0, 1].
            (Optional) Tuple[torch.Tensor, torch.Tensor]: If return_intermediates is True.
        """
        gap_s = torch.mean(s, dim=(-2, -1), keepdim=True)
        m_scale = self.sigmoid(self.scale_mlp(gap_s))
        if return_intermediates:
            return m_scale, gap_s
        return m_scale

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        return_attention_maps: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass of Adaptive Channel Attention.

        Fuses standard channel attention A_channel with scale modulation M_scale,
        then modulates input X:
            A_adaptive = A_channel (x) M_scale
            X_channel = X (x) A_adaptive

        Args:
            x (torch.Tensor): Input feature tensor X of shape (B, C, H, W).
            s (torch.Tensor): Scale encoding tensor S of shape (B, D, H, W).
            return_attention_maps (bool): If True, also returns intermediate attention tensors.

        Returns:
            torch.Tensor: Modulated output feature tensor X_channel of shape (B, C, H, W).
            (Optional) Dict[str, torch.Tensor]: Intermediate maps if return_attention_maps=True.
        """
        # Type validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected x to be a torch.Tensor, got {type(x).__name__}")
        if not isinstance(s, torch.Tensor):
            raise TypeError(f"Expected s to be a torch.Tensor, got {type(s).__name__}")

        # Dimensionality validation
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor for x with shape (B, C, H, W), got {x.dim()}D tensor with shape {tuple(x.shape)}"
            )
        if s.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor for s with shape (B, D, H, W), got {s.dim()}D tensor with shape {tuple(s.shape)}"
            )

        # Device contracts (no cross-device execution or silent device migration)
        target_device = self.channel_mlp[0].weight.device

        if x.device != target_device:
            raise RuntimeError(
                f"Device mismatch: x is on {x.device}, but module is on {target_device}"
            )
        if s.device != target_device:
            raise RuntimeError(
                f"Device mismatch: s is on {s.device}, but module is on {target_device}"
            )

        # Channel & scale_dim validation
        B, C, H, W = x.shape
        B_s, D, H_s, W_s = s.shape

        if C != self.channels:
            raise ValueError(
                f"Channel mismatch: x has {C} channels, but module was configured with channels={self.channels}"
            )
        if D != self.scale_dim:
            raise ValueError(
                f"Scale dim mismatch: s has {D} scale dimension, but module was configured with scale_dim={self.scale_dim}"
            )
        if B != B_s:
            raise ValueError(
                f"Batch size mismatch: x has batch size {B}, but s has batch size {B_s}"
            )
        if (H, W) != (H_s, W_s):
            raise ValueError(
                f"Spatial resolution mismatch: x has spatial size ({H}, {W}), but s has ({H_s}, {W_s})"
            )

        # 1. Standard channel attention: A_channel in R^(B x C x 1 x 1)
        a_channel, gap_x, gmp_x = self.compute_channel_attention(x, return_intermediates=True)

        # 2. Scale modulation: M_scale in R^(B x C x 1 x 1)
        m_scale, gap_s = self.compute_scale_modulation(s, return_intermediates=True)

        # 3. Adaptive channel attention: A_adaptive = A_channel * M_scale in R^(B x C x 1 x 1)
        a_adaptive = a_channel * m_scale

        # 4. Output recalibration: X_channel = X * A_adaptive in R^(B x C x H x W)
        x_channel = x * a_adaptive

        if return_attention_maps:
            maps = {
                "gap_x": gap_x,
                "gmp_x": gmp_x,
                "a_channel": a_channel,
                "gap_s": gap_s,
                "m_scale": m_scale,
                "a_adaptive": a_adaptive,
            }
            return x_channel, maps

        return x_channel
