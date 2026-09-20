"""
Feature Fusion Module for AMSA-YOLO.

Paper Reference:
    AMSA-YOLO: Real-time object detection with adaptive multi-scale attention mechanism
    Neural Networks, Volume 197, Article 108545 (January 2026), Section 3.5.

Mathematical Formulation:
    1. Feature Concatenation:
       X_concat = Concat(X_channel, X_spatial)
       where X_channel in R^(B x C x H x W), X_spatial in R^(B x C x H x W),
       Concat is along the channel dimension (dim=1),
       X_concat in R^(B x 2C x H x W).

    2. Channel Compression:
       X_fused = Conv1x1(X_concat)
       where Conv1x1 maps 2C -> C,
       X_fused in R^(B x C x H x W).

    3. Residual Fusion & Normalization:
       X_output = ReLU(BN(X_fused) + X)
       where BN is BatchNorm2d(C), X is the original input tensor,
       X_output in R^(B x C x H x W).

Implementation Interpretations & Assumptions:
    - Bias: The paper does not specify whether Conv1x1 includes a bias parameter.
      We expose `bias: bool = True` as a configurable constructor parameter.
    - No Extra Layers: In strict adherence to Section 3.5, no extra SiLU, Dropout,
      additional convolutions, normalization, or gating layers are introduced.
    - Precision & AMP Compatibility: We do not enforce rigid parameter-dtype checks,
      preserving compatibility with PyTorch Automatic Mixed Precision (torch.autocast)
      where master weights remain float32 while activations are float16/bfloat16.
"""

from typing import Dict, Tuple, Union
import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """
    Feature Fusion module for AMSA-YOLO.

    Fuses channel-recalibrated features X_channel and spatial-recalibrated features X_spatial
    via concatenation, 1x1 convolution compression, batch normalization, residual addition
    with original features X, and ReLU activation.

    Args:
        channels (int): Number of feature channels C in input tensors.
        bias (bool): Whether to include bias in the 1x1 convolution. Default: True.
    """

    def __init__(self, channels: int, bias: bool = True) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be a positive integer, got {channels}")

        self.channels = channels
        self.bias = bias

        # 1. 1x1 Convolution: 2C -> C
        self.conv = nn.Conv2d(
            in_channels=2 * channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        # 2. Batch Normalization: BN(C)
        self.bn = nn.BatchNorm2d(channels)

        # 3. Activation: ReLU
        self.relu = nn.ReLU(inplace=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize convolution and batch normalization parameters."""
        nn.init.kaiming_uniform_(self.conv.weight, a=1.0)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(
        self,
        x: torch.Tensor,
        x_channel: torch.Tensor,
        x_spatial: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass of Feature Fusion.

        Fuses channel and spatial attention features with original features:
            X_concat = Concat(X_channel, X_spatial)
            X_fused = Conv1x1(X_concat)
            X_output = ReLU(BN(X_fused) + X)

        Args:
            x (torch.Tensor): Original input feature tensor X of shape (B, C, H, W).
            x_channel (torch.Tensor): Channel-modulated feature tensor of shape (B, C, H, W).
            x_spatial (torch.Tensor): Spatial-modulated feature tensor of shape (B, C, H, W).
            return_intermediates (bool): If True, also returns intermediate tensors.

        Returns:
            torch.Tensor: Fused output feature tensor X_output of shape (B, C, H, W).
            (Optional) Dict[str, torch.Tensor]: Intermediate tensors if return_intermediates=True.
        """
        # Type validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected x to be a torch.Tensor, got {type(x).__name__}")
        if not isinstance(x_channel, torch.Tensor):
            raise TypeError(f"Expected x_channel to be a torch.Tensor, got {type(x_channel).__name__}")
        if not isinstance(x_spatial, torch.Tensor):
            raise TypeError(f"Expected x_spatial to be a torch.Tensor, got {type(x_spatial).__name__}")

        # Dimensionality validation
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor for x with shape (B, C, H, W), got {x.dim()}D tensor with shape {tuple(x.shape)}"
            )
        if x_channel.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor for x_channel with shape (B, C, H, W), got {x_channel.dim()}D tensor with shape {tuple(x_channel.shape)}"
            )
        if x_spatial.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor for x_spatial with shape (B, C, H, W), got {x_spatial.dim()}D tensor with shape {tuple(x_spatial.shape)}"
            )

        # Device contracts (no cross-device execution or silent migration)
        target_device = self.conv.weight.device
        if x.device != target_device:
            raise RuntimeError(
                f"Device mismatch: x is on {x.device}, but module is on {target_device}"
            )
        if x_channel.device != target_device:
            raise RuntimeError(
                f"Device mismatch: x_channel is on {x_channel.device}, but module is on {target_device}"
            )
        if x_spatial.device != target_device:
            raise RuntimeError(
                f"Device mismatch: x_spatial is on {x_spatial.device}, but module is on {target_device}"
            )

        # Shape validation
        B, C, H, W = x.shape
        B_c, C_c, H_c, W_c = x_channel.shape
        B_s, C_s, H_s, W_s = x_spatial.shape

        if C != self.channels:
            raise ValueError(
                f"Channel mismatch: x has {C} channels, but module was configured with channels={self.channels}"
            )
        if C_c != self.channels:
            raise ValueError(
                f"Channel mismatch: x_channel has {C_c} channels, but module was configured with channels={self.channels}"
            )
        if C_s != self.channels:
            raise ValueError(
                f"Channel mismatch: x_spatial has {C_s} channels, but module was configured with channels={self.channels}"
            )

        if not (B == B_c == B_s):
            raise ValueError(
                f"Batch size mismatch: x has batch {B}, x_channel has {B_c}, x_spatial has {B_s}"
            )
        if not ((H, W) == (H_c, W_c) == (H_s, W_s)):
            raise ValueError(
                f"Spatial resolution mismatch: x has {(H, W)}, x_channel has {(H_c, W_c)}, x_spatial has {(H_s, W_s)}"
            )

        # 1. Feature concatenation: Concat(X_channel, X_spatial) -> (B, 2C, H, W)
        x_concat = torch.cat([x_channel, x_spatial], dim=1)

        # 2. Channel compression: Conv1x1 -> (B, C, H, W)
        x_fused = self.conv(x_concat)

        # 3. Residual fusion & normalization: ReLU(BN(X_fused) + X) -> (B, C, H, W)
        x_bn = self.bn(x_fused)
        x_residual = x_bn + x
        x_output = self.relu(x_residual)

        if return_intermediates:
            intermediates = {
                "x_concat": x_concat,
                "x_fused": x_fused,
                "x_bn": x_bn,
                "x_residual": x_residual,
                "x_output": x_output,
            }
            return x_output, intermediates

        return x_output
