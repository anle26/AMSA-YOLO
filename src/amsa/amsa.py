"""
Adaptive Multi-Scale Attention (AMSA) Module for AMSA-YOLO.

Paper Reference:
    AMSA-YOLO: Real-time object detection with adaptive multi-scale attention mechanism
    Neural Networks, Volume 197, Article 108545 (January 2026), Section 3.

AMSA Orchestration Pipeline:
    Input: X in R^(B x C x H x W), scale_level s in {3, 4, 5}
    1. Scale-Aware Encoding:
       S = ScaleAwareModule(X, scale_level)
       where S in R^(B x D x H x W), default D = 64.
    2. Adaptive Spatial Attention:
       X_spatial = AdaptiveSpatialAttention(X, S)
       where X_spatial in R^(B x C x H x W).
    3. Adaptive Channel Attention:
       X_channel = AdaptiveChannelAttention(X, S)
       where X_channel in R^(B x C x H x W).
    4. Feature Fusion:
       X_output = FeatureFusion(X, X_channel, X_spatial)
       where X_output in R^(B x C x H x W).

Architectural Boundaries:
    This wrapper orchestrates the four standalone sub-modules without altering,
    duplicating, or injecting additional convolutional, normalization, or gating operations.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.amsa.channel_attention import AdaptiveChannelAttention
from src.amsa.fusion import FeatureFusion
from src.amsa.scale_aware import ScaleAwareModule
from src.amsa.spatial_attention import AdaptiveSpatialAttention


class AMSAModule(nn.Module):
    """
    Adaptive Multi-Scale Attention (AMSA) module composing:
        - ScaleAwareModule
        - AdaptiveSpatialAttention
        - AdaptiveChannelAttention
        - FeatureFusion

    Args:
        channels (int): Number of feature channels C in input tensor X.
        scale_level (Optional[Union[int, str]]): Default pyramid level identifier (3, 4, 5).
        scale_dim (int): Embedding dimension D of scale tensor S. Default: 64.
        spatial_local_reduction (int): Channel reduction for spatial local attention. Default: 4.
        spatial_scale_reduction (int): Dimension reduction for spatial scale MLP. Default: 4.
        channel_reduction (int): Channel reduction for channel attention MLP. Default: 16.
        channel_scale_reduction (int): Dimension reduction for channel scale MLP. Default: 4.
        spatial_bias (bool): Whether to include bias in spatial attention convolutions. Default: True.
        channel_bias (bool): Whether to include bias in channel attention convolutions. Default: True.
        fusion_bias (bool): Whether to include bias in feature fusion 1x1 convolution. Default: True.
    """

    def __init__(
        self,
        channels: int,
        scale_level: Optional[Union[int, str]] = None,
        scale_dim: int = 64,
        spatial_local_reduction: int = 4,
        spatial_scale_reduction: int = 4,
        channel_reduction: int = 16,
        channel_scale_reduction: int = 4,
        spatial_bias: bool = True,
        channel_bias: bool = True,
        fusion_bias: bool = True,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be a positive integer, got {channels}")
        if scale_dim <= 0:
            raise ValueError(f"scale_dim must be a positive integer, got {scale_dim}")

        self.channels = channels
        self.scale_dim = scale_dim

        # 1. Scale-Aware Module (Section 3.2)
        self.scale_aware = ScaleAwareModule(
            embed_dim=scale_dim,
        )

        # Validate and store constructor scale_level if provided
        if scale_level is not None:
            self.scale_aware._normalize_scale_level(scale_level)
        self.scale_level = scale_level

        # 2. Adaptive Spatial Attention (Section 3.3)
        self.spatial_attention = AdaptiveSpatialAttention(
            channels=channels,
            scale_dim=scale_dim,
            local_reduction=spatial_local_reduction,
            scale_reduction=spatial_scale_reduction,
            bias=spatial_bias,
        )

        # 3. Adaptive Channel Attention (Section 3.4)
        self.channel_attention = AdaptiveChannelAttention(
            channels=channels,
            scale_dim=scale_dim,
            channel_reduction=channel_reduction,
            scale_reduction=channel_scale_reduction,
            bias=channel_bias,
        )

        # 4. Feature Fusion (Section 3.5)
        self.fusion = FeatureFusion(
            channels=channels,
            bias=fusion_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        scale_level: Optional[Union[int, str]] = None,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass of AMSAModule.

        Pipeline:
            S = self.scale_aware(x, resolved_scale_level)
            X_spatial = self.spatial_attention(x, S)
            X_channel = self.channel_attention(x, S)
            X_output = self.fusion(x, X_channel, X_spatial)

        Args:
            x (torch.Tensor): Input feature tensor X of shape (B, C, H, W).
            scale_level (Optional[Union[int, str]]): Pyramid level identifier (3, 4, 5).
                If provided, overrides constructor scale_level. If neither is provided, raises ValueError.
            return_intermediates (bool): If True, also returns dictionary of intermediate tensors.

        Returns:
            torch.Tensor: Calibrated output feature tensor X_output of shape (B, C, H, W).
            (Optional) Dict[str, torch.Tensor]: Intermediates if return_intermediates=True.
        """
        # Resolution rule:
        # - explicit forward scale_level wins
        # - otherwise use constructor scale_level
        # - raise ValueError if neither exists
        resolved_scale_level = scale_level if scale_level is not None else self.scale_level
        if resolved_scale_level is None:
            raise ValueError(
                "scale_level must be specified either in AMSAModule constructor or in forward()"
            )
        # Type validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected x to be a torch.Tensor, got {type(x).__name__}")

        # Dimensionality validation
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor for x with shape (B, C, H, W), got {x.dim()}D tensor with shape {tuple(x.shape)}"
            )

        # Channel validation
        if x.shape[1] != self.channels:
            raise ValueError(
                f"Channel mismatch: x has {x.shape[1]} channels, but module was configured with channels={self.channels}"
            )

        # 1. Scale-aware descriptor encoding
        s = self.scale_aware(x, resolved_scale_level)

        # 2. Adaptive spatial attention modulation
        x_spatial = self.spatial_attention(x, s)

        # 3. Adaptive channel attention modulation
        x_channel = self.channel_attention(x, s)

        # 4. Feature fusion
        x_output = self.fusion(x, x_channel, x_spatial)

        if return_intermediates:
            intermediates = {
                "scale_encoding": s,
                "x_spatial": x_spatial,
                "x_channel": x_channel,
            }
            return x_output, intermediates

        return x_output
