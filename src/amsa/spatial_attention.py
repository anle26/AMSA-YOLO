"""
Adaptive Spatial Attention Module for AMSA-YOLO.

Paper Reference:
    AMSA-YOLO: Real-time object detection with adaptive multi-scale attention mechanism
    Neural Networks, Volume 197, Article 108545 (January 2026), Section 3.3.

Mathematical Formulation:
    1. Scale-Dependent Weights:
       W_local, W_global = Sigmoid(GAP(MLP(S)))
       where S in R^(B x D x H x W) is the scale encoding from ScaleAwareModule,
       W_local in R^(B x 1 x 1 x 1), W_global in R^(B x 1 x 1 x 1).

    2. Local Attention:
       A_local = Sigmoid(Conv3x3(Conv1x1(X)))
       where X in R^(B x C x H x W), Conv1x1 performs dimensionality reduction,
       A_local in R^(B x C x H x W).

    3. Global Attention:
       A_global = Sigmoid(GAP(X) + GMP(X))
       where GAP and GMP perform global average and max pooling over (H, W),
       A_global in R^(B x C x 1 x 1).

    4. Adaptive Fusion:
       A_spatial = W_local (x) A_local + W_global (x) Broadcast(A_global)
       where (x) is element-wise multiplication with automatic broadcasting,
       A_spatial in R^(B x C x H x W).

    5. Output:
       X_spatial = X (x) A_spatial
       where X_spatial in R^(B x C x H x W).

Implementation Assumptions (Unresolved Paper Ambiguities Handled):
    - Scale MLP Architecture: The paper does not specify the hidden dimension or reduction ratio
      for MLP(S). We parameterize this via `scale_reduction` (default: 4), giving hidden dimension
      D_hidden = max(1, D // scale_reduction). The MLP is implemented as 1x1 Conv -> ReLU -> 1x1 Conv(to 2 channels).
    - Local Conv Reduction: The paper states Conv1x1 reduces dimensionality but omits the exact ratio.
      We parameterize this via `local_reduction` (default: 4), giving C_reduced = max(1, C // local_reduction).
    - Local Convolution Chain: The paper literally defines A_local = Sigmoid(Conv3x3(Conv1x1(X))).
      No intermediate activation (ReLU/SiLU) or normalization (BatchNorm) is placed between Conv1x1 and Conv3x3.
    - Conv3x3 Padding: Standard padding=1 with stride=1 is used to strictly preserve spatial dimensions (H, W).
    - Fusion Broadcasting: Standard PyTorch tensor broadcasting is utilized without materializing repeated tensors.
    - Precision & AMP Compatibility: We do not enforce strict parameter-dtype equality on activations,
      preserving compatibility with PyTorch Automatic Mixed Precision (torch.autocast) where master
      weights remain float32 while activations are float16/bfloat16. No manual tensor casting is performed.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


class AdaptiveSpatialAttention(nn.Module):
    """
    Adaptive Spatial Attention module for AMSA-YOLO.

    Combines scale-aware weights W_local and W_global with fine-grained local spatial attention
    A_local and contextual global spatial attention A_global to recalibrate input features X.

    Args:
        channels (int): Number of feature channels C in input tensor X.
        scale_dim (int): Embedding dimension D of scale tensor S. Default: 64.
        local_reduction (int): Channel reduction factor for the local attention branch. Default: 4.
        scale_reduction (int): Dimension reduction factor for the scale MLP branch. Default: 4.
        bias (bool): Whether to include bias terms in convolution layers. Default: True.
    """

    def __init__(
        self,
        channels: int,
        scale_dim: int = 64,
        local_reduction: int = 4,
        scale_reduction: int = 4,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be a positive integer, got {channels}")
        if scale_dim <= 0:
            raise ValueError(f"scale_dim must be a positive integer, got {scale_dim}")
        if local_reduction <= 0:
            raise ValueError(f"local_reduction must be a positive integer, got {local_reduction}")
        if scale_reduction <= 0:
            raise ValueError(f"scale_reduction must be a positive integer, got {scale_reduction}")

        self.channels = channels
        self.scale_dim = scale_dim
        self.local_reduction = local_reduction
        self.scale_reduction = scale_reduction
        self.bias = bias

        # Compute reduced dimensions (ensuring at least 1)
        self.c_reduced = max(1, channels // local_reduction)
        self.scale_hidden = max(1, scale_dim // scale_reduction)

        # ---------------------------------------------------------------------
        # 1. Scale-Dependent Weight Branch: S -> MLP -> GAP -> Sigmoid -> (W_local, W_global)
        # S in R^(B x D x H x W)
        # MLP: Conv1x1(D -> D_hidden) -> ReLU -> Conv1x1(D_hidden -> 2)
        # ---------------------------------------------------------------------
        self.scale_mlp = nn.Sequential(
            nn.Conv2d(scale_dim, self.scale_hidden, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.scale_hidden, 2, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        # ---------------------------------------------------------------------
        # 2. Local Attention Branch: X -> Conv1x1 -> Conv3x3 -> Sigmoid -> A_local
        # X in R^(B x C x H x W)
        # Conv1x1: C -> C_reduced
        # Conv3x3: C_reduced -> C (preserves H, W via padding=1)
        # Follows paper literally: A_local = Sigmoid(Conv3x3(Conv1x1(X)))
        # ---------------------------------------------------------------------
        self.local_conv1 = nn.Conv2d(
            channels, self.c_reduced, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.local_conv2 = nn.Conv2d(
            self.c_reduced, channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

        # Sigmoid activation shared across attention gating
        self.sigmoid = nn.Sigmoid()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize convolution layer weights and biases."""
        # Scale MLP initialization
        for m in self.scale_mlp.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Local branch initialization
        nn.init.kaiming_uniform_(self.local_conv1.weight, a=1.0)
        if self.local_conv1.bias is not None:
            nn.init.zeros_(self.local_conv1.bias)

        nn.init.kaiming_uniform_(self.local_conv2.weight, a=1.0)
        if self.local_conv2.bias is not None:
            nn.init.zeros_(self.local_conv2.bias)

    def compute_scale_weights(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scale-dependent fusion weights W_local and W_global from scale tensor S.

        Equation:
            W_local, W_global = Sigmoid(GAP(MLP(S)))

        Args:
            s (torch.Tensor): Scale tensor S of shape (B, D, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - W_local of shape (B, 1, 1, 1), range [0, 1]
                - W_global of shape (B, 1, 1, 1), range [0, 1]
        """
        # S: (B, D, H, W) -> MLP -> (B, 2, H, W)
        mlp_out = self.scale_mlp(s)
        # GAP over spatial dimensions (H, W) -> (B, 2, 1, 1)
        gap_out = torch.mean(mlp_out, dim=(-2, -1), keepdim=True)
        # Sigmoid -> (B, 2, 1, 1)
        weights = self.sigmoid(gap_out)
        # Split along channel dimension into W_local (B, 1, 1, 1) and W_global (B, 1, 1, 1)
        w_local, w_global = torch.chunk(weights, chunks=2, dim=1)
        return w_local, w_global

    def compute_local_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute local spatial attention map A_local from input tensor X.

        Equation:
            A_local = Sigmoid(Conv3x3(Conv1x1(X)))

        Args:
            x (torch.Tensor): Input feature tensor X of shape (B, C, H, W).

        Returns:
            torch.Tensor: A_local of shape (B, C, H, W), range [0, 1].
        """
        # X: (B, C, H, W) -> Conv1x1 -> (B, C_reduced, H, W)
        feat = self.local_conv1(x)
        # (B, C_reduced, H, W) -> Conv3x3 -> (B, C, H, W)
        feat = self.local_conv2(feat)
        # Sigmoid -> (B, C, H, W)
        return self.sigmoid(feat)

    def compute_global_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute global spatial attention map A_global from input tensor X.

        Equation:
            A_global = Sigmoid(GAP(X) + GMP(X))

        Args:
            x (torch.Tensor): Input feature tensor X of shape (B, C, H, W).

        Returns:
            torch.Tensor: A_global of shape (B, C, 1, 1), range [0, 1].
        """
        # GAP: mean over (H, W) -> (B, C, 1, 1)
        gap = torch.mean(x, dim=(-2, -1), keepdim=True)
        # GMP: max over (H, W) -> (B, C, 1, 1)
        gmp = torch.amax(x, dim=(-2, -1), keepdim=True)
        # Sigmoid(GAP(X) + GMP(X)) -> (B, C, 1, 1)
        return self.sigmoid(gap + gmp)

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        return_attention_maps: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass of Adaptive Spatial Attention.

        Fuses local spatial attention and global spatial attention conditioned on scale
        weights, then modulates input X:
            A_spatial = W_local (x) A_local + W_global (x) A_global
            X_spatial = X (x) A_spatial

        Args:
            x (torch.Tensor): Input feature tensor X of shape (B, C, H, W).
            s (torch.Tensor): Scale encoding tensor S of shape (B, D, H, W).
            return_attention_maps (bool): If True, also returns intermediate attention tensors.

        Returns:
            torch.Tensor: Modulated output feature tensor X_spatial of shape (B, C, H, W).
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
        target_device = self.local_conv1.weight.device

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

        # 1. Scale-dependent weights: W_local, W_global in R^(B x 1 x 1 x 1)
        w_local, w_global = self.compute_scale_weights(s)

        # 2. Local attention: A_local in R^(B x C x H x W)
        a_local = self.compute_local_attention(x)

        # 3. Global attention: A_global in R^(B x C x 1 x 1)
        a_global = self.compute_global_attention(x)

        # 4. Adaptive spatial fusion:
        # PyTorch broadcasting automatically expands:
        # - w_local (B, 1, 1, 1) * a_local (B, C, H, W) -> (B, C, H, W)
        # - w_global (B, 1, 1, 1) * a_global (B, C, 1, 1) -> (B, C, 1, 1)
        # - addition broadcasts (B, C, 1, 1) to (B, C, H, W)
        a_spatial = (w_local * a_local) + (w_global * a_global)

        # 5. Output modulation: X_spatial = X (x) A_spatial
        x_spatial = x * a_spatial

        if return_attention_maps:
            maps = {
                "w_local": w_local,
                "w_global": w_global,
                "a_local": a_local,
                "a_global": a_global,
                "a_spatial": a_spatial,
            }
            return x_spatial, maps

        return x_spatial
