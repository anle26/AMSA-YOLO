"""
Scale-Aware Module for AMSA-YOLO.

Paper Reference:
    AMSA-YOLO: Real-time object detection with adaptive multi-scale attention mechanism
    Neural Networks, Volume 197, Article 108545 (January 2026)

Mathematical Formulation:
    Input: X in R^(B x C x H x W)
    Scale levels: s in {3, 4, 5} corresponding to P3, P4, P5
    Learnable scale embeddings: E_s in R^D (default D = 64)
    Scale encoding: S = ScaleProj(E_s (x) 1_(H x W))
    Where:
        - E_s is spatially broadcast across (B, D, H, W)
        - ScaleProj is a learnable 1x1 convolution: Conv2d(D, D, kernel_size=1)
        - Output S in R^(B x D x H x W)
"""

from typing import Union
import torch
import torch.nn as nn


class ScaleAwareModule(nn.Module):
    """
    Scale-Aware Module for generating scale-adaptive feature encodings.

    Maintains distinct learnable scale embeddings for feature pyramid levels
    P3, P4, and P5. Given an input tensor X and an explicit scale_level s in {3, 4, 5},
    the module selects E_s in R^D, broadcasts it spatially across the batch and feature dimensions
    (B, D, H, W), and projects it via a learnable 1x1 convolution.

    Args:
        embed_dim (int): Dimension D of the scale embedding. Default: 64.
        bias (bool): Whether to include bias in the 1x1 convolution. Default: True.
    """

    VALID_SCALE_LEVELS = (3, 4, 5)

    def __init__(self, embed_dim: int = 64, bias: bool = True) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be a positive integer, got {embed_dim}")

        self.embed_dim = embed_dim

        # Distinct learnable scale embeddings for P3, P4, P5
        # Each E_s in R^D
        self.scale_embeddings = nn.ParameterDict({
            "3": nn.Parameter(torch.empty(embed_dim)),
            "4": nn.Parameter(torch.empty(embed_dim)),
            "5": nn.Parameter(torch.empty(embed_dim)),
        })

        # Learnable 1x1 convolution projection: ScaleProj: R^D -> R^D
        self.scale_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize scale embeddings and projection layer weights."""
        for param in self.scale_embeddings.values():
            nn.init.normal_(param, mean=0.0, std=0.02)

        nn.init.kaiming_uniform_(self.scale_proj.weight, a=1.0)
        if self.scale_proj.bias is not None:
            nn.init.zeros_(self.scale_proj.bias)

    @property
    def embed_p3(self) -> nn.Parameter:
        """Scale embedding for level P3 (s=3)."""
        return self.scale_embeddings["3"]

    @property
    def embed_p4(self) -> nn.Parameter:
        """Scale embedding for level P4 (s=4)."""
        return self.scale_embeddings["4"]

    @property
    def embed_p5(self) -> nn.Parameter:
        """Scale embedding for level P5 (s=5)."""
        return self.scale_embeddings["5"]

    def _normalize_scale_level(self, scale_level: Union[int, str]) -> str:
        """
        Validate and normalize scale_level to string key '3', '4', or '5'.

        Raises:
            ValueError: If scale_level is not one of {3, 4, 5}.
        """
        if isinstance(scale_level, int):
            if scale_level not in self.VALID_SCALE_LEVELS:
                raise ValueError(
                    f"Invalid scale_level: {scale_level}. Must be one of {set(self.VALID_SCALE_LEVELS)}."
                )
            return str(scale_level)

        if isinstance(scale_level, str):
            cleaned = scale_level.strip().upper().lstrip("P")
            if cleaned not in ("3", "4", "5"):
                raise ValueError(
                    f"Invalid scale_level: '{scale_level}'. Must be one of {{3, 4, 5}} or {{'P3', 'P4', 'P5'}}."
                )
            return cleaned

        raise ValueError(
            f"Invalid scale_level type: {type(scale_level).__name__}. Expected int or str in {{3, 4, 5}}."
        )

    def get_embedding(self, scale_level: Union[int, str]) -> nn.Parameter:
        """
        Retrieve the learnable scale embedding E_s for the specified scale level.

        Args:
            scale_level (int | str): Scale level identifier (3, 4, 5 or 'P3', 'P4', 'P5').

        Returns:
            nn.Parameter: Embedding vector E_s of shape (D,).
        """
        key = self._normalize_scale_level(scale_level)
        return self.scale_embeddings[key]

    def forward(self, x: torch.Tensor, scale_level: Union[int, str]) -> torch.Tensor:
        """
        Forward pass of the Scale-Aware Module.

        The module adheres to strict dtype and device contracts:
        - Input tensor x and module parameters must reside on the same device and share dtype.
        - Parameters are not silently moved or cast inside forward().

        Args:
            x (torch.Tensor): Input feature tensor of shape (B, C, H, W).
            scale_level (int | str): Explicit scale level in {3, 4, 5}.

        Returns:
            torch.Tensor: Scale encoding tensor S of shape (B, D, H, W).
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input x to be a torch.Tensor, got {type(x).__name__}")

        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input tensor with shape (B, C, H, W), got {x.dim()}D tensor with shape {tuple(x.shape)}"
            )

        # Enforce device and dtype contract: module and input must match
        if x.device != self.scale_proj.weight.device:
            raise RuntimeError(
                f"Device mismatch: input tensor is on {x.device}, but {self.__class__.__name__} is on {self.scale_proj.weight.device}"
            )
        if x.dtype != self.scale_proj.weight.dtype:
            raise TypeError(
                f"Dtype mismatch: input tensor has dtype {x.dtype}, but {self.__class__.__name__} has dtype {self.scale_proj.weight.dtype}. "
                "Ensure module is cast/moved together with the model (e.g. module.double() or module.to(dtype))."
            )

        # Retrieve learnable embedding E_s in R^D (Shape: (D,))
        e_s = self.get_embedding(scale_level)

        # Dynamic extraction of batch and spatial dimensions
        B, _, H, W = x.shape

        # Broadcast E_s across batch, height, and width: E_s (x) 1_(H x W)
        # Input E_s: (D,) -> Reshape: (1, D, 1, 1) -> Expand: (B, D, H, W)
        e_broadcast = e_s.view(1, self.embed_dim, 1, 1).expand(B, self.embed_dim, H, W)

        # Apply learnable 1x1 convolution projection: S = ScaleProj(E_s (x) 1_(H x W))
        # Input: (B, D, H, W) -> Output: (B, D, H, W)
        s = self.scale_proj(e_broadcast)

        return s
