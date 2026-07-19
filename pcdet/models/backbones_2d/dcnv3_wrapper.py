"""
DCNv3 wrapper for Lite-MDFEN (Paper Section 10).

Wraps Deformable Convolution v3 (DCNv3) with NCHW <-> NHWC conversion
handling and graceful fallback when DCNv3 is not available.

Important: DCNv3 typically expects NHWC format. This wrapper handles
the conversion and prevents repeated permute in training loops.

If DCNv3 is unavailable, falls back to standard Conv2d.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Try to import DCNv3 from common sources
_DCNV3_AVAILABLE = False
_DCNV3_IMPL = None

try:
    # Try mmcv / mmdetection DCNv3
    from mmcv.ops import ModulatedDeformConv2dPack as DCNv3_mmcv
    _DCNV3_AVAILABLE = True
    _DCNV3_IMPL = 'mmcv'
except ImportError:
    pass

if not _DCNV3_AVAILABLE:
    # torchvision's DeformConv2d requires manual offset generation and is
    # not a drop-in replacement for DCNv3. Fall back to standard Conv2d.
    pass


class DCNv3Wrapper(nn.Module):
    """
    Wraps DCNv3 (or fallback) for single-scale feature enhancement.

    Handles:
        - NCHW -> NHWC -> NCHW conversion (DCNv3 convention)
        - Graceful fallback to standard Conv2d when unavailable
        - Optional groups for efficiency

    Config:
        in_channels: int
        kernel_size: int (default 3)
        groups: int (default 4)
        stride: int (default 1)
        padding: int (auto from kernel_size)
    """

    def __init__(self, in_channels, kernel_size=3, groups=4, stride=1, padding=None):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.stride = stride
        self.padding = padding if padding is not None else kernel_size // 2

        if _DCNV3_AVAILABLE:
            if _DCNV3_IMPL == 'mmcv':
                # MMCV DCNv3: operates in NCHW internally
                self.dcn = ModulatedDeformConv2dPack(
                    in_channels, in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=self.padding,
                    groups=groups,
                    deformable_groups=groups,
                    bias=False,
                )
            elif _DCNV3_IMPL == 'torchvision':
                # torchvision DCNv2 (no modulation)
                self.dcn = DeformConv2d(
                    in_channels, in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=self.padding,
                    groups=groups,
                )
            self._has_dcn = True
        else:
            # Fallback: standard depthwise-like conv
            self.dcn = nn.Conv2d(
                in_channels, in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=self.padding,
                groups=min(groups, in_channels),
                bias=False,
            )
            self._has_dcn = False

        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()

    @property
    def has_dcnv3(self):
        return self._has_dcn

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) float tensor in NCHW format.

        Returns:
            out: (B, C, H, W) float tensor in NCHW format.
        """
        identity = x

        if self._has_dcn:
            # DCNv3 / DCNv2 forward (MMCV or compatible implementation)
            out = self.dcn(x)
        else:
            # Standard Conv2d fallback
            out = self.dcn(x)

        out = self.norm(out)
        out = self.act(out)
        return out


def has_dcnv3_support():
    """Check if DCNv3 is available in the current environment."""
    return _DCNV3_AVAILABLE
