"""
RepDWC: Re-parameterizable Depth-Wise Convolution Block (Paper Section 9).

Training structure (multi-branch):
    Depth-wise stage:
        y' = Act[ BN(D_3x3(x)) + BN(D_1x1(x)) ]     -- eq (9.1a)
    Point-wise stage:
        y  = Act[ BN(P_3x3(y')) + BN(P_1x1(y')) + BN(y') ]  -- eq (9.1b)

Deployment structure:
    Single 3x3 convolution (depth-wise) + Single 1x1 or 3x3 (point-wise)
    obtained by fusing Conv+BN and merging parallel branches.

Reference: RadarNeXt (Springer Nature, 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reparameterize import (
    fuse_conv_bn,
    pad_1x1_to_3x3,
    identity_to_conv,
    merge_branches,
)


class RepDWCBlock(nn.Module):
    """
    Single RepDWC block with re-parameterizable depth-wise + point-wise stages.

    Architecture:
        Input: x (B, C, H, W)

        -- Depth-wise stage --
        branch_1: D_3x3(x) -> BN -> |
        branch_2: D_1x1(x) -> BN -> |
        y' = ReLU(branch_1 + branch_2)

        -- Point-wise stage --
        branch_3: P_3x3(y') -> BN -> |
        branch_4: P_1x1(y') -> BN -> |
        branch_5: y' -> BN ->         |  (identity if C_in == C_out)
        y = ReLU(branch_3 + branch_4 + branch_5)

    For stride > 1, only the first depth-wise stage uses stride.

    Args:
        in_channels: int.
        out_channels: int.
        stride: int, spatial stride.
        use_identity: bool, include identity branch in point-wise stage.
        deploy_mode: bool, use fused single-path weights.
    """

    def __init__(self, in_channels, out_channels, stride=1, deploy_mode=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.deploy_mode = deploy_mode

        if deploy_mode:
            # Deployment: single fused depth-wise + point-wise convs
            self.deploy_dw_conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3,
                stride=stride, padding=1, groups=in_channels, bias=True
            )
            self.deploy_pw_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                stride=1, padding=0, bias=True
            )
            return

        # === Training branches ===

        # Depth-wise stage branches
        self.dw_3x3 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.dw_3x3_bn = nn.BatchNorm2d(in_channels)

        self.dw_1x1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1,
            stride=stride, padding=0, groups=in_channels, bias=False
        )
        self.dw_1x1_bn = nn.BatchNorm2d(in_channels)

        # Point-wise stage branches
        self.pw_3x3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=1, padding=1, groups=1, bias=False
        )
        self.pw_3x3_bn = nn.BatchNorm2d(out_channels)

        self.pw_1x1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, groups=1, bias=False
        )
        self.pw_1x1_bn = nn.BatchNorm2d(out_channels)

        # Identity branch (only if C_in == C_out and stride == 1)
        self.use_identity = (in_channels == out_channels and stride == 1)
        if self.use_identity:
            self.id_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.deploy_mode:
            x = F.relu(self.deploy_dw_conv(x))
            x = F.relu(self.deploy_pw_conv(x))
            return x

        # === Depth-wise stage ===
        dw_3 = self.dw_3x3_bn(self.dw_3x3(x))
        dw_1 = self.dw_1x1_bn(self.dw_1x1(x))
        y_dash = F.relu(dw_3 + dw_1)

        # === Point-wise stage ===
        pw_3 = self.pw_3x3_bn(self.pw_3x3(y_dash))
        pw_1 = self.pw_1x1_bn(self.pw_1x1(y_dash))
        y_sum = pw_3 + pw_1

        if self.use_identity:
            y_sum = y_sum + self.id_bn(y_dash)

        y = F.relu(y_sum)
        return y

    def get_deploy_weights(self):
        """
        Compute fused deployment weights from training branches.

        Returns:
            dw_conv: nn.Conv2d, fused depth-wise convolution.
            pw_conv: nn.Conv2d, fused point-wise convolution.
        """
        assert not self.deploy_mode, "Already in deploy mode"

        # === Depth-wise stage fusion ===
        # Fuse each branch: Conv + BN
        dw_3_fused = fuse_conv_bn(self.dw_3x3, self.dw_3x3_bn)
        dw_1_fused = fuse_conv_bn(self.dw_1x1, self.dw_1x1_bn)

        # Pad 1x1 kernel to 3x3
        dw_1_kernel = pad_1x1_to_3x3(dw_1_fused.weight.data)

        # Merge depth-wise branches
        dw_kernel, dw_bias = merge_branches([
            (dw_3_fused.weight.data, dw_3_fused.bias.data),
            (dw_1_kernel, dw_1_fused.bias.data),
        ])

        dw_conv = nn.Conv2d(
            self.in_channels, self.in_channels,
            kernel_size=3, stride=self.stride, padding=1,
            groups=self.in_channels, bias=True,
        )
        dw_conv.weight.data = dw_kernel
        dw_conv.bias.data = dw_bias

        # === Point-wise stage fusion ===
        pw_3_fused = fuse_conv_bn(self.pw_3x3, self.pw_3x3_bn)
        pw_1_fused = fuse_conv_bn(self.pw_1x1, self.pw_1x1_bn)

        # Pad 1x1 to 3x3
        pw_1_kernel = pad_1x1_to_3x3(pw_1_fused.weight.data)

        branches = [
            (pw_3_fused.weight.data, pw_3_fused.bias.data),
            (pw_1_kernel, pw_1_fused.bias.data),
        ]

        # Identity branch (pointwise: out_ch x in_ch x 3 x 3)
        if self.use_identity:
            id_kernel, id_bias = identity_to_conv(self.in_channels, self.out_channels)
            id_kernel = id_kernel.to(pw_3_fused.weight.device)
            id_bias = id_bias.to(pw_3_fused.weight.device)
            # Fuse identity BN
            # Identity kernel: for each output channel, 3x3 with center=1
            # Apply BN fusion manually
            gamma = self.id_bn.weight.data
            beta = self.id_bn.bias.data
            mu = self.id_bn.running_mean.data
            var = self.id_bn.running_var.data
            eps = self.id_bn.eps
            std = torch.sqrt(var + eps)

            # Fused identity kernel
            fused_id_kernel = id_kernel * (gamma / std).view(-1, 1, 1, 1)
            fused_id_bias = -gamma * mu / std + beta

            branches.append((fused_id_kernel, fused_id_bias))

        merged_kernel, merged_bias = merge_branches(branches)

        # The merged point-wise kernel is 3x3 (since we padded 1x1s to 3x3)
        # For deployment efficiency, use 3x3
        pw_conv = nn.Conv2d(
            self.in_channels, self.out_channels,
            kernel_size=3, stride=1, padding=1,
            groups=1, bias=True,
        )
        pw_conv.weight.data = merged_kernel
        pw_conv.bias.data = merged_bias

        return dw_conv, pw_conv

    def _build_deploy_conv(self):
        """Build and apply deployment weights to self."""
        dw, pw = self.get_deploy_weights()
        self.deploy_dw_conv = dw
        self.deploy_pw_conv = pw
        self.deploy_mode = True
        return self


class RepDWCDownsample(nn.Module):
    """
    RepDWC block with stride-2 in depth-wise stage for spatial downsampling.
    Includes a parallel stride-2 1x1 conv for the residual path.
    """

    def __init__(self, in_channels, out_channels, deploy_mode=False):
        super().__init__()
        self.dwc_block = RepDWCBlock(
            in_channels, out_channels, stride=2, deploy_mode=deploy_mode
        )
        # Skip connection for downsampling
        if deploy_mode:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=True)
        else:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_channels)
        self.deploy_mode = deploy_mode

    def forward(self, x):
        if self.deploy_mode:
            main = self.dwc_block(x)
            skip = self.skip(x)
        else:
            main = self.dwc_block(x)
            skip = self.skip_bn(self.skip_conv(x))
        return F.relu(main + skip)

    def get_deploy_weights(self):
        """Fuse skip connection and convert inner block for deployment."""
        if self.deploy_mode:
            return None
        self.dwc_block._build_deploy_conv()
        self.skip = fuse_conv_bn(self.skip_conv, self.skip_bn)
        self.deploy_mode = True
