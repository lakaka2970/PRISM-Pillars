"""
Reparameterization utilities for RepDWC backbone (Paper Section 9.2).

Converts training-time multi-branch structures into deployment-time
single-path convolutions by fusing Conv+BN and merging parallel branches.

Equivalence test requirement (Paper Section 9.3):
    max|f_train(x) - f_deploy(x)| < 1e-4  (FP32)
    max|f_train(x) - f_deploy(x)| < 1e-3  (FP16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def fuse_conv_bn(conv, bn):
    """
    Fuse Conv2d and BatchNorm2d into a single Conv2d.

    W' = gamma / sqrt(sigma^2 + eps) * W
    b' = beta - gamma * mu / sqrt(sigma^2 + eps) + gamma / sqrt(sigma^2 + eps) * b

    Note: assumes conv has bias (even if zero) for mathematical convenience.
           If the original conv has bias=False, treat b=0.

    Args:
        conv: nn.Conv2d
        bn: nn.BatchNorm2d

    Returns:
        fused_conv: nn.Conv2d with same config but fused weights.
    """
    assert conv.groups == 1 or conv.groups == conv.in_channels, \
        "Depthwise conv fusion requires special handling"

    # Get BN parameters
    bn_weight = bn.weight.data.clone()        # gamma
    bn_bias = bn.bias.data.clone()            # beta
    bn_running_mean = bn.running_mean.data.clone()  # mu
    bn_running_var = bn.running_var.data.clone()    # sigma^2
    bn_eps = bn.eps

    # BN standard deviation
    bn_std = torch.sqrt(bn_running_var + bn_eps)

    # Get conv parameters
    conv_weight = conv.weight.data.clone()
    has_bias = conv.bias is not None

    # Fused weight: reshape BN params for broadcasting
    # conv_weight: (out_ch, in_ch/groups, kH, kW)
    # bn_weight: (out_ch,)
    fused_weight = conv_weight * (bn_weight / bn_std).view(-1, 1, 1, 1)

    # Fused bias
    if has_bias:
        conv_bias = conv.bias.data.clone()
        fused_bias = bn_bias - bn_weight * bn_running_mean / bn_std + bn_weight * conv_bias / bn_std
    else:
        fused_bias = bn_bias - bn_weight * bn_running_mean / bn_std

    # Create fused conv
    fused_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    )
    fused_conv.weight.data = fused_weight
    fused_conv.bias.data = fused_bias

    return fused_conv


def pad_1x1_to_3x3(kernel):
    """
    Pad a 1x1 convolution kernel to 3x3 by zero-padding.

    Args:
        kernel: (out_ch, in_ch/groups, 1, 1) float tensor.

    Returns:
        padded: (out_ch, in_ch/groups, 3, 3) float tensor.
    """
    return F.pad(kernel, [1, 1, 1, 1], value=0.0)


def identity_to_conv(in_channels, out_channels=None):
    """
    Create a 3x3 convolution kernel equivalent to identity.

    When out_channels is provided (pointwise mode):
        kernel shape: (out_channels, in_channels, 3, 3)
        diagonal entries at center pixel set to 1.

    When out_channels is None (depthwise mode):
        kernel shape: (in_channels, 1, 3, 3)
        all center pixels set to 1 (groups=in_channels expected).

    Args:
        in_channels: int, number of input channels.
        out_channels: int or None. None = depthwise mode.

    Returns:
        kernel: float tensor.
        bias: float tensor (zeros).
    """
    if out_channels is not None:
        # Pointwise identity: out_ch x in_ch x 3 x 3, with diag entries at center
        kernel = torch.zeros(out_channels, in_channels, 3, 3)
        min_ch = min(in_channels, out_channels)
        for c in range(min_ch):
            kernel[c, c, 1, 1] = 1.0
        bias = torch.zeros(out_channels)
    else:
        # Depthwise identity: in_ch x 1 x 3 x 3
        kernel = torch.zeros(in_channels, 1, 3, 3)
        kernel[:, 0, 1, 1] = 1.0
        bias = torch.zeros(in_channels)
    return kernel, bias


def merge_branches(branches):
    """
    Merge multiple parallel convolutional branches by summing their
    kernels and biases.

    Each branch is a tuple (kernel, bias) where kernels must all be
    padded to the same spatial size.

    Args:
        branches: list of (kernel, bias) tuples.

    Returns:
        merged_kernel: float tensor.
        merged_bias: float tensor.
    """
    merged_kernel = branches[0][0].clone()
    merged_bias = branches[0][1].clone()

    for kernel, bias in branches[1:]:
        merged_kernel = merged_kernel + kernel
        merged_bias = merged_bias + bias

    return merged_kernel, merged_bias


def validate_deploy_equivalence(model_train, model_deploy, x, tolerance=1e-4):
    """
    Validate that training and deployment models produce equivalent outputs.

    Args:
        model_train: nn.Module in training mode (multi-branch).
        model_deploy: nn.Module in deployment mode (fused).
        x: torch.Tensor, input tensor.
        tolerance: float, maximum allowed absolute difference.

    Returns:
        is_valid: bool, True if max difference < tolerance.
        max_diff: float, maximum absolute element-wise difference.
    """
    model_train.eval()
    model_deploy.eval()

    with torch.no_grad():
        y_train = model_train(x)
        y_deploy = model_deploy(x)

    max_diff = (y_train - y_deploy).abs().max().item()
    is_valid = max_diff < tolerance

    return is_valid, max_diff


def deploy_model(model, verbose=True):
    """
    Recursively convert a model's RepDWC blocks from training to deployment mode.

    Args:
        model: nn.Module.
        verbose: bool, print conversion progress.

    Returns:
        model: nn.Module with RepDWC blocks converted.
    """
    converted_count = 0

    for name, module in model.named_modules():
        if hasattr(module, '_build_deploy_conv'):
            if verbose:
                print(f"  Converting: {name}")
            module.deploy_mode = True
            # Build deployed weights
            deploy_conv = module._build_deploy_conv()
            # Replace the multi-branch structure
            module.deploy_conv = deploy_conv
            converted_count += 1

    if verbose:
        print(f"Converted {converted_count} RepDWC blocks to deploy mode")

    return model
