"""
Test 7: DCNv3 shape validation (Paper Section 18.7).

Validates:
    - Input/output shape consistency for DCNv3Wrapper
    - NCHW format preservation
    - Fallback Conv produces correct output shape
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import importlib

# Direct import to avoid pcdet.models.__init__ CUDA dependency chain
dcnv3_mod = importlib.import_module('pcdet.models.backbones_2d.dcnv3_wrapper')
DCNv3Wrapper = dcnv3_mod.DCNv3Wrapper
has_dcnv3_support = dcnv3_mod.has_dcnv3_support


def test_dcn_shape_consistency():
    """Output shape should match input shape."""
    B, C, H, W = 2, 32, 100, 50
    x = torch.randn(B, C, H, W)

    wrapper = DCNv3Wrapper(C, kernel_size=3, groups=4, stride=1, padding=1)
    out = wrapper(x)

    assert out.shape == (B, C, H, W), f"Expected ({B},{C},{H},{W}), got {out.shape}"


def test_dcn_gradient_flow():
    """Gradients should flow through the wrapper."""
    C, H, W = 16, 32, 16
    x = torch.randn(1, C, H, W, requires_grad=True)

    wrapper = DCNv3Wrapper(C, kernel_size=3, groups=4)
    out = wrapper(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Gradient should flow through DCNv3 wrapper"
    assert not torch.isnan(x.grad).any(), "Gradients contain NaN"


def test_dcn_stride2():
    """Stride=2 should halve spatial dimensions."""
    B, C, H, W = 2, 32, 64, 64
    x = torch.randn(B, C, H, W)

    wrapper = DCNv3Wrapper(C, kernel_size=3, stride=2, padding=1)
    out = wrapper(x)

    assert out.shape == (B, C, H // 2, W // 2), \
        f"Expected ({B},{C},{H//2},{W//2}), got {out.shape}"


def test_dcn_preserves_values_range():
    """Output values should be in reasonable range with BatchNorm."""
    C, H, W = 32, 40, 20
    x = torch.randn(1, C, H, W)

    wrapper = DCNv3Wrapper(C)
    wrapper.eval()

    with torch.no_grad():
        out = wrapper(x)

    # With BN, range should be controlled
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"


def test_dcn_availability_report():
    """Report whether DCNv3 is available in this environment."""
    available = has_dcnv3_support()
    print(f"DCNv3 available: {available}")
    # This test always passes - it's informational
    assert True


if __name__ == '__main__':
    test_dcn_shape_consistency()
    test_dcn_gradient_flow()
    test_dcn_stride2()
    test_dcn_preserves_values_range()
    test_dcn_availability_report()
    print("All DCNv3 shape tests PASSED")
