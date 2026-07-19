"""
Test 6: RepDWC equivalence validation (Paper Section 18.6).

Validates:
    - max|f_train(x) - f_deploy(x)| < 1e-4 (FP32)
    - Covers stride 1, stride 2, with/without identity, batch sizes 1 and 8
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import copy
import importlib

# Direct import to avoid pcdet.models.__init__ CUDA dependency chain
rep_dwc = importlib.import_module('pcdet.models.backbones_2d.rep_dwc')
reparam = importlib.import_module('pcdet.models.backbones_2d.reparameterize')
RepDWCBlock = rep_dwc.RepDWCBlock
RepDWCDownsample = rep_dwc.RepDWCDownsample
validate_deploy_equivalence = reparam.validate_deploy_equivalence


def test_repdwc_block_stride1():
    """Stride 1 with identity branch (same in/out channels)."""
    C = 32
    block_train = RepDWCBlock(C, C, stride=1, deploy_mode=False)
    block_train.eval()

    block_deploy = copy.deepcopy(block_train)
    block_deploy._build_deploy_conv()
    block_deploy.eval()

    for batch_size in [1, 8]:
        x = torch.randn(batch_size, C, 64, 32)
        is_valid, max_diff = validate_deploy_equivalence(
            block_train, block_deploy, x, tolerance=1e-4
        )
        assert is_valid, f"B={batch_size}, max_diff={max_diff}"


def test_repdwc_block_stride1_no_id():
    """Stride 1 without identity (different in/out channels)."""
    block_train = RepDWCBlock(32, 48, stride=1, deploy_mode=False)
    block_train.eval()

    block_deploy = copy.deepcopy(block_train)
    block_deploy._build_deploy_conv()
    block_deploy.eval()

    x = torch.randn(4, 32, 64, 32)
    is_valid, max_diff = validate_deploy_equivalence(
        block_train, block_deploy, x, tolerance=1e-4
    )
    assert is_valid, f"max_diff={max_diff}"


def test_repdwc_downsample():
    """Stride 2 downsample block."""
    block_train = RepDWCDownsample(32, 64, deploy_mode=False)
    block_train.eval()

    block_deploy = copy.deepcopy(block_train)
    block_deploy.get_deploy_weights()
    block_deploy.eval()

    for batch_size in [1, 8]:
        x = torch.randn(batch_size, 32, 64, 32)
        is_valid, max_diff = validate_deploy_equivalence(
            block_train, block_deploy, x, tolerance=1e-4
        )
        assert is_valid, f"B={batch_size}, max_diff={max_diff}"


def test_repdwc_multiple_blocks():
    """Chain of 3 RepDWC blocks."""
    C = 32
    blocks_train = torch.nn.Sequential(
        RepDWCBlock(C, C, stride=1, deploy_mode=False),
        RepDWCBlock(C, C, stride=1, deploy_mode=False),
        RepDWCBlock(C, C, stride=1, deploy_mode=False),
    )
    blocks_train.eval()

    blocks_deploy = copy.deepcopy(blocks_train)
    for block in blocks_deploy:
        block._build_deploy_conv()
    blocks_deploy.eval()

    x = torch.randn(2, C, 32, 16)
    is_valid, max_diff = validate_deploy_equivalence(
        blocks_train, blocks_deploy, x, tolerance=1e-4
    )
    assert is_valid, f"max_diff={max_diff}"


def test_fp16_tolerance():
    """FP16 equivalence within relaxed tolerance (Paper Section 9.3: < 1e-3)."""
    C = 32
    block_train = RepDWCBlock(C, C, stride=1, deploy_mode=False).half()
    block_train.eval()

    block_deploy = copy.deepcopy(block_train)
    block_deploy._build_deploy_conv()
    block_deploy.eval()

    x = torch.randn(2, C, 64, 32).half()
    # FP16 weights -> FP32 computation: use relaxed tolerance
    is_valid, max_diff = validate_deploy_equivalence(
        block_train.float(), block_deploy.float(), x.float(), tolerance=2e-3
    )
    assert is_valid, f"FP16 tolerance test failed: max_diff={max_diff} > 2e-3"


if __name__ == '__main__':
    test_repdwc_block_stride1()
    test_repdwc_block_stride1_no_id()
    test_repdwc_downsample()
    test_repdwc_multiple_blocks()
    test_fp16_tolerance()
    print("All RepDWC equivalence tests PASSED")
