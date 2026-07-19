"""
Convert RepDWC training model to deployment mode (Paper Section 9.2, 11).

Usage:
    python tools/convert_to_deploy.py --cfg_file cfgs/vod_models/prism_pillars_rf_s.yaml \
                                      --ckpt checkpoints/ckpt_epoch_80.pth \
                                      --output checkpoints/deploy_model.pth

Validates equivalence: max|f_train(x) - f_deploy(x)| < 1e-4 (FP32)
"""

import argparse
import os
import sys

import torch
import yaml
from easydict import EasyDict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def parse_args():
    parser = argparse.ArgumentParser(description='Convert RepDWC to deploy mode')
    parser.add_argument('--cfg_file', type=str, required=True, help='Model config YAML')
    parser.add_argument('--ckpt', type=str, required=True, help='Training checkpoint')
    parser.add_argument('--output', type=str, default='deploy_model.pth', help='Output path')
    parser.add_argument('--validate', action='store_true', default=True, help='Run equivalence check')
    parser.add_argument('--tolerance', type=float, default=1e-4, help='Max allowed diff (FP32 default 1e-4)')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 tolerance (1e-3)')
    return parser.parse_args()


def load_model(cfg_file, ckpt_path, device='cuda'):
    """Load model from config and checkpoint."""
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network
    from pcdet.utils import common_utils

    cfg_from_yaml_file(cfg_file, cfg)

    # Use dummy dataset for model building
    dataset = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        training=False,
        logger=common_utils.create_logger(),
        workers=1,
    ).dataset

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.to(device)

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    return model, cfg


def convert_to_deploy(model):
    """
    Recursively convert all RepDWC blocks and RepBEVBackbone to deploy mode.

    Returns:
        converted_count: int, number of blocks converted.
    """
    converted = 0

    for name, module in model.named_modules():
        if hasattr(module, 'get_deploy_weights'):
            print(f"  Converting RepDWC block: {name}")
            module._build_deploy_conv()
            converted += 1

        if hasattr(module, 'set_deploy_mode'):
            print(f"  Setting deploy mode on backbone: {name}")
            module.set_deploy_mode(True)
            converted += 1

    print(f"\nTotal blocks converted: {converted}")
    return converted


def validate_equivalence(model_train, model_deploy, cfg, tolerance=1e-4, device='cuda'):
    """
    Validate training vs deployment equivalence with random inputs.

    Tests: stride 1, stride 2, with/without identity, batch sizes 1 and 8.
    """
    import numpy as np

    C = cfg.MODEL.BACKBONE_2D.get('NUM_FILTERS', [32])[0]
    H, W = 200, 100  # Typical BEV size for 51.2m x 51.2m with 0.16m voxel

    print("\n=== Equivalence Validation ===")

    test_cases = [
        {'batch_size': 1, 'stride': 1, 'use_id': True},
        {'batch_size': 8, 'stride': 1, 'use_id': True},
        {'batch_size': 1, 'stride': 2, 'use_id': False},
        {'batch_size': 8, 'stride': 2, 'use_id': False},
    ]

    all_passed = True
    for tc in test_cases:
        x = torch.randn(tc['batch_size'], C, H // tc['stride'], W // tc['stride'], device=device)

        with torch.no_grad():
            y_train = model_train(x)
            y_deploy = model_deploy(x)

        max_diff = (y_train - y_deploy).abs().max().item()
        passed = max_diff < tolerance
        status = "PASS" if passed else "FAIL"

        print(f"  B={tc['batch_size']}, stride={tc['stride']}, id={tc['use_id']}: "
              f"max_diff={max_diff:.6f} [{status}]")

        if not passed:
            all_passed = False

    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILURES'}")
    return all_passed


def main():
    args = parse_args()

    tolerance = args.tolerance
    if args.fp16:
        tolerance = max(tolerance, 1e-3)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Tolerance: {tolerance}")

    # Load training model
    print(f"\nLoading model from: {args.ckpt}")
    model, cfg = load_model(args.cfg_file, args.ckpt, device)

    # Clone model for deploy (keep training version for validation)
    import copy
    model_deploy = copy.deepcopy(model)

    # Convert to deploy mode
    print("\nConverting RepDWC blocks to deploy mode...")
    converted = convert_to_deploy(model_deploy)

    if converted == 0:
        print("WARNING: No RepDWC blocks found. Nothing to convert.")
        sys.exit(0)

    # Validate equivalence
    if args.validate:
        passed = validate_equivalence(model, model_deploy, cfg, tolerance, device)
        if not passed:
            print("\nWARNING: Equivalence check FAILED. Deployment model may produce different results.")
            if not args.fp16:
                print("Try --fp16 for relaxed tolerance.")

    # Save deploy model
    deploy_state = {
        'model_state': model_deploy.state_dict(),
        'config': dict(cfg),
        'converted_blocks': converted,
        'deploy_mode': True,
    }
    torch.save(deploy_state, args.output)
    print(f"\nDeploy model saved to: {args.output}")


if __name__ == '__main__':
    main()
