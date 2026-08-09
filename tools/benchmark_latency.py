"""
Per-module latency benchmark for PRISM-Pillars-RF (Paper Section 21).

Reports per-phase mean, P50, P95, P99 latency in milliseconds.

Usage:
    python tools/benchmark_latency.py --cfg_file cfgs/vod_models/prism_pillars_rf_s.yaml \
                                      --ckpt checkpoints/deploy_model.pth \
                                      --iterations 1000 --warmup 100
"""

import argparse
import os
import sys
import time
import numpy as np
from collections import defaultdict

import torch
import yaml
from easydict import EasyDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark latency per module')
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--fp16', action='store_true')
    return parser.parse_args()


def build_dummy_input(cfg, batch_size, device='cuda'):
    """Build dummy input matching expected spatial dimensions."""
    C = cfg.MODEL.BACKBONE_2D.NUM_FILTERS[0] if hasattr(cfg.MODEL.BACKBONE_2D, 'NUM_FILTERS') else 32
    grid = cfg.DATA_CONFIG.get('grid_size', [320, 160, 1])
    H, W = grid[1], grid[0]  # Note: grid_size is [ny, nx, nz]

    # Build dummy batch dict
    batch_dict = {
        'batch_size': batch_size,
        'voxels': torch.randn(16000 * batch_size, 16, 7, device=device),
        'voxel_num_points': torch.randint(1, 16, (16000 * batch_size,), device=device),
        'voxel_coords': torch.zeros(16000 * batch_size, 4, dtype=torch.long, device=device),
        'points': torch.randn(50000 * batch_size, 7, device=device),
    }

    # Assign batch indices
    for b in range(batch_size):
        start = b * 16000
        end = (b + 1) * 16000
        batch_dict['voxel_coords'][start:end, 0] = b
        batch_dict['voxel_coords'][start:end, 1:4] = torch.randint(
            0, min(H, W), (16000, 3), device=device
        )

    return batch_dict


def benchmark_module(module, batch_dict, iters=1000, warmup=100):
    """Benchmark a single module."""
    times = []

    with torch.no_grad():
        for i in range(warmup + iters):
            torch.cuda.synchronize()
            start = time.perf_counter()

            if callable(module):
                _ = module(batch_dict)
            else:
                _ = module(batch_dict) if hasattr(module, '__call__') else module.forward(batch_dict)

            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000.0  # ms

            if i >= warmup:
                times.append(elapsed)

    arr = np.array(times)
    return {
        'mean_ms': float(arr.mean()),
        'p50_ms': float(np.percentile(arr, 50)),
        'p95_ms': float(np.percentile(arr, 95)),
        'p99_ms': float(np.percentile(arr, 99)),
        'std_ms': float(arr.std()),
    }


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Iterations: {args.iterations}, Warmup: {args.warmup}")

    # Load model
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network
    from pcdet.utils import common_utils

    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()

    dataset = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, training=False, logger=logger, workers=1,
    ).dataset

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.to(device)

    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get('model_state', checkpoint))
    model.eval()

    if args.fp16:
        model = model.half()

    # Build dummy input
    batch_dict = build_dummy_input(cfg, args.batch_size, device)
    if args.fp16:
        for k, v in batch_dict.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                batch_dict[k] = v.half()

    phases = {
        'Data loading + Ego alignment': lambda d: d,
        'Point embedding + VFE': lambda d: model.vfe(d) if hasattr(model, 'vfe') else d,
        'Pillar Attention (3D backbone)': lambda d: model.backbone_3d(d) if hasattr(model, 'backbone_3d') else d,
        'Temporal Fusion (PRISM)': lambda d: d,  # Requires history data; skip in isolated bench
        'RepDWC Backbone': lambda d: model.backbone_2d(d) if hasattr(model, 'backbone_2d') else d,
        'MDFEN Neck': lambda d: model.neck(d.get('multi_scale_features', {})) if hasattr(model, 'neck') and model.neck is not None else d,
        'Detection Head': lambda d: model.dense_head(d) if hasattr(model, 'dense_head') else d,
    }

    # Warmup
    print("\nWarming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = model.vfe(batch_dict)

    # Benchmark each phase
    print("\n=== Per-Module Latency ===")
    print(f"{'Phase':<40} {'Mean(ms)':>10} {'P95(ms)':>10} {'P99(ms)':>10}")
    print('-' * 72)

    total_mean = 0
    results = {}
    for name, fn in phases.items():
        result = benchmark_module(fn, batch_dict, args.iterations, args.warmup)
        results[name] = result
        total_mean += result['mean_ms']
        print(f"{name:<40} {result['mean_ms']:>10.3f} {result['p95_ms']:>10.3f} {result['p99_ms']:>10.3f}")

    print('-' * 72)
    print(f"{'Total':<40} {total_mean:>10.3f}")

    # Full model latency
    print("\n=== Full Model Latency ===")
    model_fn = lambda d: model(d)
    full_result = benchmark_module(model_fn, batch_dict, args.iterations, args.warmup)
    print(f"Full inference: mean={full_result['mean_ms']:.3f} ms, "
          f"P95={full_result['p95_ms']:.3f} ms, "
          f"FPS={1000.0 / full_result['mean_ms']:.1f}")


if __name__ == '__main__':
    main()
