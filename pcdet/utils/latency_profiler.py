"""
Latency profiling utilities for PRISM-Pillars-RF.

Provides per-module timing with context manager support and reporting
for average/P95 latency across inference iterations.
"""

import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Phase labels matching paper Section 21 (Efficiency Evaluation)
# ---------------------------------------------------------------------------
_PHASE_ORDER = [
    'Data loading',
    'Ego alignment',
    'Point embedding',
    'Reliability',
    'Uncertainty Tube',
    'Probabilistic Routing',
    'Temporal candidate retrieval',
    'Temporal attention',
    'RepDWC Backbone',
    'MDFEN',
    'Detection Head',
    'Post-processing',
]

_PHASE_INDEX = {name: i for i, name in enumerate(_PHASE_ORDER)}


class LatencyProfiler:
    """
    Per-module latency profiler with warm-up skipping and P95 reporting.

    Usage:
        profiler = LatencyProfiler(warmup_iters=100)

        with profiler.phase('Point embedding'):
            features = embedder(points)

        profiler.report()
    """

    def __init__(self, warmup_iters=100, max_iters=2000):
        """
        Args:
            warmup_iters: Number of initial iterations to skip.
            max_iters: Maximum iterations to record (after warmup).
        """
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self._iter_count = 0
        self._records = defaultdict(list)
        self._start_times = {}

    def reset(self):
        self._iter_count = 0
        self._records.clear()
        self._start_times.clear()

    @contextmanager
    def phase(self, name):
        """
        Context manager for profiling a named phase.

        Usage:
            with profiler.phase('Backbone'):
                x = backbone(x)
        """
        if name not in _PHASE_INDEX:
            _PHASE_INDEX[name] = len(_PHASE_ORDER)
            _PHASE_ORDER.append(name)

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0  # ms
            if self._iter_count >= self.warmup_iters:
                self._records[name].append(elapsed)

    def step(self):
        """Mark end of one full inference iteration."""
        self._iter_count += 1

    def get_stats(self):
        """
        Returns:
            dict: phase_name -> {'mean_ms': float, 'p95_ms': float, 'count': int}
        """
        stats = {}
        for name in _PHASE_ORDER:
            if name not in self._records or len(self._records[name]) == 0:
                continue
            arr = np.array(self._records[name])
            stats[name] = {
                'mean_ms': float(np.mean(arr)),
                'p50_ms': float(np.percentile(arr, 50)),
                'p95_ms': float(np.percentile(arr, 95)),
                'p99_ms': float(np.percentile(arr, 99)),
                'std_ms': float(np.std(arr)),
                'count': len(arr),
            }
        return stats

    def report(self, logger=None):
        """
        Print a formatted latency report.

        Args:
            logger: Optional logger object with .info() method.
        """
        stats = self.get_stats()

        total_mean = sum(v['mean_ms'] for v in stats.values())

        lines = ['=' * 70]
        lines.append(f'{"Phase":<28} {"Mean(ms)":>10} {"P95(ms)":>10} {"%":>8}')
        lines.append('-' * 70)

        for name in _PHASE_ORDER:
            if name not in stats:
                continue
            s = stats[name]
            pct = s['mean_ms'] / total_mean * 100.0 if total_mean > 0 else 0.0
            lines.append(
                f'{name:<28} {s["mean_ms"]:>10.3f} {s["p95_ms"]:>10.3f} {pct:>7.1f}%'
            )

        lines.append('-' * 70)
        lines.append(f'{"Total":<28} {total_mean:>10.3f}')
        lines.append('=' * 70)

        report_str = '\n'.join(lines)
        if logger is not None:
            logger.info('\n' + report_str)
        else:
            print(report_str)

        return report_str


def profile_inference(model, dataloader, num_iters=500, warmup=100, device='cuda'):
    """
    Run inference profiling over multiple batches.

    Args:
        model: nn.Module to profile.
        dataloader: DataLoader yielding batch_dict.
        num_iters: Number of inference iterations to record.
        warmup: Number of warmup iterations.
        device: Device string.

    Returns:
        LatencyProfiler with collected stats.
    """
    model.eval()
    profiler = LatencyProfiler(warmup_iters=warmup)

    iterator = iter(dataloader)
    total_iters = warmup + num_iters

    with torch.no_grad():
        for i in range(total_iters):
            try:
                batch_dict = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch_dict = next(iterator)

            # Move data to GPU
            with profiler.phase('Data loading'):
                for key, val in batch_dict.items():
                    if isinstance(val, np.ndarray):
                        batch_dict[key] = torch.from_numpy(val).float().to(device)

            # Run inference with per-module profiling
            with profiler.phase('Total'):
                _ = model(batch_dict)

            profiler.step()

    return profiler


# ---------------------------------------------------------------------------
# Global profiler singleton (optional convenience)
# ---------------------------------------------------------------------------
_global_profiler = None


def get_global_profiler():
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = LatencyProfiler()
    return _global_profiler
