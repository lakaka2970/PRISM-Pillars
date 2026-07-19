"""
Test 4: Reliability weight ordering (Paper Section 18.4).

Validates:
    - Given same geometric probability, q1 > q2 => sum_j w_1j > sum_j w_2j
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch


def test_reliability_ordering():
    """Higher reliability should produce higher total weight."""
    K = 25
    eps = 1e-8

    # Two points with identical geometry but different reliability
    g = torch.rand(1, K)  # Same geometric profile for both
    q1 = 0.8
    q2 = 0.3

    pi = g / (g.sum(dim=-1, keepdim=True) + eps)
    w1 = q1 * pi
    w2 = q2 * pi

    sum1 = w1.sum()
    sum2 = w2.sum()

    assert sum1 > sum2, f"q1={q1} weight sum {sum1} should > q2={q2} weight sum {sum2}"
    assert abs(sum1 - q1) < 1e-5, f"Sum should equal q1: {sum1} vs {q1}"
    assert abs(sum2 - q2) < 1e-5, f"Sum should equal q2: {sum2} vs {q2}"


def test_reliability_zero_suppression():
    """Zero reliability should completely suppress a point."""
    K = 25
    eps = 1e-8

    g = torch.rand(1, K)
    q = 0.0

    pi = g / (g.sum(dim=-1, keepdim=True) + eps)
    w = q * pi

    assert (w == 0).all(), "Zero reliability should zero all weights"


def test_reliability_ordering_multiple():
    """Multiple points with varying q: ordering should be preserved."""
    N = 10
    K = 25
    eps = 1e-8

    g = torch.rand(N, K)
    q = torch.arange(N).float() / N + 0.1  # Monotonically increasing

    pi = g / (g.sum(dim=-1, keepdim=True) + eps)
    w = q.unsqueeze(-1) * pi
    sums = w.sum(dim=-1)

    # Sums should also be monotonically increasing (with high probability)
    assert (sums[1:] - sums[:-1] > -1e-5).all(), \
        f"Reliability ordering violated: sums = {sums}"


if __name__ == '__main__':
    test_reliability_ordering()
    test_reliability_zero_suppression()
    test_reliability_ordering_multiple()
    print("All reliability weight tests PASSED")
