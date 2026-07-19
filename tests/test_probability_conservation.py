"""
Test 3: Probability conservation (Paper Section 18.3).

Validates:
    - Geometry normalization: sum_j pi_ij ≈ 1 for each point i
    - Reliability weighting: sum_j w_ij ≈ q_i for each point i
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch


def test_geometry_normalization():
    """After soft geometry normalization, sum of pi_ij should be ~1."""
    N = 50  # points
    K = 25  # pillars (5x5 neighborhood)
    eps = 1e-8

    # Random geometric probabilities
    g = torch.rand(N, K)  # Unnormalized

    # Normalize
    g_sum = g.sum(dim=-1, keepdim=True) + eps
    pi = g / g_sum

    row_sums = pi.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(N), atol=1e-5), \
        f"Row sums deviate from 1: max error = {(row_sums - 1).abs().max()}"


def test_reliability_weighting():
    """After multiplying by reliability: sum_j w_ij ≈ q_i."""
    N = 50
    K = 25
    eps = 1e-8

    g = torch.rand(N, K)
    q = torch.rand(N) * 0.8 + 0.2  # [0.2, 1.0]

    pi = g / (g.sum(dim=-1, keepdim=True) + eps)
    w = q.unsqueeze(-1) * pi

    w_sums = w.sum(dim=-1)
    expected = q

    diff = (w_sums - expected).abs()
    max_diff = diff.max().item()
    assert max_diff < 1e-5, f"Reliability weighting error: max diff = {max_diff}"


def test_reliability_not_canceled():
    """
    Paper design rule: normalize geometry FIRST, then multiply by reliability.
    If q_i is put in numerator AND denominator, it cancels out.
    """
    N = 50
    K = 25
    eps = 1e-8

    g = torch.rand(N, K)
    q = torch.tensor([0.3, 0.7]).repeat(25)  # Half low, half high reliability

    # WRONG way: q_i * g_ij / sum_j' q_i * g_ij'  (q cancels out)
    wrong_num = q.unsqueeze(-1) * g
    wrong_den = wrong_num.sum(dim=-1, keepdim=True) + eps
    wrong_w = wrong_num / wrong_den

    # CORRECT way: first normalize g, then multiply by q
    pi = g / (g.sum(dim=-1, keepdim=True) + eps)
    correct_w = q.unsqueeze(-1) * pi

    # Wrong way should be independent of q (all rows cancel)
    wrong_sums = wrong_w.sum(dim=-1)
    assert torch.allclose(wrong_sums, torch.ones(N), atol=1e-5), \
        "Wrong normalization: q_i should cancel out"

    # Correct way should preserve q_i information
    correct_sums = correct_w.sum(dim=-1)
    assert not torch.allclose(correct_sums, torch.ones(N), atol=0.01), \
        "Correct normalization should NOT equal 1 for varying q_i"
    assert torch.allclose(correct_sums, q, atol=1e-5), \
        "Correct sums should equal q_i"


def test_batch_conservation():
    """Large batch probability conservation sanity check."""
    N = 1000
    K = 25

    g = torch.rand(N, K).abs() + 0.1  # All positive
    pi = g / g.sum(dim=-1, keepdim=True)

    # All pi_ij should be in [0, 1]
    assert (pi >= 0).all(), "Probabilities should be non-negative"
    assert (pi <= 1).all(), "Probabilities should be <= 1"

    # Rows should sum to 1
    assert torch.allclose(pi.sum(dim=-1), torch.ones(N), atol=1e-5)


if __name__ == '__main__':
    test_geometry_normalization()
    test_reliability_weighting()
    test_reliability_not_canceled()
    test_batch_conservation()
    print("All probability conservation tests PASSED")
