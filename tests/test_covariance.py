"""
Test 2: Covariance matrix validation (Paper Section 18.2).

Validates:
    - Positive definiteness: lambda_min(Sigma) > 0
    - Physical constraint: sigma_t >= sigma_r
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from pcdet.utils.covariance_2d import (
    build_anisotropic_covariance,
    validate_covariance,
    check_tangential_geq_radial,
)
from pcdet.utils.radar_geometry import (
    compute_line_of_sight_unit_vector,
    compute_tangential_direction,
)


def test_positive_definite():
    """All covariance matrices must be positive definite."""
    N = 100
    s_r = torch.rand(N) * 0.3 + 0.05  # [0.05, 0.35]
    s_t = s_r + torch.rand(N) * 0.5     # [s_r, s_r+0.5] -> s_t >= s_r

    # Random directions
    theta = torch.rand(N) * 2 * 3.14159
    u = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    Sigma = build_anisotropic_covariance(s_r, s_t, u, sigma_0=0.03)
    is_valid, min_eig, lambda_min, lambda_max = validate_covariance(Sigma)

    assert is_valid, f"Non-positive definite covariance found. min_eig = {min_eig}"
    assert (lambda_min > 0).all(), f"Negative eigenvalues found"
    assert (lambda_max > lambda_min).all(), f"lambda_max <= lambda_min"


def test_isotropic_limit():
    """When s_r = s_t, covariance should be isotropic (if sigma_0 dominates)."""
    N = 10
    s_r = torch.full((N,), 0.1)
    s_t = torch.full((N,), 0.1)

    # All pointing in x-direction
    u = torch.zeros(N, 2)
    u[:, 0] = 1.0

    Sigma = build_anisotropic_covariance(s_r, s_t, u, sigma_0=1.0)

    # With sigma_0=1.0 dominating, matrix should be nearly isotropic
    # Diagonal elements should be close
    assert torch.allclose(Sigma[:, 0, 0], Sigma[:, 1, 1], atol=0.1)


def test_anisotropy():
    """With s_t >> s_r, covariance should be highly anisotropic."""
    N = 10
    s_r = torch.full((N,), 0.05)
    s_t = torch.full((N,), 2.0)

    # x-direction (radial)
    u = torch.zeros(N, 2)
    u[:, 0] = 1.0

    Sigma = build_anisotropic_covariance(s_r, s_t, u, sigma_0=0.03)
    is_valid, min_eig, lambda_min, lambda_max = validate_covariance(Sigma)

    assert is_valid
    # Major axis eigenvalue should be >> minor axis eigenvalue
    ratio = lambda_max / lambda_min.clamp(min=1e-8)
    assert (ratio > 10).any(), f"Expected high anisotropy, ratio range: [{ratio.min():.2f}, {ratio.max():.2f}]"


def test_tangential_ge_radial():
    """Physical constraint: tangential uncertainty >= radial uncertainty."""
    s_r_good = torch.tensor([0.1, 0.2, 0.15])
    s_t_good = torch.tensor([0.3, 0.5, 0.2])

    is_valid, violations = check_tangential_geq_radial(s_t_good, s_r_good)
    assert is_valid, "Valid case should pass"

    s_r_bad = torch.tensor([0.3, 0.2, 0.5])
    s_t_bad = torch.tensor([0.1, 0.5, 0.4])  # First one violates

    is_valid, violations = check_tangential_geq_radial(s_t_bad, s_r_bad)
    assert not is_valid, "Invalid case should fail"
    assert violations[0], "First element should have violation"


def test_build_with_direction():
    """Covariance should align with specified direction."""
    s_r = torch.tensor([0.5])
    s_t = torch.tensor([2.0])
    u = torch.tensor([[1.0, 0.0]])  # x-direction

    Sigma = build_anisotropic_covariance(s_r, s_t, u, sigma_0=0.0)

    # Sigma = s_r^2 * u*u^T + s_t^2 * n*n^T
    # = 0.25 * [[1,0],[0,0]] + 4.0 * [[0,0],[0,1]]
    # = [[0.25, 0], [0, 4.0]]
    assert torch.abs(Sigma[0, 0, 0] - 0.25) < 1e-5
    assert torch.abs(Sigma[0, 1, 0]) < 1e-5
    assert torch.abs(Sigma[0, 0, 1]) < 1e-5
    assert torch.abs(Sigma[0, 1, 1] - 4.0) < 1e-5


if __name__ == '__main__':
    test_positive_definite()
    test_isotropic_limit()
    test_anisotropy()
    test_tangential_ge_radial()
    test_build_with_direction()
    print("All covariance tests PASSED")
