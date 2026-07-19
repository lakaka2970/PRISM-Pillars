"""
Test 1: Time direction sign validation (Paper Section 18.1).

Construct synthetic case:
    - Point directly in front of radar (+x direction)
    - Positive radial velocity (moving away)
    - Known delta_t

Verify: mean prediction shifts in the correct radial direction.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

# These utility functions have no heavy dependencies (no CUDA ops needed)
from pcdet.utils.radar_geometry import (
    compute_line_of_sight_unit_vector,
    decompose_radial_velocity,
)


def test_positive_velocity_shifts_forward():
    """Point moving away (+v_comp) should shift outward along LOS."""
    # Point at x=10, y=0 (straight ahead), z=0
    point = torch.tensor([[10.0, 0.0, 0.0, -10.0, 0.0, 3.0, 2.0]])  # v_comp=3.0, dt=2.0
    dt = torch.tensor([2.0])
    u = compute_line_of_sight_unit_vector(point[:, 0], point[:, 1])

    # Deterministic prediction
    mu = point[:, :2] + dt.view(1, 1) * point[:, 5:6] * u  # p + dt * v_comp * u

    # u should be [1, 0] (pointing right/forward)
    assert torch.abs(u[0, 0] - 1.0) < 1e-6, f"Expected u_x=1.0, got {u[0, 0]}"
    assert torch.abs(u[0, 1]) < 1e-6, f"Expected u_y=0, got {u[0, 1]}"

    # mu should be [10 + 2*3*1, 0 + 2*3*0] = [16, 0]
    assert torch.abs(mu[0, 0] - 16.0) < 1e-6, f"Expected mu_x=16.0, got {mu[0, 0]}"
    assert torch.abs(mu[0, 1]) < 1e-6, f"Expected mu_y=0, got {mu[0, 1]}"


def test_negative_velocity_shifts_backward():
    """Point with negative v_comp should shift inward (toward sensor)."""
    point = torch.tensor([[10.0, 0.0, 0.0, -10.0, 0.0, -3.0, 1.0]])  # v_comp=-3.0, dt=1.0
    dt = torch.tensor([1.0])
    u = compute_line_of_sight_unit_vector(point[:, 0], point[:, 1])

    mu = point[:, :2] + dt.view(1, 1) * point[:, 5:6] * u

    # Should be closer to origin: [10 + 1*(-3)*1, 0] = [7, 0]
    assert torch.abs(mu[0, 0] - 7.0) < 1e-6, f"Expected mu_x=7.0, got {mu[0, 0]}"


def test_lateral_point_direction():
    """Point to the side (y-offset) should have correct LOS direction."""
    point = torch.tensor([[0.0, 10.0, 0.0, -10.0, 0.0, 2.0, 0.5]])  # At +y
    u = compute_line_of_sight_unit_vector(point[:, 0], point[:, 1])

    # u should be approximately [0, 1]
    assert torch.abs(u[0, 0]) < 1e-6, f"Expected u_x~0, got {u[0, 0]}"
    assert torch.abs(u[0, 1] - 1.0) < 1e-6, f"Expected u_y=1.0, got {u[0, 1]}"


def test_velocity_decomposition():
    """Velocity decomposition should produce correct x/y components."""
    x = torch.tensor([3.0, 1.0])
    y = torch.tensor([4.0, 0.0])
    v_comp = torch.tensor([5.0, 2.0])

    u = compute_line_of_sight_unit_vector(x, y)
    # Point 1: u ~ [3/5, 4/5] = [0.6, 0.8]
    # Point 2: u ~ [1.0, 0.0]
    assert torch.abs(u[0, 0] - 0.6) < 1e-6
    assert torch.abs(u[0, 1] - 0.8) < 1e-6
    assert torch.abs(u[1, 0] - 1.0) < 1e-6

    v_xy = decompose_radial_velocity(v_comp, u)
    # Point 1: vx=5*0.6=3.0, vy=5*0.8=4.0
    assert torch.abs(v_xy[0, 0] - 3.0) < 1e-6
    assert torch.abs(v_xy[0, 1] - 4.0) < 1e-6
    # Point 2: vx=2*1=2.0, vy=0
    assert torch.abs(v_xy[1, 0] - 2.0) < 1e-6
    assert torch.abs(v_xy[1, 1]) < 1e-6


def test_zero_delta_t():
    """Zero time delta should produce no position shift."""
    point = torch.tensor([[5.0, 3.0, 0.0, -10.0, 0.0, 5.0, 0.0]])
    dt = torch.tensor([0.0])

    mu = point[:, :2] + dt.view(1, 1) * point[:, 5:6] * compute_line_of_sight_unit_vector(
        point[:, 0], point[:, 1]
    )

    assert torch.allclose(mu, point[:, :2])


if __name__ == '__main__':
    test_positive_velocity_shifts_forward()
    test_negative_velocity_shifts_backward()
    test_lateral_point_direction()
    test_velocity_decomposition()
    test_zero_delta_t()
    print("All time-sign tests PASSED")
