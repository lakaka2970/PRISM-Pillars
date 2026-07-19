"""
Radar geometry utilities for PRISM-Pillars-RF.

Provides unified functions for line-of-sight computation, velocity decomposition,
azimuth encoding, local point statistics, and ego-motion residual calculation.
"""

import torch
import torch.nn.functional as F

_EPS = 1e-8


def compute_line_of_sight_unit_vector(x, y):
    """
    Compute line-of-sight unit vector u_i for radar points.

    u_i = [x_i, y_i]^T / sqrt(x_i^2 + y_i^2 + epsilon)

    Args:
        x: (N,) or (N, 1) float tensor, x-coordinates in sensor frame.
        y: (N,) or (N, 1) float tensor, y-coordinates in sensor frame.

    Returns:
        u: (N, 2) float tensor, unit vectors [u_x, u_y].
    """
    x = x.view(-1)
    y = y.view(-1)
    r = torch.sqrt(x * x + y * y) + _EPS
    u_x = x / r
    u_y = y / r
    return torch.stack([u_x, u_y], dim=-1)


def compute_tangential_direction(u):
    """
    Compute tangential (cross-range) direction from line-of-sight unit vector.

    n_i = [-u_y, u_x]^T

    This is a 90-degree counter-clockwise rotation of u in the BEV plane.

    Args:
        u: (..., 2) float tensor, line-of-sight unit vectors.

    Returns:
        n: (..., 2) float tensor, tangential direction vectors.
    """
    n_x = -u[..., 1:2]
    n_y = u[..., 0:1]
    return torch.cat([n_x, n_y], dim=-1)


def decompose_radial_velocity(v_comp, u):
    """
    Decompose compensated radial velocity into x/y components.

    v_{x,i} = v^{comp}_{r,i} * u_{x,i}
    v_{y,i} = v^{comp}_{r,i} * u_{y,i}

    Args:
        v_comp: (N,) or (N, 1) float tensor, compensated radial velocity.
        u: (N, 2) float tensor, line-of-sight unit vectors.

    Returns:
        v_xy: (N, 2) float tensor, [v_x, v_y] per point.
    """
    v_comp = v_comp.view(-1, 1)
    return v_comp * u


def compute_azimuth_encoding(x, y):
    """
    Compute sin/cos azimuth encoding for each radar point.

    Args:
        x: (N,) or (N, 1) float tensor, x-coordinates.
        y: (N,) or (N, 1) float tensor, y-coordinates.

    Returns:
        sin_azimuth: (N, 1) float tensor.
        cos_azimuth: (N, 1) float tensor.
    """
    x = x.view(-1)
    y = y.view(-1)
    azimuth = torch.atan2(y, x + _EPS)
    sin_az = torch.sin(azimuth).unsqueeze(-1)
    cos_az = torch.cos(azimuth).unsqueeze(-1)
    return sin_az, cos_az


def compute_range(x, y):
    """
    Compute range (radial distance in BEV plane) for each point.

    Args:
        x: (N,) float tensor.
        y: (N,) float tensor.

    Returns:
        r: (N, 1) float tensor, range values.
    """
    return torch.sqrt(x * x + y * y + _EPS).unsqueeze(-1)


def compute_local_point_stats(points_xy, features, k=8):
    """
    Compute local statistics for each point using k-nearest neighbors.

    Args:
        points_xy: (N, 2) float tensor, BEV coordinates [x, y].
        features: (N, D) float tensor, point features (RCS, Doppler, etc.).
        k: Number of neighbors. Defaults to 8.

    Returns:
        local_density: (N, 1) float tensor, mean distance to k neighbors.
        local_rcs_mean: (N, 1) float tensor, local mean RCS (feature dim 0).
        local_doppler_std: (N, 1) float tensor, local Doppler std (feature dim 3).
    """
    N = points_xy.shape[0]
    if N < 2:
        return (
            torch.zeros(N, 1, device=points_xy.device),
            torch.zeros(N, 1, device=points_xy.device),
            torch.zeros(N, 1, device=points_xy.device),
        )

    # Compute pairwise distances
    dist_matrix = torch.cdist(points_xy, points_xy, p=2)  # (N, N)

    # Find k-nearest neighbors (excluding self)
    k_actual = min(k + 1, N)
    topk_dist, topk_idx = torch.topk(dist_matrix, k=k_actual, dim=-1, largest=False)

    # Remove self (first neighbor is the point itself)
    if k_actual > 1:
        topk_dist = topk_dist[:, 1:]  # distances to k neighbors (excluding self)
        topk_idx = topk_idx[:, 1:]    # indices of k neighbors
    else:
        topk_dist = topk_dist
        topk_idx = topk_idx[:, 1:]

    # Local density: mean distance to k neighbors
    local_density = topk_dist.float().mean(dim=-1, keepdim=True)

    # Local RCS mean (assuming RCS is feature dim 0)
    if features.shape[1] > 0:
        # Gather neighbor features
        neighbor_rcs = features[topk_idx, 0]  # (N, k)
        local_rcs_mean = neighbor_rcs.float().mean(dim=-1, keepdim=True)
    else:
        local_rcs_mean = torch.zeros(N, 1, device=points_xy.device)

    # Local Doppler std (assuming Doppler/v_comp is feature dim 3)
    if features.shape[1] > 3:
        neighbor_doppler = features[topk_idx, 3]  # (N, k)
        local_doppler_std = neighbor_doppler.float().std(dim=-1, keepdim=True)
    else:
        local_doppler_std = torch.zeros(N, 1, device=points_xy.device)

    return local_density, local_rcs_mean, local_doppler_std


def compute_ego_motion_residual(points, ego_delta_pose):
    """
    Compute residual between ego-motion compensated position and raw position.

    Args:
        points: (N, 3) float tensor, point positions AFTER ego-motion alignment.
        ego_delta_pose: (4, 4) float tensor, ego-motion transform used.

    Returns:
        residual: (N, 1) float tensor, per-point compensation residual magnitude.
    """
    # For a pass-through when no ego-motion info is available
    if ego_delta_pose is None:
        return torch.zeros(points.shape[0], 1, device=points.device)

    # Compute residual as translational component magnitude
    translation = ego_delta_pose[:3, 3]
    residual_mag = torch.norm(translation).expand(points.shape[0], 1)
    return residual_mag


def compute_mean_direction_per_pillar(u_vectors, pillar_indices, num_pillars):
    """
    Compute mean line-of-sight direction per pillar for covariance aggregation.

    Args:
        u_vectors: (N, 2) float tensor, per-point LOS unit vectors.
        pillar_indices: (N,) long tensor, pillar assignment per point.
        num_pillars: int, total number of pillars.

    Returns:
        mean_u: (num_pillars, 2) float tensor, mean direction per pillar.
    """
    mean_u = torch.zeros(num_pillars, 2, device=u_vectors.device)
    ones = torch.ones(num_pillars, device=u_vectors.device)

    # Scatter sum
    counts = torch.zeros(num_pillars, device=u_vectors.device)
    mean_u = mean_u.scatter_add(0, pillar_indices.unsqueeze(-1).expand(-1, 2), u_vectors)
    counts = counts.scatter_add(0, pillar_indices, ones)

    counts = counts.clamp(min=1).unsqueeze(-1)
    mean_u = mean_u / counts

    # Re-normalize direction vectors
    norm = torch.norm(mean_u, dim=-1, keepdim=True) + _EPS
    mean_u = mean_u / norm

    return mean_u
