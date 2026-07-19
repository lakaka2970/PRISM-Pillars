"""
2D covariance matrix utilities for PRISM-Pillars-RF.

Provides anisotropic covariance construction, validation, Mahalanobis distance,
and related geometric operations for Doppler-aware uncertainty modeling.
"""

import torch
import torch.nn.functional as F

_EPS = 1e-8


def build_anisotropic_covariance(s_r, s_t, u, sigma_0=0.03):
    """
    Build anisotropic 2x2 covariance matrix from radial/tangential uncertainties.

    Sigma_i = s_r^2 * u * u^T + s_t^2 * n * n^T + sigma_0^2 * I

    where:
        s_r = radial standard deviation (Doppler-constrained)
        s_t = tangential standard deviation (poorly observable)
        u   = line-of-sight unit vector [u_x, u_y]
        n   = tangential unit vector [-u_y, u_x]
        sigma_0 = minimum isotropic background uncertainty

    Args:
        s_r: (N,) or (N, 1) float tensor, radial standard deviations.
        s_t: (N,) or (N, 1) float tensor, tangential standard deviations.
        u: (N, 2) float tensor, line-of-sight unit vectors.
        sigma_0: float, isotropic regularization. Defaults to 0.03.

    Returns:
        Sigma: (N, 2, 2) float tensor, covariance matrices per point.
    """
    s_r = s_r.view(-1, 1)
    s_t = s_t.view(-1, 1)
    N = s_r.shape[0]

    # Tangential direction: 90-degree CCW rotation
    n_x = -u[:, 1:2]
    n_y = u[:, 0:1]
    n = torch.cat([n_x, n_y], dim=-1)  # (N, 2)

    # Outer products
    # uu^T: (N, 2, 2)
    u_outer = torch.bmm(u.unsqueeze(-1), u.unsqueeze(-2))
    # nn^T: (N, 2, 2)
    n_outer = torch.bmm(n.unsqueeze(-1), n.unsqueeze(-2))

    # Build covariance
    s_r_sq = (s_r * s_r).unsqueeze(-1)  # (N, 1, 1)
    s_t_sq = (s_t * s_t).unsqueeze(-1)  # (N, 1, 1)

    Sigma = s_r_sq * u_outer + s_t_sq * n_outer
    # Add isotropic background
    eye = torch.eye(2, device=u.device, dtype=u.dtype).unsqueeze(0).expand(N, -1, -1)
    Sigma = Sigma + (sigma_0 * sigma_0) * eye

    return Sigma


def validate_covariance(Sigma):
    """
    Check that all covariance matrices are positive definite.

    Args:
        Sigma: (..., 2, 2) float tensor, covariance matrices.

    Returns:
        is_valid: bool, True if all eigenvalues are positive.
        min_eigenvalue: float, minimum eigenvalue across all matrices.
    """
    # Compute eigenvalues of 2x2 matrices
    # For 2x2 matrix [[a, b], [c, d]]:
    #   trace = a + d
    #   det = a*d - b*c
    #   lambda = (trace +/- sqrt(trace^2 - 4*det)) / 2

    a = Sigma[..., 0, 0]
    b = Sigma[..., 0, 1]
    c = Sigma[..., 1, 0]
    d = Sigma[..., 1, 1]

    trace = a + d
    det = a * d - b * c

    # Discriminant must be non-negative for real eigenvalues
    discriminant = trace * trace - 4.0 * det
    sqrt_disc = torch.sqrt(F.relu(discriminant) + _EPS)

    lambda_min = (trace - sqrt_disc) / 2.0
    lambda_max = (trace + sqrt_disc) / 2.0

    min_eig = lambda_min.min().item()
    is_valid = min_eig > 0

    return is_valid, min_eig, lambda_min, lambda_max


def check_tangential_geq_radial(s_t, s_r):
    """
    Verify physical constraint: tangential uncertainty >= radial uncertainty.

    s_t >= s_r must hold because Doppler directly constrains radial motion
    while tangential motion is not adequately observable.

    Args:
        s_t: (N,) float tensor, tangential standard deviations.
        s_r: (N,) float tensor, radial standard deviations.

    Returns:
        is_valid: bool.
        violations: (N,) bool tensor, points where constraint is violated.
    """
    violations = s_t < s_r
    return violations.sum().item() == 0, violations


def mahalanobis_distance(x, mu, Sigma):
    """
    Compute squared Mahalanobis distance.

    d^2 = (x - mu)^T Sigma^{-1} (x - mu)

    Args:
        x: (..., 2) float tensor, query points.
        mu: (..., 2) float tensor, distribution means.
        Sigma: (..., 2, 2) float tensor, covariance matrices.

    Returns:
        d2: (...,) float tensor, squared Mahalanobis distances.
    """
    diff = x - mu  # (..., 2)

    # Solve Sigma * v = diff for v (equivalent to Sigma^{-1} * diff)
    # Using torch.linalg.solve for numerical stability
    try:
        v = torch.linalg.solve(Sigma, diff.unsqueeze(-1)).squeeze(-1)  # (..., 2)
    except RuntimeError:
        # Fallback: add small regularization if singular
        eye = torch.eye(2, device=Sigma.device, dtype=Sigma.dtype)
        Sigma_reg = Sigma + _EPS * eye.expand_as(Sigma)
        v = torch.linalg.solve(Sigma_reg, diff.unsqueeze(-1)).squeeze(-1)

    d2 = (diff * v).sum(dim=-1)  # (...,)
    return d2


def mahalanobis_bias(c_i, c_j, Sigma_j, sigma_c=0.1):
    """
    Compute Mahalanobis-based geometric bias for temporal attention.

    b_{ij}^{geo} = -0.5 * (c_i - c_j)^T (Sigma_j + sigma_c^2 * I)^{-1} (c_i - c_j)

    This is equivalent to the log of a Gaussian likelihood (up to constant).

    Args:
        c_i: (Q, 2) float tensor, current pillar BEV centers.
        c_j: (K, 2) float tensor, history pillar BEV centers.
        Sigma_j: (K, 2, 2) float tensor, history pillar covariances.
        sigma_c: float, additional isotropic uncertainty. Defaults to 0.1.

    Returns:
        bias: (Q, K) float tensor, geometric attention bias.
    """
    d2 = pairwise_mahalanobis(c_i, c_j, Sigma_j, sigma_c)  # (Q, K)
    return -0.5 * d2


def pairwise_mahalanobis(c_i, c_j, Sigma_j, sigma_c=0.1):
    """
    Efficient pairwise Mahalanobis distance between two sets of points.

    Args:
        c_i: (Q, 2) float tensor, query points.
        c_j: (K, 2) float tensor, key points.
        Sigma_j: (K, 2, 2) float tensor, covariances for key points.
        sigma_c: float, additional isotropic term added to diagonal.

    Returns:
        d2: (Q, K) float tensor, squared Mahalanobis distances.
    """
    Q = c_i.shape[0]
    K = c_j.shape[0]
    diff = c_i.unsqueeze(1) - c_j.unsqueeze(0)  # (Q, K, 2)

    # Add sigma_c^2 to diagonal of each covariance
    eye = torch.eye(2, device=Sigma_j.device, dtype=Sigma_j.dtype)
    Sigma_reg = Sigma_j + (sigma_c * sigma_c) * eye.unsqueeze(0)  # (K, 2, 2)

    # For each query point q, compute diff[q,k]^T @ Sigma_reg[k]^{-1} @ diff[q,k]
    # Vectorized approach: batched solve
    diff_flat = diff.reshape(-1, 2)  # (Q*K, 2)
    Sigma_flat = Sigma_reg.unsqueeze(0).expand(Q, -1, -1, -1).reshape(-1, 2, 2)  # (Q*K, 2, 2)

    # Solve for each pair
    try:
        v = torch.linalg.solve(Sigma_flat, diff_flat.unsqueeze(-1)).squeeze(-1)  # (Q*K, 2)
    except RuntimeError:
        eye = torch.eye(2, device=Sigma_flat.device, dtype=Sigma_flat.dtype)
        Sigma_flat_reg = Sigma_flat + _EPS * eye.unsqueeze(0).expand(Sigma_flat.shape[0], -1, -1)
        v = torch.linalg.solve(Sigma_flat_reg, diff_flat.unsqueeze(-1)).squeeze(-1)

    d2 = (diff_flat * v).sum(dim=-1)  # (Q*K,)
    d2 = d2.reshape(Q, K)

    return d2


def log_det_covariance(Sigma):
    """
    Compute log determinant of 2x2 covariance matrices.

    For 2x2 matrix: det = a*d - b*c

    Args:
        Sigma: (..., 2, 2) float tensor.

    Returns:
        log_det: (...,) float tensor.
    """
    a = Sigma[..., 0, 0]
    b = Sigma[..., 0, 1]
    c = Sigma[..., 1, 0]
    d = Sigma[..., 1, 1]

    det = a * d - b * c
    log_det = torch.log(det.clamp(min=_EPS))

    return log_det


def aggregate_covariances_to_pillar(Sigma_per_point, weights, pillar_indices, num_pillars):
    """
    Aggregate per-point covariances to per-pillar covariances using weighted averaging.

    Sigma_pillar = sum_i w_i * Sigma_i / sum_i w_i

    Args:
        Sigma_per_point: (N, 2, 2) float tensor.
        weights: (N,) float tensor, per-point weights.
        pillar_indices: (N,) long tensor, pillar assignments.
        num_pillars: int.

    Returns:
        Sigma_pillar: (num_pillars, 2, 2) float tensor.
    """
    device = Sigma_per_point.device
    dtype = Sigma_per_point.dtype
    N = Sigma_per_point.shape[0]

    Sigma_pillar = torch.zeros(num_pillars, 2, 2, device=device, dtype=dtype)
    weight_sum = torch.zeros(num_pillars, device=device, dtype=dtype)

    # Weighted scatter add
    weighted = Sigma_per_point * weights.view(N, 1, 1)  # (N, 2, 2)
    flat_weighted = weighted.reshape(N, 4)
    flat_Sigma = Sigma_pillar.reshape(num_pillars, 4)
    flat_Sigma = flat_Sigma.scatter_add(0, pillar_indices.unsqueeze(-1).expand(-1, 4), flat_weighted)
    Sigma_pillar = flat_Sigma.reshape(num_pillars, 2, 2)

    weight_sum = weight_sum.scatter_add(0, pillar_indices, weights)

    # Normalize
    weight_sum = weight_sum.clamp(min=_EPS).view(-1, 1, 1)
    Sigma_pillar = Sigma_pillar / weight_sum

    return Sigma_pillar
