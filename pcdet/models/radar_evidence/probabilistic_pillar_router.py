"""
RAPR: Reliability-Aware Probabilistic Routing (Paper Section 7 / Module D).

Routes historical radar points to neighboring BEV pillars using a soft,
probability-weighted scheme that respects:
    1. Anisotropic motion uncertainty (Doppler tube)
    2. Point-level temporal reliability (STER)
    3. Evidence mass gating for low-support regions

Key design principle (paper):
    "First normalize geometry, THEN multiply by reliability"
    This ordering prevents q_i from being canceled out in normalization.

Outputs per history pillar:
    - features (weighted average of point embeddings)
    - evidence_mass (sum of weights, for confidence gating)
    - pillar_reliability (weighted average of point reliabilities)
    - pillar_covariance (aggregated covariance)
    - mean_delta_t (for temporal decay)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....utils.covariance_2d import mahalanobis_distance


class ProbabilisticPillarRouter(nn.Module):
    """
    Routes historical points to BEV pillars using reliability-weighted
    Gaussian probabilities derived from anisotropic uncertainty.
    """

    def __init__(self, model_cfg):
        """
        Args (from model_cfg.PROBABILISTIC_ROUTING):
            NEIGHBOR_SIZE: int (default 5), pillar search radius K_r.
            USE_RELIABILITY: bool (default True)
            USE_EVIDENCE_MASS_GATE: bool (default True)
            MAX_HISTORY_POINTS: int (default 2048)
            MIN_RELIABILITY: float (default 0.05)
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.neighbor_size = model_cfg.get('NEIGHBOR_SIZE', 5)
        self.use_reliability = model_cfg.get('USE_RELIABILITY', True)
        self.use_evidence_mass_gate = model_cfg.get('USE_EVIDENCE_MASS_GATE', True)
        self.max_history_points = model_cfg.get('MAX_HISTORY_POINTS', 2048)
        self.min_reliability = model_cfg.get('MIN_RELIABILITY', 0.05)

        # BEV grid parameters (set during first forward or from config)
        self._voxel_size = None
        self._pc_range = None
        self._grid_shape = None

    def set_grid_params(self, voxel_size, pc_range, grid_shape):
        """Set BEV grid parameters for pillar coordinate mapping."""
        self._voxel_size = (voxel_size[0], voxel_size[1])
        self._pc_range = pc_range  # [xmin, ymin, zmin, xmax, ymax, zmax]
        self._grid_shape = grid_shape  # [nx, ny, nz]

    def _world_to_pillar_center(self, x, y, batch_size=1):
        """
        Convert world BEV coordinates to pillar centers + indices.

        Args:
            x, y: (N,) float tensors in world coordinates.
            batch_size: int.

        Returns:
            pillar_centers: (N, 2) float tensor of pillar center coordinates.
            pillar_indices: (N,) long tensor of flattened pillar indices.
            valid_mask: (N,) bool tensor.
        """
        if self._voxel_size is None:
            raise RuntimeError("Must call set_grid_params before routing")

        vx, vy = self._voxel_size
        xmin, ymin = self._pc_range[0], self._pc_range[1]
        xmax, ymax = self._pc_range[3], self._pc_range[4]
        nx = int((xmax - xmin) / vx)
        ny = int((ymax - ymin) / vy)

        # Pillar integer coordinates
        px = ((x - xmin) / vx).long()
        py = ((y - ymin) / vy).long()

        valid = (px >= 0) & (px < nx) & (py >= 0) & (py < ny)

        # Pillar centers in world coordinates
        cx = (px.float() + 0.5) * vx + xmin
        cy = (py.float() + 0.5) * vy + ymin

        # Flattened index
        idx = py * nx + px  # (N,)

        return torch.stack([cx, cy], dim=-1), idx, valid

    def _search_neighbor_pillars(self, mu_x, mu_y, valid_mask):
        """
        Search K_r x K_r neighboring pillars around each point's predicted mean.

        Args:
            mu_x, mu_y: (N,) float tensors.
            valid_mask: (N,) bool tensor.

        Returns:
            neighbor_centers: (N, K, 2) float tensor.
            neighbor_indices: (N, K) long tensor.
            valid_neighbors: (N, K) bool tensor.
        """
        N = mu_x.shape[0]
        K = self.neighbor_size
        half_K = K // 2
        device = mu_x.device

        if self._voxel_size is None:
            raise RuntimeError("Must call set_grid_params before routing")

        vx, vy = self._voxel_size
        xmin, ymin = self._pc_range[0], self._pc_range[1]
        xmax, ymax = self._pc_range[3], self._pc_range[4]
        nx = int((xmax - xmin) / vx)
        ny = int((ymax - ymin) / vy)

        # Pillar coordinates at mu
        px = ((mu_x - xmin) / vx).long()
        py = ((mu_y - ymin) / vy).long()

        # Generate offsets
        offsets = torch.arange(-half_K, half_K + 1, device=device)
        offset_grid_x, offset_grid_y = torch.meshgrid(offsets, offsets, indexing='ij')
        offsets_flat = torch.stack([
            offset_grid_x.reshape(-1),
            offset_grid_y.reshape(-1),
        ], dim=-1)  # (K*K, 2)

        # Apply offsets
        px_grid = px.unsqueeze(-1) + offsets_flat[:, 0].unsqueeze(0)  # (N, K*K)
        py_grid = py.unsqueeze(-1) + offsets_flat[:, 1].unsqueeze(0)  # (N, K*K)

        # Clamp to grid bounds
        px_grid = px_grid.clamp(0, nx - 1)
        py_grid = py_grid.clamp(0, ny - 1)

        # World coordinates of neighboring pillar centers
        cx_grid = (px_grid.float() + 0.5) * vx + xmin
        cy_grid = (py_grid.float() + 0.5) * vy + ymin

        # Flattened indices
        idx_grid = py_grid * nx + px_grid

        # Valid mask
        valid_grid = valid_mask.unsqueeze(-1).expand(-1, K * K)

        return (
            torch.stack([cx_grid, cy_grid], dim=-1),  # (N, K*K, 2)
            idx_grid,  # (N, K*K)
            valid_grid,  # (N, K*K)
        )

    def forward(
        self,
        point_features,
        mean,
        covariance,
        reliability,
        batch_idx,
        delta_t=None,
    ):
        """
        Route historical points to BEV pillars.

        Algorithm (paper Section 7):
        1. For each historical point i, search neighboring pillars
        2. Compute geometric probability g_ij from anisotropic Gaussian
        3. Normalize geometry FIRST: pi_ij = g_ij / sum_j' g_ij'
        4. THEN multiply by reliability: w_ij = q_i * pi_ij
        5. Aggregate per pillar: weighted average of point features
        6. Apply evidence mass gate: H_j = (1 - exp(-m_j)) * H_j_raw

        Args:
            point_features: (N, F) float tensor, embedded point features phi(z_i).
            mean: (N, 2) float tensor, predicted BEV positions mu_i.
            covariance: (N, 2, 2) float tensor, uncertainty Sigma_i.
            reliability: (N, 1) float tensor, reliability scores q_i.
            batch_idx: (N,) long tensor, batch index per point.
            delta_t: (N,) float tensor (optional), time deltas.

        Returns:
            dict with keys:
                features: (num_pillars, F) float tensor
                evidence_mass: (num_pillars,) float tensor
                pillar_reliability: (num_pillars,) float tensor
                pillar_covariance: (num_pillars, 2, 2) float tensor
                mean_delta_t: (num_pillars,) float tensor
                coords: (num_pillars, 3) long tensor [batch, y, x]
        """
        N = point_features.shape[0]
        device = point_features.device
        dtype = point_features.dtype
        F_dim = point_features.shape[1]

        if N == 0:
            return {
                'features': torch.zeros(0, F_dim, device=device, dtype=dtype),
                'evidence_mass': torch.zeros(0, device=device, dtype=dtype),
                'pillar_reliability': torch.zeros(0, device=device, dtype=dtype),
                'pillar_covariance': torch.zeros(0, 2, 2, device=device, dtype=dtype),
                'mean_delta_t': torch.zeros(0, device=device, dtype=dtype),
                'coords': torch.zeros(0, 3, dtype=torch.long, device=device),
            }

        # Optional: subsample history points
        if N > self.max_history_points:
            keep_idx = torch.randperm(N, device=device)[: self.max_history_points]
            point_features = point_features[keep_idx]
            mean = mean[keep_idx]
            covariance = covariance[keep_idx]
            reliability = reliability[keep_idx]
            batch_idx = batch_idx[keep_idx]
            if delta_t is not None:
                delta_t = delta_t[keep_idx]
            N = self.max_history_points

        reliability = reliability.view(N)
        q = reliability.clamp(min=self.min_reliability) if self.use_reliability else torch.ones(N, device=device)

        # === Step 1: Search neighboring pillars ===
        mu_x = mean[:, 0]
        mu_y = mean[:, 1]
        valid_mask = torch.ones(N, dtype=torch.bool, device=device)

        neighbor_centers, neighbor_indices, neighbor_valid = self._search_neighbor_pillars(
            mu_x, mu_y, valid_mask
        )
        K_total = neighbor_centers.shape[1]  # K_r * K_r

        # === Step 2: Geometric probability (Gaussian) ===
        # g_ij = exp(-0.5 * (c_j - mu_i)^T Sigma_i^{-1} (c_j - mu_i))
        # Vectorized: compute for all point-neighbor pairs

        diff = neighbor_centers - mean.unsqueeze(1)  # (N, K_total, 2)

        # Batch solve: Sigma_i @ v_ij = diff_ij -> v_ij = Sigma_i^{-1} @ diff_ij
        diff_flat = diff.reshape(N * K_total, 2).unsqueeze(-1)  # (N*K_total, 2, 1)
        Sigma_flat = covariance.unsqueeze(1).expand(-1, K_total, -1, -1).reshape(N * K_total, 2, 2)

        try:
            v_flat = torch.linalg.solve(Sigma_flat, diff_flat)  # (N*K_total, 2, 1)
        except RuntimeError:
            eye = torch.eye(2, device=device, dtype=dtype)
            Sigma_flat_reg = Sigma_flat + 1e-6 * eye.unsqueeze(0)
            v_flat = torch.linalg.solve(Sigma_flat_reg, diff_flat)

        diff_flat_sq = diff_flat.view(N * K_total, 2)
        v_flat_sq = v_flat.view(N * K_total, 2)
        d2 = (diff_flat_sq * v_flat_sq).sum(dim=-1)  # (N*K_total,)
        d2 = d2.view(N, K_total)  # (N, K_total)

        g = torch.exp(-0.5 * d2)  # (N, K_total)
        g = g * neighbor_valid.float()  # Mask invalid pillars

        # === Step 3: Normalize geometry FIRST ===
        g_sum = g.sum(dim=-1, keepdim=True) + 1e-8
        pi = g / g_sum  # (N, K_total)

        # === Step 4: Multiply by reliability AFTER normalization ===
        w = q.unsqueeze(-1) * pi  # (N, K_total)

        # === Step 5: Aggregate per pillar ===
        # We need per-pillar accumulation. Find all (point, pillar) pairs with w > 0.
        # Use sparse accumulation approach with batch-pillar indexing.

        # Flatten for scatter
        w_flat = w.reshape(-1)  # (N*K_total,)
        idx_flat = neighbor_indices.reshape(-1)  # (N*K_total,)

        # Filter zero-weighted entries for efficiency
        nonzero_mask = w_flat > 1e-8
        w_nz = w_flat[nonzero_mask]
        idx_nz = idx_flat[nonzero_mask]

        if w_nz.numel() == 0:
            return {
                'features': torch.zeros(0, F_dim, device=device, dtype=dtype),
                'evidence_mass': torch.zeros(0, device=device, dtype=dtype),
                'pillar_reliability': torch.zeros(0, device=device, dtype=dtype),
                'pillar_covariance': torch.zeros(0, 2, 2, device=device, dtype=dtype),
                'mean_delta_t': torch.zeros(0, device=device, dtype=dtype),
                'coords': torch.zeros(0, 3, dtype=torch.long, device=device),
            }

        # Get unique pillar indices
        unique_indices, inverse_indices = torch.unique(idx_nz, return_inverse=True)
        num_pillars = unique_indices.shape[0]

        # Scatter add for each quantity
        # --- Features ---
        feat_nz = point_features.unsqueeze(1).expand(-1, K_total, -1).reshape(N * K_total, F_dim)[nonzero_mask]
        feat_weighted = feat_nz * w_nz.unsqueeze(-1)  # (M, F)
        feat_accum = torch.zeros(num_pillars, F_dim, device=device, dtype=dtype)
        feat_accum = feat_accum.scatter_add(0, inverse_indices.unsqueeze(-1).expand(-1, F_dim), feat_weighted)

        # --- Evidence mass ---
        mass_accum = torch.zeros(num_pillars, device=device, dtype=dtype)
        mass_accum = mass_accum.scatter_add(0, inverse_indices, w_nz)

        # --- Reliability (weighted average) ---
        q_per_pair = q.unsqueeze(-1).expand(-1, K_total).reshape(-1)[nonzero_mask]
        q_weighted = q_per_pair * w_nz
        q_accum = torch.zeros(num_pillars, device=device, dtype=dtype)
        q_accum = q_accum.scatter_add(0, inverse_indices, q_weighted)
        pillar_reliability = q_accum / (mass_accum + 1e-8)

        # --- Pillar covariance (weighted average) ---
        cov_flat = covariance.unsqueeze(1).expand(-1, K_total, -1, -1).reshape(N * K_total, 2, 2)[nonzero_mask]
        cov_weighted = cov_flat * w_nz.view(-1, 1, 1)
        cov_accum = torch.zeros(num_pillars, 2, 2, device=device, dtype=dtype)
        cov_flat_4 = cov_weighted.view(-1, 4)
        cov_acc_4 = torch.zeros(num_pillars, 4, device=device, dtype=dtype)
        cov_acc_4 = cov_acc_4.scatter_add(0, inverse_indices.unsqueeze(-1).expand(-1, 4), cov_flat_4)
        pillar_covariance = cov_acc_4.view(num_pillars, 2, 2) / (mass_accum.view(-1, 1, 1) + 1e-8)

        # --- Mean delta_t ---
        if delta_t is not None:
            dt_flat = delta_t.unsqueeze(-1).expand(-1, K_total).reshape(-1)[nonzero_mask]
            dt_weighted = dt_flat * w_nz
            dt_accum = torch.zeros(num_pillars, device=device, dtype=dtype)
            dt_accum = dt_accum.scatter_add(0, inverse_indices, dt_weighted)
            mean_delta_t = dt_accum / (mass_accum + 1e-8)
        else:
            mean_delta_t = torch.zeros(num_pillars, device=device, dtype=dtype)

        # === Step 6: Evidence mass gate ===
        raw_features = feat_accum / (mass_accum.view(-1, 1) + 1e-8)

        if self.use_evidence_mass_gate:
            gate = 1.0 - torch.exp(-mass_accum)  # (num_pillars,)
            features = raw_features * gate.unsqueeze(-1)
        else:
            features = raw_features

        # === Convert pillar indices to 3D coords [batch_id, y, x] ===
        if self._voxel_size is not None:
            vx = self._voxel_size[0]
            xmin, ymin = self._pc_range[0], self._pc_range[1]
            xmax = self._pc_range[3]
            nx = int((xmax - xmin) / vx)
            # unique_indices is flattened (y * nx + x)
            py = unique_indices // nx
            px = unique_indices % nx
            # No batch separation for now (all same batch)
            coords = torch.stack([
                torch.zeros(num_pillars, dtype=torch.long, device=device),
                py,
                px,
            ], dim=-1)
        else:
            coords = torch.zeros(num_pillars, 3, dtype=torch.long, device=device)

        return {
            'features': features,
            'evidence_mass': mass_accum,
            'pillar_reliability': pillar_reliability,
            'pillar_covariance': pillar_covariance,
            'mean_delta_t': mean_delta_t,
            'coords': coords,
        }
