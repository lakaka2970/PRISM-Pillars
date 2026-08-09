"""
Unified radar point feature embedding for PRISM-Pillars-RF.

Implements the point format from paper Section 4:
    x, y, z, log_rcs, v_rel, v_comp, v_x, v_y,
    delta_t, range, sin_azimuth, cos_azimuth,
    local_density, local_rcs_mean, local_doppler_std, ego_comp_residual

Shared between current-frame and historical branches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.radar_geometry import (
    compute_line_of_sight_unit_vector,
    compute_tangential_direction,
    decompose_radial_velocity,
    compute_azimuth_encoding,
    compute_range,
    compute_local_point_stats,
)


class RadarPointEmbedding(nn.Module):
    """
    Shared point-level feature embedding for current and historical radar points.

    Feature indices expected in input (configurable):
        0: x, 1: y, 2: z, 3: rcs, 4: v_rel, 5: v_comp, [6: time/delta_t]

    Output features:
        x, y, z, log_rcs, v_rel, v_comp, v_x, v_y,
        delta_t, range, sin_azimuth, cos_azimuth,
        local_density, local_rcs_mean, local_doppler_std, ego_comp_residual

    Config:
        OUTPUT_DIM: int, output feature dimension (default 32)
        USE_RCS: bool
        USE_RELATIVE_VELOCITY: bool
        USE_COMPENSATED_VELOCITY: bool
        USE_VELOCITY_COMPONENTS: bool
        USE_DELTA_T: bool
        USE_AZIMUTH_ENCODING: bool
        RCS_INDEX, VELOCITY_REL_INDEX, VELOCITY_COMP_INDEX, TIME_INDEX
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

        self.use_rcs = model_cfg.get('USE_RCS', True)
        self.use_relative_velocity = model_cfg.get('USE_RELATIVE_VELOCITY', True)
        self.use_compensated_velocity = model_cfg.get('USE_COMPENSATED_VELOCITY', True)
        self.use_velocity_components = model_cfg.get('USE_VELOCITY_COMPONENTS', True)
        self.use_delta_t = model_cfg.get('USE_DELTA_T', True)
        self.use_azimuth_encoding = model_cfg.get('USE_AZIMUTH_ENCODING', True)

        self.rcs_index = model_cfg.get('RCS_INDEX', 3)
        self.velocity_rel_index = model_cfg.get('VELOCITY_REL_INDEX', None)
        self.velocity_comp_index = model_cfg.get('VELOCITY_COMP_INDEX', 5)
        self.time_index = model_cfg.get('TIME_INDEX', None)

        # Compute input feature dimension
        num_features = 3  # x, y, z
        if self.use_rcs:
            num_features += 1
        if self.use_relative_velocity:
            num_features += 1
        if self.use_compensated_velocity:
            num_features += 1
        if self.use_velocity_components:
            num_features += 2  # v_x, v_y
        if self.use_delta_t:
            num_features += 1
        if self.use_azimuth_encoding:
            num_features += 2  # sin_azimuth, cos_azimuth
        num_features += 1  # range
        num_features += 3  # local stats: density, rcs_mean, doppler_std

        self.input_dim = num_features
        self.output_dim = model_cfg.get('OUTPUT_DIM', 32)

        # Shared MLP embedding
        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim * 2),
            nn.LayerNorm(self.output_dim * 2),
            nn.SiLU(),
            nn.Linear(self.output_dim * 2, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.SiLU(),
        )

    def build_point_features(self, points, delta_t=None):
        """
        Build unified point feature vector from raw radar points.

        Args:
            points: (N, D_raw) float tensor, raw radar points.
                    Expected columns: [x, y, z, rcs, v_rel?, v_comp?, time?]
            delta_t: (N,) float tensor (optional). Per-point time delta.
                     If None, uses TIME_INDEX from points or zeros.

        Returns:
            features: (N, input_dim) float tensor, constructed features.
        """
        N = points.shape[0]
        device = points.device
        dtype = points.dtype
        feature_list = []

        # --- Core coordinates ---
        x = points[:, 0:1]
        y = points[:, 1:2]
        z = points[:, 2:3]
        feature_list.extend([x, y, z])

        # --- RCS (log scale) ---
        if self.use_rcs:
            rcs = points[:, self.rcs_index : self.rcs_index + 1]
            # Log RCS (clamp to avoid log(0))
            log_rcs = torch.log(rcs.clamp(min=1e-8))
            feature_list.append(log_rcs)
        else:
            feature_list.append(torch.zeros(N, 1, device=device, dtype=dtype))

        # --- Relative radial velocity ---
        if self.use_relative_velocity and self.velocity_rel_index is not None:
            v_rel = points[:, self.velocity_rel_index : self.velocity_rel_index + 1]
            feature_list.append(v_rel)
        else:
            feature_list.append(torch.zeros(N, 1, device=device, dtype=dtype))

        # --- Compensated radial velocity ---
        if self.use_compensated_velocity:
            v_comp = points[:, self.velocity_comp_index : self.velocity_comp_index + 1]
            feature_list.append(v_comp)
        else:
            v_comp = torch.zeros(N, 1, device=device, dtype=dtype)
            feature_list.append(v_comp)

        # --- Velocity components (v_x, v_y from Doppler) ---
        if self.use_velocity_components:
            u = compute_line_of_sight_unit_vector(x, y)  # (N, 2)
            if self.use_compensated_velocity:
                v_comp_xy = decompose_radial_velocity(v_comp.squeeze(-1), u)
            else:
                v_comp_xy = torch.zeros(N, 2, device=device, dtype=dtype)
            feature_list.append(v_comp_xy[:, 0:1])  # v_x
            feature_list.append(v_comp_xy[:, 1:2])  # v_y
        else:
            feature_list.extend([
                torch.zeros(N, 1, device=device, dtype=dtype),
                torch.zeros(N, 1, device=device, dtype=dtype),
            ])

        # --- Time delta ---
        if self.use_delta_t:
            if delta_t is not None:
                dt = delta_t.view(N, 1)
            elif self.time_index is not None:
                dt = points[:, self.time_index : self.time_index + 1]
            else:
                dt = torch.zeros(N, 1, device=device, dtype=dtype)
            feature_list.append(dt)
        else:
            feature_list.append(torch.zeros(N, 1, device=device, dtype=dtype))

        # --- Range ---
        r = compute_range(x.squeeze(-1), y.squeeze(-1))
        feature_list.append(r)

        # --- Azimuth encoding ---
        if self.use_azimuth_encoding:
            sin_az, cos_az = compute_azimuth_encoding(x.squeeze(-1), y.squeeze(-1))
            feature_list.append(sin_az)
            feature_list.append(cos_az)
        else:
            feature_list.extend([
                torch.zeros(N, 1, device=device, dtype=dtype),
                torch.zeros(N, 1, device=device, dtype=dtype),
            ])

        # --- Local point statistics ---
        points_xy = torch.cat([x, y], dim=-1)  # (N, 2)
        # Use v_comp (index ~3-5) as feature for local stats
        local_features = torch.cat([feature_list[3], feature_list[5]], dim=-1) if len(feature_list) > 5 else x
        density, rcs_mean, doppler_std = compute_local_point_stats(points_xy, local_features, k=8)
        feature_list.append(density)
        feature_list.append(rcs_mean)
        feature_list.append(doppler_std)

        features = torch.cat(feature_list, dim=-1)
        return features

    def forward(self, points, delta_t=None):
        """
        Args:
            points: (N, D_raw) float tensor, raw radar points.
            delta_t: (N,) float tensor (optional), per-point time deltas.

        Returns:
            embedded: (N, output_dim) float tensor, embedded point features.
        """
        point_features = self.build_point_features(points, delta_t=delta_t)
        embedded = self.embedding_mlp(point_features)
        return embedded
