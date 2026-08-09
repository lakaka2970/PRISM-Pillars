"""
Radar Process Augmentation for cross-domain robustness.

Implements physics-aware radar point perturbations that simulate
real-world sensor variations (weather, calibration drift, multipath)
without breaking the geometric structure of the scene.

Design follows paper_plans/great_upgrader_2.md §D and
paper_plans/construct_guide.md §4.6.

Supported augmentations:
  - RCS scale / shift (radar gain, weather attenuation)
  - Range-dependent dropout (far-range miss, low-RCS targets)
  - Azimuth noise (angular resolution error)
  - Doppler bias / scale (ego-compensation error, velocity noise)
  - Ego-motion compensation noise
  - Sweep dropout (missing frames)
  - Local ghost injection (multipath reflection)
"""

import numpy as np


class RadarProcessAugmentor:
    """
    Physics-aware radar point augmentation for domain randomization.

    Applied at the dataset level before point feature encoding.
    Operates on raw radar point feature columns.

    Args:
        config: RADAR_AUG config dict.
        feature_indices: dict mapping feature name → column index in points array.
            Required keys: 'x', 'y', 'rcs', 'v_r', 'v_r_comp'
            Optional keys: 'time' (needed for sweep dropout)
    """

    def __init__(self, config, feature_indices):
        self._cfg = config
        self._fi = feature_indices

        # Validate required feature indices
        for key in ('x', 'y', 'rcs', 'v_r', 'v_r_comp'):
            if key not in self._fi:
                raise KeyError(
                    f"RadarProcessAugmentor requires feature '{key}' "
                    f"in feature_indices, got: {list(self._fi.keys())}"
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, points, rng=None):
        """
        Apply radar process augmentation to a point cloud.

        Args:
            points: (N, D) float32 numpy array of raw radar points.
            rng: optional numpy RandomState for reproducibility.

        Returns:
            augmented: (N', D) float32 numpy array (N' may differ from N
                       due to dropout and ghost injection).
        """
        if rng is None:
            rng = np.random

        cfg = self._cfg
        points = points.copy()

        # --- RCS perturbation ---
        if cfg.get('RCS_SCALE') is not None or cfg.get('RCS_SHIFT') is not None:
            points = self._perturb_rcs(points, rng)

        # --- Range-dependent dropout ---
        rd_cfg = cfg.get('RANGE_DROPOUT', {})
        if isinstance(rd_cfg, dict) and rd_cfg.get('ENABLED', False):
            points = self._range_dropout(points, rng, rd_cfg)
        elif isinstance(rd_cfg, bool) and rd_cfg:
            points = self._range_dropout(points, rng)

        # --- Azimuth noise ---
        if cfg.get('ANGLE_NOISE_STD', 0.0) > 0:
            points = self._azimuth_noise(points, rng)

        # --- Doppler bias ---
        if cfg.get('DOPPLER_BIAS_STD', 0.0) > 0:
            points = self._doppler_bias(points, rng)

        # --- Doppler scale ---
        if cfg.get('DOPPLER_SCALE') is not None:
            points = self._doppler_scale(points, rng)

        # --- Ego-motion compensation noise ---
        if cfg.get('EGO_COMP_NOISE_STD', 0.0) > 0:
            points = self._ego_comp_noise(points, rng)

        # --- Sweep dropout ---
        sweep_prob = cfg.get('SWEEP_DROPOUT_PROB', 0.0)
        if sweep_prob > 0 and 'time' in self._fi:
            points = self._sweep_dropout(points, rng, sweep_prob)

        # --- Ghost injection ---
        ghost_prob = cfg.get('GHOST_PROB', 0.0)
        if ghost_prob > 0:
            points = self._ghost_injection(points, rng, ghost_prob)

        return points

    # ------------------------------------------------------------------
    # Augmentation methods
    # ------------------------------------------------------------------

    def _perturb_rcs(self, points, rng):
        """Apply random scale and shift to RCS values."""
        idx = self._fi['rcs']
        scale_low, scale_high = self._cfg.get('RCS_SCALE', [0.8, 1.2])
        shift_low, shift_high = self._cfg.get('RCS_SHIFT', [-0.5, 0.5])

        scale = rng.uniform(scale_low, scale_high)
        shift = rng.uniform(shift_low, shift_high)
        points[:, idx] = points[:, idx] * scale + shift
        return points

    def _range_dropout(self, points, rng, rd_cfg=None):
        """
        Drop points with probability proportional to range.

        P(drop | r) = base_prob + far_gain * (r / r_max)
        """
        if rd_cfg is None:
            rd_cfg = {}

        base_prob = rd_cfg.get('BASE_PROB', 0.05)
        far_gain = rd_cfg.get('FAR_GAIN', 0.25)

        x_idx = self._fi['x']
        y_idx = self._fi['y']
        r = np.sqrt(points[:, x_idx] ** 2 + points[:, y_idx] ** 2)
        r_max = max(r.max(), 1.0)

        drop_prob = base_prob + far_gain * (r / r_max)
        keep_mask = rng.uniform(size=len(points)) >= drop_prob
        return points[keep_mask]

    def _azimuth_noise(self, points, rng):
        """
        Add angular noise to x, y coordinates.

        Simulates azimuth angular resolution error by applying
        a small random rotation perturbation to each point.
        """
        std = self._cfg.get('ANGLE_NOISE_STD', 0.003)
        x_idx = self._fi['x']
        y_idx = self._fi['y']

        # Per-point small angle perturbation
        delta_theta = rng.normal(0, std, size=len(points))
        cos_d = np.cos(delta_theta)
        sin_d = np.sin(delta_theta)

        x = points[:, x_idx]
        y = points[:, y_idx]
        points[:, x_idx] = x * cos_d - y * sin_d
        points[:, y_idx] = x * sin_d + y * cos_d
        return points

    def _doppler_bias(self, points, rng):
        """Add random bias to radial velocity (both raw and compensated)."""
        std = self._cfg.get('DOPPLER_BIAS_STD', 0.15)

        def _add_bias(idx):
            if idx is not None and idx < points.shape[1]:
                points[:, idx] += rng.normal(0, std, size=len(points))

        _add_bias(self._fi['v_r'])
        _add_bias(self._fi['v_r_comp'])
        return points

    def _doppler_scale(self, points, rng):
        """Apply random multiplicative scale to Doppler velocities."""
        scale_low, scale_high = self._cfg.get('DOPPLER_SCALE', [0.9, 1.1])

        def _scale(idx):
            if idx is not None and idx < points.shape[1]:
                scale = rng.uniform(scale_low, scale_high)
                points[:, idx] *= scale

        _scale(self._fi['v_r'])
        _scale(self._fi['v_r_comp'])
        return points

    def _ego_comp_noise(self, points, rng):
        """
        Add noise to compensated radial velocity.

        Simulates imperfect ego-motion compensation.
        Only perturbs v_r_comp (the compensated velocity), not raw v_r.
        """
        std = self._cfg.get('EGO_COMP_NOISE_STD', 0.10)
        idx = self._fi['v_r_comp']
        if idx is not None and idx < points.shape[1]:
            points[:, idx] += rng.normal(0, std, size=len(points))
        return points

    def _sweep_dropout(self, points, rng, sweep_prob):
        """
        Randomly drop entire historical sweeps (time slices).

        The most recent sweep (max time) is always kept.
        Historical sweeps are dropped independently with sweep_prob.
        """
        time_idx = self._fi['time']
        times = points[:, time_idx]

        # Identify unique sweeps by discretizing time
        sweep_ids = np.unique(times)

        if len(sweep_ids) <= 1:
            return points

        # Keep the most recent sweep (largest time value)
        current_sweep = sweep_ids.max()
        keep_mask = np.ones(len(points), dtype=bool)

        for sweep in sweep_ids:
            if sweep == current_sweep:
                continue  # Always keep current frame
            if rng.random() < sweep_prob:
                # Drop this sweep
                keep_mask[times == sweep] = False

        return points[keep_mask]

    def _ghost_injection(self, points, rng, ghost_prob):
        """
        Inject ghost points near existing points to simulate multipath.

        Each point has ghost_prob chance of generating a ghost neighbor
        with slightly perturbed position and features.
        """
        n = len(points)
        n_ghosts = max(1, int(n * ghost_prob))
        ghost_indices = rng.choice(n, size=n_ghosts, replace=True)

        ghosts = points[ghost_indices].copy()

        # Perturb positions with small noise
        x_idx = self._fi['x']
        y_idx = self._fi['y']
        ghosts[:, x_idx] += rng.normal(0, 0.10, size=n_ghosts)
        ghosts[:, y_idx] += rng.normal(0, 0.10, size=n_ghosts)

        # Slightly attenuate RCS for ghosts
        rcs_idx = self._fi['rcs']
        if rcs_idx < ghosts.shape[1]:
            ghosts[:, rcs_idx] *= rng.uniform(0.3, 0.8, size=n_ghosts)

        # Add Doppler jitter
        for key in ('v_r', 'v_r_comp'):
            idx = self._fi.get(key)
            if idx is not None and idx < ghosts.shape[1]:
                ghosts[:, idx] += rng.normal(0, 0.1, size=n_ghosts)

        return np.concatenate([points, ghosts], axis=0)


def build_radar_augmentor(dataset_cfg):
    """
    Factory: build a RadarProcessAugmentor from a dataset config.

    Returns None if RADAR_AUG is not present or not enabled.
    """
    rad_cfg = dataset_cfg.get('RADAR_AUG', None)
    if rad_cfg is None:
        return None
    if isinstance(rad_cfg, dict) and not rad_cfg.get('ENABLED', True):
        return None

    # Derive feature indices from the dataset's point feature encoding config
    pfe_cfg = dataset_cfg.get('POINT_FEATURE_ENCODING', {})
    used_features = pfe_cfg.get('used_feature_list', ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'])

    feature_indices = {name: idx for idx, name in enumerate(used_features)}

    return RadarProcessAugmentor(config=rad_cfg, feature_indices=feature_indices)
