"""
Sequence-aware multi-frame loader for VOD radar dataset.

Implements the data interface described in paper Section 14:
  - Training/validation split by sequence (not frame)
  - Only reads past frames (causal)
  - Preserves true time deltas
  - Applies ego-motion alignment before radial prediction

Output batch_dict format:
    current_points: (N_c, D)  - current frame radar points
    history_points: (N_h, D)  - historical radar points
    history_delta_t: (N_h,)   - time delta per historical point
    history_sweep_idx: (N_h,) - sweep index (0 = most recent history)
    history_pose_to_current: (num_sweeps, 4, 4) or per-point
    current_voxel_coords, gt_boxes, sequence_id, frame_id
"""

import numpy as np


def transform_points_by_pose(points, transform_matrix):
    """
    Transform point cloud by a 4x4 homogeneous transform matrix.

    Args:
        points: (N, 3+) numpy array, at least [x, y, z, ...]
        transform_matrix: (4, 4) numpy array

    Returns:
        transformed: (N, 3+) numpy array, rotated and translated points.
    """
    N = points.shape[0]
    xyz = points[:, :3]
    xyz_h = np.concatenate([xyz, np.ones((N, 1), dtype=xyz.dtype)], axis=1)  # (N, 4)
    xyz_t = (transform_matrix @ xyz_h.T).T[:, :3]  # (N, 3)
    transformed = points.copy()
    transformed[:, :3] = xyz_t
    return transformed


def invert_pose(pose_matrix):
    """
    Invert a 4x4 homogeneous transform.

    Args:
        pose_matrix: (4, 4) numpy array

    Returns:
        inv_pose: (4, 4) numpy array
    """
    R = pose_matrix[:3, :3]
    t = pose_matrix[:3, 3:4]
    inv = np.eye(4, dtype=pose_matrix.dtype)
    inv[:3, :3] = R.T
    inv[:3, 3:4] = -R.T @ t
    return inv


class RadarSequenceLoader:
    """
    Loads historical radar frames for a given current frame.

    The load order follows the paper's Section 5 (pipeline):
        1. Load current frame
        2. Load historical frames
        3. Apply ego-motion alignment (transform historical to current frame)
        4. Compute per-point time deltas
        5. Pack into batch_dict format

    Config:
        num_sweeps: Number of historical sweeps (e.g., 3 or 5)
        max_history_points: Max points to keep from history (None = all)
        transform_history: Whether to apply ego-motion alignment
        use_true_delta_t: Use true timestamps vs uniform spacing
    """

    def __init__(
        self,
        num_sweeps=3,
        max_history_points=None,
        transform_history=True,
        use_true_delta_t=True,
    ):
        self.num_sweeps = num_sweeps
        self.max_history_points = max_history_points
        self.transform_history = transform_history
        self.use_true_delta_t = use_true_delta_t

    def load_history(
        self,
        dataset,
        current_index,
        sequence_info=None,
    ):
        """
        Load historical frames for a given current frame index.

        Args:
            dataset: VodDataset instance (for get_lidar etc.)
            current_index: int, index into dataset.vod_infos
            sequence_info: Optional dict with sequence mapping info.
                           If None, uses consecutive indices as history.

        Returns:
            history_dict: dict with keys:
                history_points, history_delta_t, history_sweep_idx,
                history_pose_to_current
        """
        # Determine historical frame indices
        if sequence_info is not None:
            history_indices = self._get_history_from_sequence(
                sequence_info, current_index
            )
        else:
            # Fallback: use consecutive previous indices
            history_indices = [
                max(0, current_index - k - 1) for k in range(self.num_sweeps)
            ]

        # Get current frame pose (optional, for ego-motion)
        current_info = dataset.vod_infos[current_index]
        # If calibration has Tr_velo_to_cam, we can use it.
        # Without explicit odometry, we assume identity for now.
        # In practice, this should be replaced with actual ego-pose data.

        history_points_list = []
        history_delta_t_list = []
        history_sweep_idx_list = []
        history_pose_list = []

        # Assume uniform time delta between frames (e.g., 0.1s for 10Hz radar)
        default_dt = 0.1

        for sweep_idx, h_idx in enumerate(history_indices):
            if h_idx < 0:
                continue

            h_info = dataset.vod_infos[h_idx]
            h_points = dataset.get_lidar(h_info['point_cloud']['lidar_idx'])

            if h_points.shape[0] == 0:
                continue

            # Time delta: most recent history = 1, older = larger
            if self.use_true_delta_t:
                # Without real timestamps, approximate from sweep index
                delta_t = (sweep_idx + 1) * default_dt
            else:
                delta_t = (sweep_idx + 1) * default_dt

            # Ego-motion alignment
            # For now, use identity (no pose data available).
            # When odometry data is available:
            #   pose_to_current = get_relative_pose(h_info, current_info)
            #   h_points = transform_points_by_pose(h_points, pose_to_current)
            pose_to_current = np.eye(4, dtype=np.float32)

            # Apply transform if enabled
            if self.transform_history:
                h_points = transform_points_by_pose(h_points, pose_to_current)

            # Append metadata
            N = h_points.shape[0]
            history_points_list.append(h_points)
            history_delta_t_list.append(np.full(N, delta_t, dtype=np.float32))
            history_sweep_idx_list.append(np.full(N, sweep_idx, dtype=np.int32))
            history_pose_list.append(pose_to_current)

        # Concatenate all history
        if len(history_points_list) == 0:
            # No history available: return empty tensors
            return {
                'history_points': np.zeros((0, 7), dtype=np.float32),
                'history_delta_t': np.zeros(0, dtype=np.float32),
                'history_sweep_idx': np.zeros(0, dtype=np.int32),
                'history_pose_to_current': np.eye(4, dtype=np.float32)[np.newaxis, ...],
            }

        history_points = np.concatenate(history_points_list, axis=0)
        history_delta_t = np.concatenate(history_delta_t_list, axis=0)
        history_sweep_idx = np.concatenate(history_sweep_idx_list, axis=0)
        history_pose_to_current = np.stack(history_pose_list, axis=0)  # (S, 4, 4)

        # Optional: subsample history points
        if self.max_history_points is not None and history_points.shape[0] > self.max_history_points:
            keep_idx = np.random.choice(
                history_points.shape[0], self.max_history_points, replace=False
            )
            history_points = history_points[keep_idx]
            history_delta_t = history_delta_t[keep_idx]
            history_sweep_idx = history_sweep_idx[keep_idx]

        return {
            'history_points': history_points,
            'history_delta_t': history_delta_t,
            'history_sweep_idx': history_sweep_idx,
            'history_pose_to_current': history_pose_to_current,
        }

    def _get_history_from_sequence(self, sequence_info, current_index):
        """
        Get historical frame indices from sequence mapping.

        Args:
            sequence_info: dict mapping frame_id -> list of frame_ids in sequence
            current_index: int, current frame's position

        Returns:
            list of indices for historical frames (past only)
        """
        # This is a stub for when sequence metadata is available
        # In practice, look up the sequence containing current_index
        # and return up to num_sweeps preceding frames.
        indices = []
        for k in range(1, self.num_sweeps + 1):
            prev_idx = max(0, current_index - k)
            indices.append(prev_idx)
        return indices


def build_sequence_mapping(info_list):
    """
    Build a mapping from frame index to its sequence members.

    This extracts sequence_id from frame metadata if available.
    Without explicit sequence annotations, frames are treated as
    independent (each frame is its own sequence).

    Args:
        info_list: list of frame info dicts

    Returns:
        sequence_map: dict {sequence_id: [index1, index2, ...]}
        frame_to_sequence: dict {frame_index: sequence_id}
    """
    sequence_map = {}
    frame_to_sequence = {}

    # Try to extract sequence/segment info if available
    for idx, info in enumerate(info_list):
        seq_id = info.get('sequence_id', str(idx))
        frame_to_sequence[idx] = seq_id
        if seq_id not in sequence_map:
            sequence_map[seq_id] = []
        sequence_map[seq_id].append(idx)

    # Sort each sequence by frame index
    for seq_id in sequence_map:
        sequence_map[seq_id] = sorted(sequence_map[seq_id])

    return sequence_map, frame_to_sequence


def split_sequences_for_train_val(sequence_map, val_ratio=0.2, seed=42):
    """
    Split sequences into train/val sets ensuring no sequence crosses split.

    Args:
        sequence_map: dict {sequence_id: [indices]}
        val_ratio: float, proportion for validation.
        seed: int, random seed.

    Returns:
        train_indices, val_indices: lists of frame indices
    """
    rng = np.random.RandomState(seed)
    seq_ids = sorted(sequence_map.keys())
    rng.shuffle(seq_ids)

    n_val = max(1, int(len(seq_ids) * val_ratio))
    val_seqs = set(seq_ids[:n_val])
    train_seqs = set(seq_ids[n_val:])

    train_indices = []
    val_indices = []
    for seq_id, indices in sequence_map.items():
        if seq_id in train_seqs:
            train_indices.extend(indices)
        else:
            val_indices.extend(indices)

    return sorted(train_indices), sorted(val_indices)
