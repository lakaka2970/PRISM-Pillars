"""
Local Candidate Retriever for CRLF (Paper Section 8).

For each current-frame pillar, retrieves Top-K_t history pillars within
a local spatial radius R, using spatial hashing for efficient lookup.

Complexity: O(p * K_t) instead of O(p^2) for global attention.
"""

import torch
import torch.nn as nn


class LocalCandidateRetriever(nn.Module):
    """
    Retrieves local historical pillar candidates for each current pillar.

    Uses grid-based spatial hashing: each pillar's BEV grid coordinate serves
    as its hash key. For a current pillar at (x, y), searches a window of
    (2R+1) x (2R+1) grid cells for history candidates.
    """

    def __init__(self, model_cfg):
        """
        Args:
            model_cfg: EasyDict with LOCAL_RADIUS (int, default 3) and TOPK (int, default 16).
        """
        super().__init__()
        self.local_radius = model_cfg.get('LOCAL_RADIUS', 3)
        self.topk = model_cfg.get('TOPK', 16)

    def forward(self, current_coords, history_coords):
        """
        Args:
            current_coords: (Q, 3) long tensor [batch_id, y_idx, x_idx].
            history_coords: (K, 3) long tensor [batch_id, y_idx, x_idx].

        Returns:
            candidate_indices: (Q, max_candidates) long tensor.
                Each row contains indices of candidate history pillars
                (padding with -1 for fewer than max_candidates).
            candidate_mask: (Q, max_candidates) bool tensor, True for valid.
        """
        Q = current_coords.shape[0]
        K = history_coords.shape[0]
        device = current_coords.device

        if Q == 0 or K == 0:
            return (
                torch.full((Q, self.topk), -1, dtype=torch.long, device=device),
                torch.zeros((Q, self.topk), dtype=torch.bool, device=device),
            )

        # Extract batch and spatial coordinates
        cur_batch = current_coords[:, 0]
        cur_y = current_coords[:, 1]
        cur_x = current_coords[:, 2]

        hist_batch = history_coords[:, 0]
        hist_y = history_coords[:, 1]
        hist_x = history_coords[:, 2]

        R = self.local_radius

        # For efficiency, process batch-by-batch
        max_batch = max(cur_batch.max().item(), hist_batch.max().item()) + 1

        all_candidate_indices = []
        all_candidate_masks = []

        for b in range(max_batch):
            cur_mask = cur_batch == b
            hist_mask = hist_batch == b

            if cur_mask.sum() == 0:
                continue

            cur_idx = torch.where(cur_mask)[0]
            h_idx = torch.where(hist_mask)[0]

            cur_y_b = cur_y[cur_mask]
            cur_x_b = cur_x[cur_mask]
            hist_y_b = hist_y[hist_mask]
            hist_x_b = hist_x[hist_mask]

            n_cur = cur_idx.shape[0]
            n_hist = h_idx.shape[0]

            if n_hist == 0:
                all_candidate_indices.append(
                    torch.full((n_cur, self.topk), -1, dtype=torch.long, device=device)
                )
                all_candidate_masks.append(
                    torch.zeros((n_cur, self.topk), dtype=torch.bool, device=device)
                )
                continue

            # Compute Chebyshev distance (max of abs diff in x and y)
            dy = (cur_y_b.unsqueeze(1) - hist_y_b.unsqueeze(0)).abs()  # (n_cur, n_hist)
            dx = (cur_x_b.unsqueeze(1) - hist_x_b.unsqueeze(0)).abs()  # (n_cur, n_hist)
            dist = torch.max(dy, dx)  # (n_cur, n_hist)

            within_radius = dist <= R

            # For each current pillar, select top-k closest candidates
            # Use distance as sorting key
            sort_dist = dist.clone()
            sort_dist[~within_radius] = float('inf')

            # Get top-k per row
            if n_hist <= self.topk:
                # Fewer history pillars than topk: return all within radius
                for i in range(n_cur):
                    row_valid = within_radius[i]
                    valid_count = row_valid.sum().item()
                    if valid_count == 0:
                        cand = torch.full((self.topk,), -1, dtype=torch.long, device=device)
                        mask = torch.zeros(self.topk, dtype=torch.bool, device=device)
                    else:
                        valid_idx = h_idx[row_valid]
                        cand = torch.full((self.topk,), -1, dtype=torch.long, device=device)
                        mask = torch.zeros(self.topk, dtype=torch.bool, device=device)
                        k_actual = min(valid_count, self.topk)
                        cand[:k_actual] = valid_idx[:k_actual]
                        mask[:k_actual] = True
                    all_candidate_indices.append(cand.unsqueeze(0))
                    all_candidate_masks.append(mask.unsqueeze(0))
            else:
                _, topk_idx = sort_dist.topk(self.topk, dim=-1, largest=False)  # (n_cur, topk)
                # Check which of the topk are within radius
                topk_within = within_radius.gather(1, topk_idx)

                # Map to global indices
                global_topk = h_idx.unsqueeze(0).expand(n_cur, -1).gather(1, topk_idx)
                global_topk[~topk_within] = -1

                all_candidate_indices.append(global_topk)
                all_candidate_masks.append(topk_within)

        if len(all_candidate_indices) == 0:
            return (
                torch.full((Q, self.topk), -1, dtype=torch.long, device=device),
                torch.zeros((Q, self.topk), dtype=torch.bool, device=device),
            )

        candidate_indices = torch.cat(all_candidate_indices, dim=0)  # (Q, topk)
        candidate_mask = torch.cat(all_candidate_masks, dim=0)  # (Q, topk)

        return candidate_indices, candidate_mask
