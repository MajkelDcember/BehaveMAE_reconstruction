"""
Anatomical feature grouping for PoseReconstructionDataset.

Reorders the tail dimensions of the feature vector so that keypoints
belonging to the same anatomical region are contiguous in memory.
This lets hBehaveMAE's patch_kernel capture one anatomical group per patch
WITHOUT changing tensor shape, featurize_keypoints, or inverse_transform.

Usage
-----
    from anatomical_grouping import build_feature_permutation, OFD_MOUSE_GROUPS

    fwd, inv = build_feature_permutation(
        keypoint_names=keypoint_names,
        center_keypoint="neck",
        groups=OFD_MOUSE_GROUPS,
        num_global_dims=4,
    )

    # In dataset:  reordered = features[..., fwd]
    # Before inverse_transform:  original = reordered[..., inv]

Patch kernel
------------
    group_dim = 2 * group_size           # 2 coords per keypoint
    --patch_kernel 1 1 <group_dim>

For the default OFD mouse (27 kpts, neck center, groups of 2):
    group_dim = 4
    --patch_kernel 1 1 4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Group definitions
# ═══════════════════════════════════════════════════════════════════════════════

# Each group is a list of keypoint names (center keypoint must NOT appear).
# All groups MUST have the same number of keypoints.

OFD_MOUSE_GROUPS: List[List[str]] = [
    # ── Head ──────────────────────────────────────────────────────────────
    ["nose",           "head_midpoint"],       # 0  head front
    ["left_ear",       "left_ear_tip"],         # 1  left ear
    ["right_ear",      "right_ear_tip"],        # 2  right ear
    ["left_eye",       "right_eye"],            # 3  eyes
    # ── Spine ─────────────────────────────────────────────────────────────
    ["mid_back",       "mouse_center"],         # 4  upper spine
    ["mid_backend",    "mid_backend2"],          # 5  mid spine
    ["mid_backend3",   "tail_base"],             # 6  lower spine
    # ── Tail ──────────────────────────────────────────────────────────────
    ["tail1",          "tail2"],                 # 7  proximal tail
    ["tail3",          "tail4"],                 # 8  mid tail
    ["tail5",          "tail_end"],              # 9  distal tail
    # ── Lateral ───────────────────────────────────────────────────────────
    ["left_shoulder",  "right_shoulder"],        # 10 shoulders
    ["left_midside",   "right_midside"],         # 11 mid-body sides
    ["left_hip",       "right_hip"],             # 12 hips
]
# 13 groups × 2 keypoints = 26 = 27 − 1 (neck removed)  ✓


# ═══════════════════════════════════════════════════════════════════════════════
# Permutation builder
# ═══════════════════════════════════════════════════════════════════════════════

def _tail_index(kpt_name: str, keypoint_names: Sequence[str], center_idx: int) -> int:
    """
    Return the index of *kpt_name* in the tail feature vector
    (i.e. the keypoint ordering AFTER the center keypoint has been removed
    by ``np.delete(..., center_idx, axis=1)`` inside ``featurize_keypoints``).
    """
    orig_idx = list(keypoint_names).index(kpt_name)
    if orig_idx == center_idx:
        raise ValueError(f"'{kpt_name}' is the center keypoint and must not appear in any group.")
    return orig_idx if orig_idx < center_idx else orig_idx - 1


def build_feature_permutation(
    keypoint_names: Sequence[str],
    center_keypoint: str,
    groups: List[List[str]],
    num_global_dims: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build forward and inverse permutation arrays for the full feature vector.

    Parameters
    ----------
    keypoint_names : sequence of str
        Full ordered keypoint name list (including center).
    center_keypoint : str
        Name of the center keypoint (removed during featurization).
    groups : list of list of str
        Anatomical groups.  Every group must have the same length.
        Every non-center keypoint must appear in exactly one group.
    num_global_dims : int
        Number of leading global dimensions (default 4: cx, cy, sin, cos).

    Returns
    -------
    fwd_perm : np.ndarray, shape (F,)
        ``reordered = features[..., fwd_perm]``
    inv_perm : np.ndarray, shape (F,)
        ``original  = reordered[..., inv_perm]``

    Raises
    ------
    ValueError
        If groups are not equal-sized, miss keypoints, or contain duplicates.
    """
    center_idx = list(keypoint_names).index(center_keypoint)
    K = len(keypoint_names)
    K_tail = K - 1  # after removing center

    # ── Validate groups ───────────────────────────────────────────────────
    group_sizes = {len(g) for g in groups}
    if len(group_sizes) != 1:
        raise ValueError(f"All groups must be equal-sized, got sizes {group_sizes}")
    group_size = group_sizes.pop()

    all_kpts_in_groups = [name for g in groups for name in g]
    if len(all_kpts_in_groups) != len(set(all_kpts_in_groups)):
        from collections import Counter
        dupes = [k for k, v in Counter(all_kpts_in_groups).items() if v > 1]
        raise ValueError(f"Duplicate keypoints in groups: {dupes}")

    expected_tail_kpts = set(keypoint_names) - {center_keypoint}
    actual_tail_kpts = set(all_kpts_in_groups)
    if actual_tail_kpts != expected_tail_kpts:
        missing = expected_tail_kpts - actual_tail_kpts
        extra = actual_tail_kpts - expected_tail_kpts
        raise ValueError(f"Group coverage mismatch.  Missing: {missing}  Extra: {extra}")

    if len(all_kpts_in_groups) != K_tail:
        raise ValueError(
            f"Groups cover {len(all_kpts_in_groups)} keypoints, "
            f"expected {K_tail} (total {K} minus center)."
        )

    # ── Build tail permutation ────────────────────────────────────────────
    # Standard tail ordering: keypoints 0..K-1 with center removed,
    # each occupying 2 consecutive dims (x, y).
    #
    # Grouped ordering: group_0_kpt0_x, group_0_kpt0_y, group_0_kpt1_x, ...

    tail_fwd = []  # new position → old tail dim
    for group in groups:
        for kpt_name in group:
            tail_idx = _tail_index(kpt_name, keypoint_names, center_idx)
            tail_fwd.append(2 * tail_idx)      # x
            tail_fwd.append(2 * tail_idx + 1)  # y

    tail_fwd = np.array(tail_fwd, dtype=np.int64)

    # ── Build full-feature permutation (global dims stay in place) ────────
    F = num_global_dims + 2 * K_tail
    fwd_perm = np.empty(F, dtype=np.int64)
    fwd_perm[:num_global_dims] = np.arange(num_global_dims)
    fwd_perm[num_global_dims:] = tail_fwd + num_global_dims  # shift to full-feature indices

    # ── Inverse permutation ───────────────────────────────────────────────
    inv_perm = np.empty_like(fwd_perm)
    inv_perm[fwd_perm] = np.arange(F)

    return fwd_perm, inv_perm


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience helpers
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnatomicalGroupingInfo:
    """Container returned by ``get_grouping_info`` for easy inspection."""
    groups: List[List[str]]
    group_size: int
    group_dim: int          # = 2 * group_size  (feature dims per group)
    num_groups: int
    num_global_dims: int
    total_F: int
    fwd_perm: np.ndarray
    inv_perm: np.ndarray
    # Mapping: group index → list of (keypoint_name, tail_kpt_index)
    group_details: List[List[Tuple[str, int]]] = field(default_factory=list)


def get_grouping_info(
    keypoint_names: Sequence[str],
    center_keypoint: str,
    groups: List[List[str]],
    num_global_dims: int = 4,
) -> AnatomicalGroupingInfo:
    """Build permutations and return a summary object."""
    fwd, inv = build_feature_permutation(
        keypoint_names, center_keypoint, groups, num_global_dims
    )
    center_idx = list(keypoint_names).index(center_keypoint)
    group_size = len(groups[0])
    group_dim = 2 * group_size

    details = []
    for g in groups:
        details.append([
            (name, _tail_index(name, keypoint_names, center_idx))
            for name in g
        ])

    return AnatomicalGroupingInfo(
        groups=groups,
        group_size=group_size,
        group_dim=group_dim,
        num_groups=len(groups),
        num_global_dims=num_global_dims,
        total_F=len(fwd),
        fwd_perm=fwd,
        inv_perm=inv,
        group_details=details,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    """Verify round-trip and coverage for the default OFD mouse config."""

    ALL_KEYPOINTS = [
        "nose", "left_ear", "right_ear", "left_ear_tip", "right_ear_tip",
        "left_eye", "right_eye", "neck", "mid_back", "mouse_center",
        "mid_backend", "mid_backend2", "mid_backend3", "tail_base",
        "tail1", "tail2", "tail3", "tail4", "tail5",
        "left_shoulder", "left_midside", "left_hip",
        "right_shoulder", "right_midside", "right_hip",
        "tail_end", "head_midpoint",
    ]

    info = get_grouping_info(ALL_KEYPOINTS, "neck", OFD_MOUSE_GROUPS)

    assert info.total_F == 56, f"Expected F=56, got {info.total_F}"
    assert info.group_dim == 4, f"Expected group_dim=4, got {info.group_dim}"
    assert info.num_groups == 13, f"Expected 13 groups, got {info.num_groups}"

    # Round-trip test
    rng = np.random.default_rng(42)
    features = rng.standard_normal((10, 1, 56)).astype(np.float32)

    reordered = features[..., info.fwd_perm]
    restored = reordered[..., info.inv_perm]

    assert np.allclose(features, restored), "Round-trip failed!"

    # Global dims untouched
    assert np.allclose(features[..., :4], reordered[..., :4]), "Global dims moved!"

    # Permutation is a valid permutation
    assert set(info.fwd_perm.tolist()) == set(range(56))
    assert set(info.inv_perm.tolist()) == set(range(56))

    print("✓ All self-tests passed.")
    print(f"  F = {info.total_F}")
    print(f"  {info.num_groups} groups × {info.group_dim} dims = {info.num_groups * info.group_dim} tail dims")
    print(f"  Recommended: --patch_kernel 1 1 {info.group_dim}")
    print()
    for i, (group_names, detail) in enumerate(zip(info.groups, info.group_details)):
        kpt_str = ", ".join(f"{name}(tail_idx={idx})" for name, idx in detail)
        print(f"  Group {i:2d}: {kpt_str}")


if __name__ == "__main__":
    _self_test()