# Shared pose geometry utilities.
# Extracted from PoseReconstructionDataset so that BOTH training (via the
# Dataset class) and standalone inference scripts use identical math.
#
# RULE: every function here is pure (stateless).  Configuration is passed
#       via a PoseGeometryConfig dataclass.

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PoseGeometryConfig:
    """
    Immutable bundle of every setting that the geometry functions need.
    Construct once, share everywhere.
    """
    keypoint_names: Tuple[str, ...]
    center_keypoint: str
    align_keypoints: Tuple[str, str]
    scale_keypoints: Optional[Tuple[str, str]]   # None → no scaling
    grid_size: float
    centeralign: bool
    num_individuals: int = 1

    # --- derived (computed automatically) ---
    num_keypoints: int = field(init=False)
    keypoint_name_to_idx: Dict[str, int] = field(init=False, repr=False)
    feature_dim: int = field(init=False)
    tail_dim: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self, "num_keypoints", len(self.keypoint_names)
        )
        object.__setattr__(
            self,
            "keypoint_name_to_idx",
            {name: i for i, name in enumerate(self.keypoint_names)},
        )
        if self.centeralign:
            # [center_x, center_y, sin, cos, kpt0_x, kpt0_y, …]
            object.__setattr__(
                self, "feature_dim", 4 + 2 * (self.num_keypoints - 1)
            )
            object.__setattr__(
                self, "tail_dim", 2 * (self.num_keypoints - 1)
            )
        else:
            object.__setattr__(
                self, "feature_dim", self.num_keypoints * 2
            )
            object.__setattr__(self, "tail_dim", self.num_keypoints * 2)


# ═══════════════════════════════════════════════════════════════════════════
# Keypoint restriction helpers
# ═══════════════════════════════════════════════════════════════════════════

def restrict_keypoints(
    keypoints: np.ndarray,
    keypoint_indices: List[int],
) -> np.ndarray:
    """(T, I, K_all, 2) → (T, I, K_sel, 2)"""
    return keypoints[:, :, keypoint_indices, :]


def restrict_confidences(
    confidences: np.ndarray,
    keypoint_indices: List[int],
) -> np.ndarray:
    """(T, I, K_all) → (T, I, K_sel)"""
    return confidences[:, :, keypoint_indices]


# ═══════════════════════════════════════════════════════════════════════════
# Reliability / confidence logic
# ═══════════════════════════════════════════════════════════════════════════

def zero_out_confidences(
    keypoints: np.ndarray,                   # (T, I, K, 2)
    confidences: np.ndarray,                 # (T, I, K)
    nose_idx: Optional[int] = None,
    tail_base_idx: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single place where frame unreliability is decided.
    Returns (keypoints_unchanged, confidences_zeroed).
    """
    FLIP_ANGLE_THRESHOLD = np.deg2rad(90.0)
    DOMINATION_WINDOW_SIZE = 25
    DOMINATION_THRESHOLD = 0.5
    BAD_CONF_THRESHOLD = 0.01

    T = keypoints.shape[0]
    K = keypoints.shape[2]
    confidences = confidences.copy()

    # 1. NaN → zero confidence
    nan_per_frame = np.any(np.isnan(keypoints), axis=(1, 2, 3))
    confidences[nan_per_frame] = 0.0

    # 2. Flip detection
    if (
        nose_idx is not None
        and tail_base_idx is not None
        and 0 <= nose_idx < K
        and 0 <= tail_base_idx < K
    ):
        nose = keypoints[:, 0, nose_idx, :]
        tail_base = keypoints[:, 0, tail_base_idx, :]
        direction = nose - tail_base

        angles = np.arctan2(direction[:, 1], direction[:, 0])
        angle_diff = np.diff(angles)
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        flip_detected = np.abs(angle_diff) > FLIP_ANGLE_THRESHOLD
        for i in np.where(flip_detected)[0]:
            confidences[i] = 0.0
            confidences[i + 1] = 0.0

    # 3. Temporal domination
    conf_flat = confidences.reshape(T, -1)
    initial_bad = np.any(conf_flat <= BAD_CONF_THRESHOLD, axis=1)

    half_window = DOMINATION_WINDOW_SIZE // 2
    to_zero = np.zeros(T, dtype=bool)

    for t in range(T):
        if initial_bad[t]:
            continue
        start = max(0, t - half_window)
        end = min(T, t + half_window + 1)
        bad_ratio = np.mean(initial_bad[start:end])
        if bad_ratio > DOMINATION_THRESHOLD:
            to_zero[t] = True

    confidences[to_zero] = 0.0
    return keypoints, confidences


# ═══════════════════════════════════════════════════════════════════════════
# Hole filling
# ═══════════════════════════════════════════════════════════════════════════

def fill_holes(vec_seq: np.ndarray) -> np.ndarray:
    """
    Fill NaN values using linear interpolation.

    Args:
        vec_seq: (T, I, K, 2)
    Returns:
        Copy with NaNs filled.
    """
    if not np.any(np.isnan(vec_seq)):
        return vec_seq

    result = vec_seq.copy()
    T = result.shape[0]
    x_indices = np.arange(T)

    for ind in range(result.shape[1]):
        for kpt in range(result.shape[2]):
            for dim in range(result.shape[3]):
                data = result[:, ind, kpt, dim]
                mask = np.isnan(data)

                if not np.any(mask):
                    continue
                if np.all(mask):
                    result[:, ind, kpt, dim] = 0
                    continue

                valid_mask = ~mask
                result[:, ind, kpt, dim] = np.interp(
                    x_indices,
                    x_indices[valid_mask],
                    data[valid_mask],
                )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Scale computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_sequence_scale(
    keypoints: np.ndarray,           # (T, I, K, 2)  – already restricted
    cfg: PoseGeometryConfig,
) -> float:
    """Median distance between ``scale_keypoints``.  Returns 1.0 if disabled."""
    if cfg.scale_keypoints is None:
        return 1.0

    start_idx = cfg.keypoint_name_to_idx[cfg.scale_keypoints[0]]
    end_idx = cfg.keypoint_name_to_idx[cfg.scale_keypoints[1]]

    start_pts = keypoints[:, 0, start_idx, :]
    end_pts = keypoints[:, 0, end_idx, :]

    distances = np.linalg.norm(start_pts - end_pts, axis=1)
    valid = distances[~np.isnan(distances)]
    if len(valid) == 0:
        return 1.0
    scale = float(np.median(valid))
    return scale if scale > 0 else 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Center + rotation alignment
# ═══════════════════════════════════════════════════════════════════════════

def transform_to_centered_data(
    data: np.ndarray,                # (T, I, K, 2)
    cfg: PoseGeometryConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Center on ``center_keypoint``, rotate so ``align_keypoints`` vector
    points along +x.

    Returns:
        center      : (T, 2) pixel positions
        rotation    : (T, 2) encoded as (sin(θ), cos(θ)) with θ = -angle
        centered_flat: (T, K*2) rotated keypoints (all K, including center)
    """
    center_idx = cfg.keypoint_name_to_idx[cfg.center_keypoint]
    align_start_idx = cfg.keypoint_name_to_idx[cfg.align_keypoints[0]]
    align_end_idx = cfg.keypoint_name_to_idx[cfg.align_keypoints[1]]

    center = data[:, 0, center_idx, :]                          # (T, 2)
    centered = data - center[:, np.newaxis, np.newaxis, :]      # (T, I, K, 2)

    align_vec = (
        centered[:, 0, align_end_idx, :]
        - centered[:, 0, align_start_idx, :]
    )  # (T, 2)
    angles = np.arctan2(align_vec[:, 1], align_vec[:, 0])       # (T,)

    theta = -angles
    c = np.cos(theta)
    s = np.sin(theta)

    x = centered[..., 0]                                        # (T, I, K)
    y = centered[..., 1]

    x2 = c[:, None, None] * x - s[:, None, None] * y
    y2 = s[:, None, None] * x + c[:, None, None] * y

    rotated = np.stack([x2, y2], axis=-1)                       # (T, I, K, 2)
    rotated_flat = rotated[:, 0].reshape(rotated.shape[0], -1)  # (T, K*2)
    rotation = np.stack([s, c], axis=1)                         # (T, 2)

    return center, rotation, rotated_flat


# ═══════════════════════════════════════════════════════════════════════════
# Featurization  (keypoints → model features)
# ═══════════════════════════════════════════════════════════════════════════

def featurize_keypoints(
    keypoints: np.ndarray,           # (T, I*K*2) flat
    num_frames: int,
    cfg: PoseGeometryConfig,
) -> np.ndarray:
    """
    Convert flattened keypoints to the feature vector the model was trained
    on.

    Returns:
        np.ndarray of shape (T, 1, F)  where F = ``cfg.feature_dim``
    """
    if not cfg.centeralign:
        features = keypoints.reshape(
            num_frames, cfg.num_individuals, cfg.num_keypoints * 2
        )
        return features.astype(np.float32)

    keypoints_reshaped = keypoints.reshape(
        num_frames, cfg.num_individuals, cfg.num_keypoints, 2
    )
    center_idx = cfg.keypoint_name_to_idx[cfg.center_keypoint]

    center, rotation, centered_kpts = transform_to_centered_data(
        keypoints_reshaped, cfg
    )

    # Remove center keypoint (its contribution is already in ``center``)
    centered_kpts = centered_kpts.reshape(num_frames, cfg.num_keypoints, 2)
    centered_kpts = np.delete(centered_kpts, center_idx, axis=1)
    centered_kpts = centered_kpts.reshape(num_frames, -1)

    arena_half = cfg.grid_size / 2.0
    center = (center - arena_half) / arena_half

    features = np.concatenate([center, rotation, centered_kpts], axis=1)
    features = features[:, np.newaxis, :]                       # (T, 1, F)
    return features.astype(np.float32)


def featurize_full_sequence(
    kpts: np.ndarray,                # (T, I, K, 2) – filled, restricted
    cfg: PoseGeometryConfig,
) -> np.ndarray:
    """
    Same geometry as ``featurize_keypoints`` but for an *arbitrary-length*
    sequence (no ``num_frames`` constraint, no padding).

    Returns:
        (T, F) — no singleton individuals axis
    """
    T = kpts.shape[0]
    flat = kpts.reshape(T, -1)
    feat_3d = featurize_keypoints(flat, T, cfg)   # (T, 1, F)
    return feat_3d[:, 0, :]                       # (T, F)


# ═══════════════════════════════════════════════════════════════════════════
# Likelihood / confidence alignment with feature layout
# ═══════════════════════════════════════════════════════════════════════════

def process_likelihoods(
    confidences: np.ndarray,         # (T, I, K)
    cfg: PoseGeometryConfig,
) -> np.ndarray:
    """
    Align per-keypoint confidences with the feature vector layout.

    Returns:
        np.ndarray of shape (T, I, F)
    """
    if cfg.centeralign:
        center_idx = cfg.keypoint_name_to_idx[cfg.center_keypoint]
        a_idx = cfg.keypoint_name_to_idx[cfg.align_keypoints[0]]
        b_idx = cfg.keypoint_name_to_idx[cfg.align_keypoints[1]]

        center_conf = confidences[:, :, center_idx : center_idx + 1]
        center_weights = np.repeat(center_conf, 2, axis=2)

        rot_conf = np.minimum(
            confidences[:, :, a_idx : a_idx + 1],
            confidences[:, :, b_idx : b_idx + 1],
        )
        rotation_weights = np.repeat(rot_conf, 2, axis=2)

        kpt_conf = np.delete(confidences, center_idx, axis=2)
        kpt_weights = np.repeat(kpt_conf, 2, axis=2)

        weights = np.concatenate(
            [center_weights, rotation_weights, kpt_weights], axis=2
        )
    else:
        weights = np.repeat(confidences, 2, axis=2)

    return weights.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Inverse transform  (features → pixel keypoints)
# ═══════════════════════════════════════════════════════════════════════════

def inverse_transform(
    features: np.ndarray,            # (..., F)
    cfg: PoseGeometryConfig,
) -> np.ndarray:
    """
    Pure inverse of ``featurize_keypoints`` / ``featurize_full_sequence``.
    Assumes features are already in the correct (scaled) space.

    Returns:
        (..., K, 2)
    """
    if not cfg.centeralign:
        return features.reshape(*features.shape[:-1], cfg.num_keypoints, 2)

    original_shape = features.shape[:-1]
    feat = features.reshape(-1, features.shape[-1])          # (N, F)

    arena_half = float(cfg.grid_size) / 2.0

    center_norm = feat[:, 0:2]
    sin_m = feat[:, 2]
    cos_m = feat[:, 3]
    rot_flat = feat[:, 4:]

    center = center_norm * arena_half + arena_half           # (N, 2)

    R_inv = np.stack(
        [
            np.stack([cos_m, sin_m], axis=1),
            np.stack([-sin_m, cos_m], axis=1),
        ],
        axis=1,
    )  # (N, 2, 2)

    K_other = cfg.num_keypoints - 1
    rot = rot_flat.reshape(-1, K_other, 2)                   # (N, K-1, 2)

    unrot = (R_inv @ rot.transpose(0, 2, 1)).transpose(0, 2, 1)
    unrot = unrot + center[:, None, :]

    full = np.zeros((unrot.shape[0], cfg.num_keypoints, 2), dtype=feat.dtype)
    cidx = cfg.keypoint_name_to_idx[cfg.center_keypoint]

    full[:, cidx, :] = center
    mask = np.ones(cfg.num_keypoints, dtype=bool)
    mask[cidx] = False
    full[:, mask, :] = unrot

    return full.reshape(original_shape + (cfg.num_keypoints, 2))




# ═══════════════════════════════════════════════════════════════════════════
# Inference-only motion diagnostics & frame-level invalidation
# ═══════════════════════════════════════════════════════════════════════════
#
# NOTE:
# - Pure NumPy, deterministic, stateless
# - NOT used during training
# - Does NOT modify geometry or features
# - Frame-level semantics: any keypoint violation → whole frame invalid

def valid_frame_mask(likelihood: np.ndarray) -> np.ndarray:
    """
    A frame is valid if ALL keypoints have likelihood >= 0.

    Args:
        likelihood: (T, K)
    Returns:
        (T,) boolean
    """
    return np.all(likelihood >= 0, axis=1)


def valid_interval_mask(frame_mask: np.ndarray) -> np.ndarray:
    """Interval t→t+1 valid iff both frames valid.  Returns (T-1,)."""
    return frame_mask[:-1] & frame_mask[1:]


def valid_accel_mask(frame_mask: np.ndarray) -> np.ndarray:
    """Triplet t→t+1→t+2 valid iff all three frames valid.  Returns (T-2,)."""
    return frame_mask[:-2] & frame_mask[1:-1] & frame_mask[2:]


def displacement(keypoints: np.ndarray) -> np.ndarray:
    """
    Frame-to-frame displacement per keypoint.

    Args:
        keypoints: (T, K, 2)
    Returns:
        (T-1, K)
    """
    return np.linalg.norm(np.diff(keypoints, axis=0), axis=2)


def num_jumps_per_keypoint(
    keypoints: np.ndarray,
    likelihood: np.ndarray,
    jump_threshold: float = 20.0,
) -> int:
    """
    Count intervals where ANY keypoint displacement exceeds threshold
    (among valid intervals only).
    """
    interval_mask = valid_interval_mask(valid_frame_mask(likelihood))
    disp = displacement(keypoints)
    jumps_any = np.any(disp > jump_threshold, axis=1)
    return int(np.sum(jumps_any & interval_mask))


def acceleration_metrics(
    keypoints: np.ndarray,
    likelihood: np.ndarray,
    accel_threshold: float = 5.0,
) -> dict:
    """Acceleration stats over valid triplets.  Frame-level semantics."""
    amask = valid_accel_mask(valid_frame_mask(likelihood))
    if amask.sum() == 0:
        return dict(mean=np.nan, std=np.nan, max=np.nan, excessive_fraction=np.nan)

    a = np.diff(np.diff(keypoints, axis=0), axis=0)   # (T-2, K, 2)
    accel = np.linalg.norm(a, axis=2)                  # (T-2, K)
    accel_any = np.any(accel > accel_threshold, axis=1)
    valid_accel = accel[amask]
    return {
        "mean": float(valid_accel.mean()),
        "std": float(valid_accel.std()),
        "max": float(valid_accel.max()),
        "excessive_fraction": float(accel_any[amask].mean()),
    }


def mpjve_metrics(
    keypoints: np.ndarray,
    likelihood: np.ndarray,
    velocity_threshold: float = 50.0,
) -> dict:
    """Mean Per-Joint Velocity Error (MPJVE).  Frame-level semantics."""
    imask = valid_interval_mask(valid_frame_mask(likelihood))
    disp = displacement(keypoints)
    disp_any = np.any(disp > velocity_threshold, axis=1)
    valid_disp = disp[imask]
    if valid_disp.size == 0:
        return dict(mean=np.nan, std=np.nan, max=np.nan, excessive_fraction=np.nan)
    return {
        "mean": float(valid_disp.mean()),
        "std": float(valid_disp.std()),
        "max": float(valid_disp.max()),
        "excessive_fraction": float(disp_any[imask].mean()),
    }


def invalidate_frames_by_motion(
    keypoints: np.ndarray,
    likelihood: np.ndarray,
    jump_threshold: float = 20.0,
    velocity_threshold: float = 50.0,
    accel_threshold: float = 5.0,
) -> np.ndarray:
    """
    Frame-level motion-based invalidation.

    Args:
        keypoints: (T, K, 2)  — filled (no NaNs)
        likelihood: (T, K)    — per-keypoint confidence
        jump_threshold: displacement threshold (px)
        velocity_threshold: velocity threshold (px/frame)
        accel_threshold: acceleration threshold (px/frame²)

    Returns:
        (T,) boolean — True = reliable, False = must be hard-masked
    """
    T = keypoints.shape[0]
    invalid = ~valid_frame_mask(likelihood)

    # Jump / velocity (interval-based → invalidate both endpoints)
    disp = displacement(keypoints)                     # (T-1, K)
    interval_bad = np.any(disp > jump_threshold, axis=1) | np.any(
        disp > velocity_threshold, axis=1
    )
    for i in np.where(interval_bad)[0]:
        invalid[i] = True
        invalid[i + 1] = True

    # Acceleration (triplet-based → invalidate all three)
    a = np.diff(np.diff(keypoints, axis=0), axis=0)   # (T-2, K, 2)
    accel_bad = np.any(np.linalg.norm(a, axis=2) > accel_threshold, axis=1)
    for i in np.where(accel_bad)[0]:
        invalid[i] = True
        invalid[i + 1] = True
        invalid[i + 2] = True

    return ~invalid