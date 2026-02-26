"""
Minimal Keypoint-MoSeq Dataset for hBehaveMAE.

This dataset loader is designed for CLEAN, PREPROCESSED Keypoint-MoSeq outputs.
It performs exactly four operations:
    1. Load keypoints and confidence scores from pickle files
    2. Interpolate NaNs (linear per keypoint/dimension)
    3. Scale per-video using median(neck ↔ tail_base) distance → 25.0
    4. Featurize identically to the existing hBehaveMAE pipeline

Shape semantics:
    Input pickle:  tuple(keypoints_dict, confidences_dict, keypoint_names_list)
                   keypoints_dict[video_name] → (T, K, 2)
                   confidences_dict[video_name] → (T, K)
    Output:        (T, 1, F) features, (T, 1, F) likelihoods

Feature layout (when centeralign=True):
    [center_x, center_y, sin(theta), cos(theta), kpt1_x, kpt1_y, ...]
    where center_keypoint is removed from the tail keypoints.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data

from .reconstruct_data2_utils import (
    PoseGeometryConfig,
    fill_holes,
    featurize_keypoints,
    process_likelihoods,
    inverse_transform as _base_inverse_transform,
)


class KeypointMoSeqDataset(torch.utils.data.Dataset):
    """
    Dataset for Keypoint-MoSeq outputs with real confidence scores.

    Expects pickle format: tuple(keypoints_dict, confidences_dict, keypoint_names_list)
        - keypoints_dict[video_name] → (T, K, 2) array
        - confidences_dict[video_name] → (T, K) array with values in [0, 1]
        - keypoint_names_list → list of K keypoint names

    Confidence handling:
        - Real confidence scores are preserved (e.g., 0.85, 0.92)
        - Frames with NaN keypoints have confidence forced to 0.0
        - These scores can be used to weight loss functions

    Args:
        data_path: Path to pickle file containing keypoint data
        keypoint_names: List of keypoint names to extract (subset of source)
        source_keypoint_names: Optional override for source keypoint names
        num_frames: Temporal window length (default: 80)
        sliding_window: Step size for windowing (default: 1)
        grid_size: Arena size for center normalization (default: 500)
        scale_target: Target median distance for scaling (default: 25.0)
        return_likelihoods: Whether to return likelihood weights (default: True)
    """

    # Fixed geometry for hBehaveMAE compatibility
    CENTER_KEYPOINT = "mouse_center"
    ALIGN_KEYPOINTS = ("neck", "tail_base")
    SCALE_KEYPOINTS = ("neck", "tail_base")

    def __init__(
        self,
        data_path: Union[str, Path],
        keypoint_names: List[str],
        source_keypoint_names: Optional[List[str]] = None,
        num_frames: int = 80,
        sliding_window: int = 1,
        grid_size: float = 500.0,
        scale_target: float = 25.0,
        return_likelihoods: bool = True,
    ):
        self.data_path = Path(data_path)
        self.num_frames = num_frames
        self.sliding_window = sliding_window
        self.grid_size = grid_size
        self.scale_target = scale_target
        self.return_likelihoods = return_likelihoods

        self.keypoint_names = keypoint_names
        self._source_keypoint_names = source_keypoint_names
        self._keypoint_indices: Optional[List[int]] = None

        # Build geometry config
        self.geo_cfg = PoseGeometryConfig(
            keypoint_names=tuple(self.keypoint_names),
            center_keypoint=self.CENTER_KEYPOINT,
            align_keypoints=self.ALIGN_KEYPOINTS,
            scale_keypoints=self.SCALE_KEYPOINTS,
            grid_size=grid_size,
            centeralign=True,
            num_individuals=1,
        )

        # Storage
        self.video_keypoints: List[np.ndarray] = []      # (T_i, 1, K, 2) per video
        self.video_confidences: List[np.ndarray] = []    # (T_i, 1, K) per video - REAL scores
        self.video_scales: List[float] = []
        self.window_indices: List[Tuple[int, int]] = []

        self._load_and_preprocess()

        print(
            f"[KeypointMoSeqDataset] Loaded {len(self.video_keypoints)} videos, "
            f"{len(self.window_indices)} windows, "
            f"{len(self.keypoint_names)} keypoints, "
            f"feature_dim={self.geo_cfg.feature_dim}"
        )

    def _load_and_preprocess(self) -> None:
        """Load pickle, interpolate NaNs, preserve real confidences."""
        
        with open(self.data_path, "rb") as f:
            raw_data = pickle.load(f)

        # Unpack the Keypoint-MoSeq tuple format
        if not isinstance(raw_data, tuple) or len(raw_data) < 3:
            raise ValueError(
                f"Expected tuple of (keypoints_dict, confidences_dict, keypoint_names), "
                f"got {type(raw_data)} with length {len(raw_data) if isinstance(raw_data, tuple) else 'N/A'}"
            )
        
        keypoints_dict = raw_data[0]
        confidences_dict = raw_data[1]
        source_keypoint_names = list(raw_data[2])

        # Set source keypoint names
        if self._source_keypoint_names is None:
            self._source_keypoint_names = source_keypoint_names

        # Build keypoint index mapping
        self._keypoint_indices = self._build_keypoint_indices()

        for video_name in keypoints_dict.keys():
            keypoints = keypoints_dict[video_name]  # (T, K_all, 2)
            confidences = confidences_dict[video_name]  # (T, K_all)

            # Ensure 4D: (T, K, 2) → (T, 1, K, 2)
            keypoints = self._ensure_shape(keypoints)
            
            # Ensure 3D for confidences: (T, K) → (T, 1, K)
            if confidences.ndim == 2:
                confidences = confidences[:, np.newaxis, :]

            # Restrict to selected keypoints
            keypoints = self._restrict_keypoints(keypoints)  # (T, 1, K_target, 2)
            confidences = confidences[:, :, self._keypoint_indices]  # (T, 1, K_target)

            # Identify NaN locations BEFORE interpolation
            nan_mask = np.any(np.isnan(keypoints), axis=-1)  # (T, 1, K)

            # Force confidence to 0.0 where keypoints were NaN
            confidences = confidences.copy()  # Don't modify original
            confidences[nan_mask] = 0.0

            # Interpolate NaNs in keypoints
            keypoints_filled = fill_holes(keypoints)

            # Compute per-video scale
            scale = self._compute_scale(keypoints_filled)

            # Store
            self.video_keypoints.append(keypoints_filled.astype(np.float32))
            self.video_confidences.append(confidences.astype(np.float32))
            self.video_scales.append(scale)

            # Build window indices
            T = keypoints_filled.shape[0]
            video_idx = len(self.video_keypoints) - 1
            for start in range(0, T - self.num_frames + 1, self.sliding_window):
                self.window_indices.append((video_idx, start))

    def _build_keypoint_indices(self) -> List[int]:
        """Build semantic index mapping from source to target keypoints."""
        source_names = self._source_keypoint_names
        target_names = self.keypoint_names
        
        indices = []
        missing = []
        
        for target_name in target_names:
            try:
                idx = source_names.index(target_name)
                indices.append(idx)
            except ValueError:
                missing.append(target_name)
        
        if missing:
            raise ValueError(
                f"Required keypoints not found in source data: {missing}\n"
                f"Source keypoints: {source_names}\n"
                f"Target keypoints: {target_names}"
            )
        
        return indices

    def _ensure_shape(self, keypoints: np.ndarray) -> np.ndarray:
        """Ensure keypoints have shape (T, I, K, 2) with I=1."""
        if keypoints.ndim == 3:
            return keypoints[:, np.newaxis, :, :]
        elif keypoints.ndim == 4:
            return keypoints
        else:
            raise ValueError(f"Expected 3D or 4D keypoints, got shape {keypoints.shape}")

    def _restrict_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """Select configured bodyparts using semantic index mapping."""
        return keypoints[:, :, self._keypoint_indices, :]

    def _compute_scale(self, keypoints: np.ndarray) -> float:
        """Compute per-video scale factor based on neck↔tail_base distance."""
        neck_idx = self.geo_cfg.keypoint_name_to_idx["neck"]
        tail_idx = self.geo_cfg.keypoint_name_to_idx["tail_base"]

        neck_pts = keypoints[:, 0, neck_idx, :]
        tail_pts = keypoints[:, 0, tail_idx, :]

        distances = np.linalg.norm(neck_pts - tail_pts, axis=1)
        valid_distances = distances[~np.isnan(distances)]

        if len(valid_distances) == 0:
            return 1.0

        median_dist = float(np.median(valid_distances))
        return self.scale_target / median_dist if median_dist > 0 else 1.0

    def _featurize(self, keypoints: np.ndarray, scale: float) -> np.ndarray:
        """Convert keypoints to model features."""
        T = keypoints.shape[0]
        flat = keypoints.reshape(T, -1)
        features = featurize_keypoints(flat, T, self.geo_cfg)
        features[:, :, 4:] *= scale
        return features

    def _build_likelihoods(self, confidences: np.ndarray) -> np.ndarray:
        """
        Build likelihood weights from real confidence scores.
        
        Args:
            confidences: (T, 1, K) real confidence scores with NaNs already zeroed
            
        Returns:
            (T, 1, F) likelihoods aligned with feature layout
        """
        return process_likelihoods(confidences, self.geo_cfg)

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, idx: int):
        """Get a single sample with features and real confidence-based likelihoods."""
        video_idx, start = self.window_indices[idx]
        end = start + self.num_frames

        keypoints = self.video_keypoints[video_idx][start:end]
        confidences = self.video_confidences[video_idx][start:end]
        scale = self.video_scales[video_idx]

        features = self._featurize(keypoints, scale)
        features = torch.from_numpy(features).float()

        if self.return_likelihoods:
            likelihoods = self._build_likelihoods(confidences)
            likelihoods = torch.from_numpy(likelihoods).float()
            combined = torch.cat([features, likelihoods], dim=-1)
            return combined, []

        return features, []

    # Utility methods
    def inverse_transform(
        self,
        features: Union[np.ndarray, torch.Tensor],
        video_idx: Optional[int] = None,
        scale: Optional[float] = None,
    ) -> np.ndarray:
        """Inverse-transform features back to keypoint coordinates."""
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        
        features = features.copy()
        
        unscale_factor = scale if scale is not None else (
            self.video_scales[video_idx] if video_idx is not None else None
        )
        
        if unscale_factor is not None and unscale_factor != 0:
            features[..., 4:] /= unscale_factor
        
        return _base_inverse_transform(features, self.geo_cfg)

    def get_scale(self, video_idx: int) -> float:
        return self.video_scales[video_idx]

    def get_video_idx_for_sample(self, sample_idx: int) -> int:
        return self.window_indices[sample_idx][0]

    @property
    def source_keypoint_names(self) -> List[str]:
        return list(self._source_keypoint_names) if self._source_keypoint_names else []

    @property
    def feature_dim(self) -> int:
        return self.geo_cfg.feature_dim

    @property
    def num_keypoints(self) -> int:
        return self.geo_cfg.num_keypoints

    @property
    def tail_dim(self) -> int:
        return self.geo_cfg.tail_dim