# Unified Pose Trajectory Dataset
# Combines functionality from OFDMouseDataset and BasePoseTrajDataset
# Supports reconstruction with augmentation-aware training
# MODIFIED: Added likelihood loading and processing
#
# Geometry functions are in reconstruct_data_utils.py — this file delegates
# to them so that training and standalone inference share identical math.

from pathlib import Path
from typing import Union, Optional, Tuple, List

import numpy as np
import torch
import torch.utils.data
from torchvision import transforms

from .augmentations import GaussianNoise, Reflect, Rotation
from .keypoint_augment import KeypointAugment
from .reconstruct_data2_utils import (
    PoseGeometryConfig,
    restrict_keypoints,
    restrict_confidences,
    zero_out_confidences,
    fill_holes,
    compute_sequence_scale,
    transform_to_centered_data,
    featurize_keypoints,
    process_likelihoods,
    inverse_transform,
)
from .anatomical_grouping import build_feature_permutation, OFD_MOUSE_GROUPS


class PoseReconstructionDataset(torch.utils.data.Dataset):
    """
    Unified pose trajectory dataset with reconstruction capabilities.

    Features:
    - Configurable keypoint subsets
    - Per-sequence scaling
    - Centeralign transformation
    - Optional augmentations
    - Returns both augmented and raw features for reconstruction loss
    - Likelihood loading and weighting
    """

    def __init__(
        self,
        mode: str,
        data_path: Union[str, Path],
        keypoint_names: List[str],
        all_keypoints: List[str],
        center_keypoint: str,
        align_keypoints: Tuple[str, str],
        scale_keypoints: Optional[Tuple[str, str]] = None,
        num_frames: int = 80,
        sliding_window: int = 1,
        sampling_rate: int = 1,
        centeralign: bool = True,
        scale: bool = True,
        augmentations: bool = False,
        augmentation_p: float = 0.5,
        grid_size: int = 500,
        fill_holes: bool = False,
        data_augment: bool = False,
        return_likelihoods: bool = False,
        likelihood_threshold: float = 0.8,
        include_testdata: bool = False,
        nan_scattered_threshold: float = 0.4,
        nan_concentrated_threshold: float = 0.05,
        anatomical_groups=None,
        **kwargs,
    ):
        self.mode = mode
        self.data_path = Path(data_path)
        self.all_keypoints = all_keypoints
        self.include_testdata = include_testdata

        # ── Geometry config (shared with inference) ───────────────────────
        self.geo_cfg = PoseGeometryConfig(
            keypoint_names=tuple(keypoint_names),
            center_keypoint=center_keypoint,
            align_keypoints=tuple(align_keypoints),
            scale_keypoints=scale_keypoints if scale else None,
            grid_size=grid_size,
            centeralign=centeralign,
        )
        # ── Anatomical grouping permutation ───────────────────────────────
        # FIX 1: Always initialize to None first
        self._fwd_perm = None
        self._inv_perm = None
        self.anatomical_groups = anatomical_groups

        if anatomical_groups is not None and centeralign:
            # FIX 2: num_global_dims is 4 (cx, cy, sin, cos) — hardcoded
            # because PoseGeometryConfig doesn't have this field
            num_global_dims = 4
            fwd, inv = build_feature_permutation(
                keypoint_names=list(self.geo_cfg.keypoint_names),
                center_keypoint=self.geo_cfg.center_keypoint,
                groups=anatomical_groups,
                num_global_dims=num_global_dims,
            )
            self._fwd_perm = fwd
            self._inv_perm = inv
            group_size = len(anatomical_groups[0])
            group_dim = 2 * group_size
            print(
                f"[Anatomical Grouping] {len(anatomical_groups)} groups × "
                f"{group_dim} dims.  Use --patch_kernel 1 1 {group_dim}"
            )



        
        # Convenience aliases so existing code that reads self.X still works
        self.keypoint_names = list(self.geo_cfg.keypoint_names)
        self.center_keypoint = self.geo_cfg.center_keypoint
        self.align_keypoints = self.geo_cfg.align_keypoints
        self.scale_keypoints = self.geo_cfg.scale_keypoints
        self.grid_size = self.geo_cfg.grid_size
        self.centeralign = self.geo_cfg.centeralign
        self.num_keypoints = self.geo_cfg.num_keypoints
        self.keypoint_name_to_idx = self.geo_cfg.keypoint_name_to_idx
        self.num_individuals = self.geo_cfg.num_individuals

        # Keypoint index mapping (all → selected)
        self.keypoint_indices = [all_keypoints.index(bp) for bp in keypoint_names]

        # Derived properties
        self.kpts_dimensions = 2
        self.keyframe_shape = (
            self.num_individuals,
            self.num_keypoints,
            self.kpts_dimensions,
        )

        # Likelihood parameters
        self.return_likelihoods = return_likelihoods
        self.likelihood_threshold = likelihood_threshold

        # Data parameters
        self.max_keypoints_len = num_frames
        self.sliding_window = sliding_window
        self.sampling_rate = sampling_rate
        self.fill_holes_enabled = fill_holes
        self.data_augment = data_augment

        # Data storage (populated by load_data and preprocess)
        self.seq_keypoints = None
        self.seq_confidences = None
        self.keypoints_ids = None
        self.sequence_scales = None
        self.items = None
        self.n_frames = None
        self.raw_data = None
        self.discarded_windows = 0

        # Setup augmentations
        # self.augmentations = None
        self.keypoint_augment = None
        if augmentations:
            # gs = (self.grid_size, self.grid_size)
            # self.augmentations = transforms.Compose(
            #     [
            #         Rotation(grid_size=gs, p=augmentation_p),
            #         GaussianNoise(p=augmentation_p),
            #         Reflect(grid_size=gs, p=augmentation_p),
            #     ]
            # )
            # Local per-frame augmentation for denoising MAE
            self.keypoint_augment = KeypointAugment(
                rotation_prob=augmentation_p,
                noise_prob=augmentation_p,
                scale_prob=augmentation_p,
            )

        self.nan_scattered_threshold = nan_scattered_threshold
        self.nan_concentrated_threshold = nan_concentrated_threshold

        # Load and preprocess
        self.load_data()
        self.preprocess()

    # ── Derived property kept for backward compat ─────────────────────────

    @property
    def tail_dim(self):
        return self.geo_cfg.tail_dim


    @property
    def fwd_perm(self):
        """Forward permutation (standard → grouped). None if grouping disabled."""
        return self._fwd_perm

    @property
    def inv_perm(self):
        """Inverse permutation (grouped → standard). None if grouping disabled."""
        return self._inv_perm

    # ── Anatomical reordering helpers (FIX 3: these were missing) ─────────

    def _reorder_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply anatomical reordering to feature tensor (..., F).
        No-op if grouping is disabled."""
        if self._fwd_perm is None:
            return features
        return features[..., self._fwd_perm]

    def _reorder_likelihoods(self, likelihoods: torch.Tensor) -> torch.Tensor:
        """Apply same reordering to likelihood tensor (..., F).
        No-op if grouping is disabled."""
        if self._fwd_perm is None:
            return likelihoods
        return likelihoods[..., self._fwd_perm]

    def unreorder_features(self, features) -> np.ndarray:
        """Undo anatomical reordering (grouped → standard). For evaluation.
        No-op if grouping is disabled."""
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        if self._inv_perm is None:
            return features
        return features[..., self._inv_perm]





    # ── Data loading ──────────────────────────────────────────────────────

    def load_data(self) -> None:
        if self.mode == "pretrain":
            self.raw_data = np.load(self.data_path, allow_pickle=True).item()
            if self.include_testdata:
                test_path = str(self.data_path).replace("train", "test")
                if Path(test_path).exists():
                    raw_data_test = np.load(test_path, allow_pickle=True).item()
                    self.raw_data["sequences"].update(raw_data_test["sequences"])
        elif self.mode == "test":
            self.raw_data = np.load(self.data_path, allow_pickle=True).item()
        else:
            raise ValueError(
                f"Invalid mode: {self.mode}. Must be 'pretrain' or 'test'"
            )

    # ── Keypoint restriction (thin wrappers) ──────────────────────────────

    def _restrict_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        return restrict_keypoints(keypoints, self.keypoint_indices)

    def _restrict_confidences(self, confidences: np.ndarray) -> np.ndarray:
        return restrict_confidences(confidences, self.keypoint_indices)

    # ── Geometry delegations ──────────────────────────────────────────────

    def compute_sequence_scale(self, keypoints: np.ndarray) -> float:
        return compute_sequence_scale(keypoints, self.geo_cfg)

    def transform_to_centered_data(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return transform_to_centered_data(data, self.geo_cfg)

    def featurize_keypoints(self, keypoints: np.ndarray) -> torch.Tensor:
        result_np = featurize_keypoints(
            keypoints, self.max_keypoints_len, self.geo_cfg
        )
        return torch.from_numpy(result_np).float()

    def process_likelihoods(
        self, confidences: np.ndarray, seq_idx: int
    ) -> torch.Tensor:
        result_np = process_likelihoods(confidences, self.geo_cfg)
        return torch.from_numpy(result_np).float()

    def inverse_transform(self, features):
        """
        Inverse-transform features back to keypoint coordinates.

        If anatomical grouping is active, automatically un-reorders first
        so that inverse_transform in the utils sees standard feature order.
        """
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        # FIX 6: Undo grouping before standard inverse transform
        if self._inv_perm is not None:
            features = features[..., self._inv_perm]
        return inverse_transform(features, self.geo_cfg)

    # ── Scaling ───────────────────────────────────────────────────────────

    def scale_subsequence(self, subsequence: np.ndarray, seq_idx: int) -> np.ndarray:
        if self.sequence_scales is None or seq_idx >= len(self.sequence_scales):
            return subsequence
        scale = self.sequence_scales[seq_idx]
        if scale > 0:
            return subsequence / scale
        return subsequence

    # ── Window validity ───────────────────────────────────────────────────

    @staticmethod
    def _find_max_contiguous_true(bool_array: np.ndarray) -> int:
        if not np.any(bool_array):
            return 0
        padded = np.concatenate([[False], bool_array, [False]])
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        if len(starts) == 0:
            return 0
        return int(np.max(ends - starts))

    def check_window_validity(
        self, window: np.ndarray, confidences: np.ndarray = None
    ) -> bool:
        if confidences is None:
            window_len = window.shape[0]
            window_flat = window.reshape(window_len, -1)
            bad_per_frame = np.any(np.isnan(window_flat), axis=1)
        else:
            window_len = confidences.shape[0]
            conf_flat = confidences.reshape(window_len, -1)
            bad_per_frame = np.any(conf_flat <= 0.01, axis=1)

        bad_frame_count = np.sum(bad_per_frame)
        if bad_frame_count == 0:
            return True

        bad_percentage = bad_frame_count / window_len
        max_contiguous = self._find_max_contiguous_true(bad_per_frame)
        concentrated_percentage = max_contiguous / window_len

        if concentrated_percentage > self.nan_concentrated_threshold:
            return False
        if bad_percentage > self.nan_scattered_threshold:
            return False
        return True

    # ── Sample preparation ────────────────────────────────────────────────

    def prepare_sample(self, subsequence, seq_idx: int):
        """
        Returns a single concatenated torch.Tensor:
        - augment + likelihoods → (T, 1, 3F)
        - augment only          → (T, 1, 2F)
        - likelihoods only      → (T, 1, 2F)
        - plain                 → (T, 1,  F)
        """
        if isinstance(subsequence, tuple):
            keypoints_subseq, confidences_subseq = subsequence
        else:
            keypoints_subseq = subsequence
            confidences_subseq = None

        sequence_reshaped = keypoints_subseq.reshape(
            self.max_keypoints_len,
            self.num_individuals,
            self.num_keypoints,
            2,
        )

        # Process likelihoods
        likelihoods = None
        if self.return_likelihoods and confidences_subseq is not None:
            confidences_reshaped = confidences_subseq.reshape(
                self.max_keypoints_len,
                self.num_individuals,
                self.num_keypoints,
            )
            likelihoods = self.process_likelihoods(confidences_reshaped, seq_idx)

        # ── With augmentation ─────────────────────────────────────────────
        if self.keypoint_augment is not None and self.data_augment:
            aug_sequence = self.keypoint_augment(sequence_reshaped.copy())

            raw_features = self.featurize_keypoints(
                sequence_reshaped.reshape(self.max_keypoints_len, -1)
            )
            aug_features = self.featurize_keypoints(
                aug_sequence.reshape(self.max_keypoints_len, -1)
            )

            raw_features_np = raw_features.numpy()
            aug_features_np = aug_features.numpy()

            if self.centeralign:
                raw_features_np[:, :, 4:] = self.scale_subsequence(
                    raw_features_np[:, :, 4:].reshape(-1, self.tail_dim),
                    seq_idx,
                ).reshape(self.max_keypoints_len, self.num_individuals, self.tail_dim)

                aug_features_np[:, :, 4:] = self.scale_subsequence(
                    aug_features_np[:, :, 4:].reshape(-1, self.tail_dim),
                    seq_idx,
                ).reshape(self.max_keypoints_len, self.num_individuals, self.tail_dim)
            else:
                raw_features_np = self.scale_subsequence(
                    raw_features_np.reshape(-1, raw_features_np.shape[-1]),
                    seq_idx,
                ).reshape(raw_features.shape)

                aug_features_np = self.scale_subsequence(
                    aug_features_np.reshape(-1, aug_features_np.shape[-1]),
                    seq_idx,
                ).reshape(aug_features.shape)
            # FIX 4: Convert to tensor FIRST, then reorder the correct variables
            raw_features = torch.tensor(raw_features_np, dtype=torch.float32)
            aug_features = torch.tensor(aug_features_np, dtype=torch.float32)

            # Apply anatomical reordering to each block independently
            raw_features = self._reorder_features(raw_features)
            aug_features = self._reorder_features(aug_features)

            if likelihoods is not None:
                likelihoods = self._reorder_likelihoods(likelihoods)


            if self.return_likelihoods and likelihoods is not None:
                return torch.cat([raw_features, aug_features, likelihoods], dim=-1)
            return torch.cat([raw_features, aug_features], dim=-1)

        # ── Without augmentation ──────────────────────────────────────────
        else:
            features = self.featurize_keypoints(
                sequence_reshaped.reshape(self.max_keypoints_len, -1)
            )

            features_np = features.numpy()

            if self.centeralign:
                features_np[:, :, 4:] = self.scale_subsequence(
                    features_np[:, :, 4:].reshape(-1, self.tail_dim),
                    seq_idx,
                ).reshape(self.max_keypoints_len, self.num_individuals, self.tail_dim)
            else:
                features_np = self.scale_subsequence(
                    features_np.reshape(-1, features_np.shape[-1]),
                    seq_idx,
                ).reshape(features.shape)

            features = torch.tensor(features_np, dtype=torch.float32)
            features = self._reorder_features(features)

            if likelihoods is not None:
                likelihoods = self._reorder_likelihoods(likelihoods)


            if self.return_likelihoods and likelihoods is not None:
                return torch.cat([features, likelihoods], dim=-1)
            return features

    # ── Preprocessing (windowing, padding, filtering) ─────────────────────

    def preprocess(self):
        sequences = self.raw_data["sequences"]

        seq_keypoints = []
        seq_confidences = [] if self.return_likelihoods else None
        keypoints_ids = []
        sequence_scales = []

        sub_seq_length = self.max_keypoints_len
        sliding_window = self.sliding_window

        total_windows = 0
        discarded_windows = 0

        for seq_ix, (seq_name, sequence) in enumerate(sequences.items()):
            vec_seq = sequence["keypoints"]
            vec_seq = self._restrict_keypoints(vec_seq)

            # Get confidences for validity checking
            conf_seq_for_validity = sequence.get("confidences", None)
            if conf_seq_for_validity is not None:
                conf_seq_for_validity = self._restrict_confidences(
                    conf_seq_for_validity
                )
            else:
                conf_seq_for_validity = np.ones(
                    vec_seq.shape[:-1], dtype=np.float32
                )
                nan_mask = np.any(np.isnan(vec_seq), axis=-1)
                conf_seq_for_validity[nan_mask] = 0.0

            nose_idx = self.keypoint_name_to_idx.get("nose", None)
            tail_base_idx = self.keypoint_name_to_idx.get("tail_base", None)

            vec_seq, conf_seq_for_validity = zero_out_confidences(
                vec_seq,
                conf_seq_for_validity,
                nose_idx=nose_idx,
                tail_base_idx=tail_base_idx,
            )

            conf_seq = None
            if self.return_likelihoods:
                conf_seq = conf_seq_for_validity.copy()

            scale = self.compute_sequence_scale(vec_seq)
            sequence_scales.append(scale)

            vec_seq_original = vec_seq.copy()

            if self.fill_holes_enabled:
                vec_seq_filled = fill_holes(vec_seq)
            else:
                vec_seq_filled = vec_seq

            # Flatten
            vec_seq_original_flat = vec_seq_original.reshape(
                vec_seq_original.shape[0], -1
            )
            vec_seq_filled_flat = vec_seq_filled.reshape(
                vec_seq_filled.shape[0], -1
            )

            conf_seq_validity_flat = None
            if conf_seq_for_validity is not None:
                conf_seq_validity_flat = conf_seq_for_validity.reshape(
                    conf_seq_for_validity.shape[0], -1
                )

            if conf_seq is not None:
                conf_seq = conf_seq.reshape(conf_seq.shape[0], -1)

            # Temporal downsampling
            if self.sampling_rate > 1:
                vec_seq_original_flat = vec_seq_original_flat[:: self.sampling_rate]
                vec_seq_filled_flat = vec_seq_filled_flat[:: self.sampling_rate]
                if conf_seq_validity_flat is not None:
                    conf_seq_validity_flat = conf_seq_validity_flat[
                        :: self.sampling_rate
                    ]
                if conf_seq is not None:
                    conf_seq = conf_seq[:: self.sampling_rate]

            # Pad sequence edges
            pad_length = min(sub_seq_length, 120)

            pad_vec_original = np.pad(
                vec_seq_original_flat,
                ((pad_length // 2, pad_length - 1 - pad_length // 2), (0, 0)),
                mode="edge",
            )

            pad_vec_filled = np.pad(
                vec_seq_filled_flat,
                ((pad_length // 2, pad_length - 1 - pad_length // 2), (0, 0)),
                mode="edge",
            )

            pad_conf_validity = None
            if conf_seq_validity_flat is not None:
                pad_conf_validity = np.pad(
                    conf_seq_validity_flat,
                    (
                        (pad_length // 2, pad_length - 1 - pad_length // 2),
                        (0, 0),
                    ),
                    mode="edge",
                )

            if conf_seq is not None:
                pad_conf = np.pad(
                    conf_seq,
                    (
                        (pad_length // 2, pad_length - 1 - pad_length // 2),
                        (0, 0),
                    ),
                    mode="edge",
                )

            seq_keypoints.append(pad_vec_filled.astype(np.float32))
            if conf_seq is not None:
                seq_confidences.append(pad_conf.astype(np.float32))

            for i in np.arange(
                0, len(pad_vec_original) - sub_seq_length + 1, sliding_window
            ):
                total_windows += 1

                window_kpts = pad_vec_original[i : i + sub_seq_length]
                window_conf = None
                if pad_conf_validity is not None:
                    window_conf = pad_conf_validity[i : i + sub_seq_length]

                if self.check_window_validity(window_kpts, window_conf):
                    keypoints_ids.append((seq_ix, i))
                else:
                    discarded_windows += 1

        # Store results
        self.seq_keypoints = seq_keypoints
        self.seq_confidences = seq_confidences
        self.sequence_scales = np.array(sequence_scales, dtype=np.float32)
        self.keypoints_ids = keypoints_ids
        self.items = list(np.arange(len(keypoints_ids)))
        self.n_frames = len(self.keypoints_ids)
        self.discarded_windows = discarded_windows

        if total_windows > 0:
            print(
                f"[Reliability Filter] Total windows: {total_windows}, "
                f"Discarded: {discarded_windows} "
                f"({100 * discarded_windows / total_windows:.1f}%), "
                f"Kept: {len(keypoints_ids)}"
            )

        del self.raw_data

    # ── PyTorch Dataset interface ─────────────────────────────────────────

    def __len__(self):
        return len(self.keypoints_ids)

    def __getitem__(self, idx: int):
        subseq_ix = self.keypoints_ids[idx]
        seq_idx = subseq_ix[0]
        start_idx = subseq_ix[1]

        keypoints_subseq = self.seq_keypoints[seq_idx][
            start_idx : start_idx + self.max_keypoints_len
        ]

        confidences_subseq = None
        if self.seq_confidences is not None:
            confidences_subseq = self.seq_confidences[seq_idx][
                start_idx : start_idx + self.max_keypoints_len
            ]

        if confidences_subseq is not None:
            result = self.prepare_sample(
                (keypoints_subseq, confidences_subseq), seq_idx=seq_idx
            )
        else:
            result = self.prepare_sample(keypoints_subseq, seq_idx=seq_idx)

        if idx == 0:
            print("DATASET __getitem__ result shape:", result.shape)

        return result, []