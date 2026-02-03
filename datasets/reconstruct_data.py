# Unified Pose Trajectory Dataset
# Combines functionality from OFDMouseDataset and BasePoseTrajDataset
# Supports reconstruction with augmentation-aware training
# MODIFIED: Added likelihood loading and processing

import copy
from pathlib import Path
from typing import Union, Optional, Tuple, List

import numpy as np
import torch
import torch.utils.data
from torchvision import transforms

from .augmentations import GaussianNoise, Reflect, Rotation


class PoseReconstructionDataset(torch.utils.data.Dataset):
    """
    Unified pose trajectory dataset with reconstruction capabilities.
    
    Features:
    - Configurable keypoint subsets
    - Per-sequence scaling
    - Centeralign transformation
    - Optional augmentations
    - Returns both augmented and raw features for reconstruction loss
    - Likelihood loading and weighting (NEW)
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
        return_likelihoods: bool = False,  # NEW
        likelihood_threshold: float = 0.8,  # NEW
        include_testdata: bool = False,
        **kwargs
    ):
        """
        Args:
            mode: 'pretrain' or 'test'
            data_path: Path to .npy data file
            keypoint_names: List of keypoint names to use (subset of all_keypoints)
            all_keypoints: List of all keypoint names in raw data
            center_keypoint: Keypoint name to center on
            align_keypoints: (start, end) keypoint names for rotation alignment
            scale_keypoints: (start, end) keypoint names for computing scale (None = no scaling)
            num_frames: Number of frames per sample
            sliding_window: Sliding window step size
            sampling_rate: Temporal downsampling rate
            centeralign: Whether to apply centeralign transformation
            scale: Whether to apply per-sequence scaling
            augmentations: Whether to enable augmentations
            augmentation_p: Probability for each augmentation
            grid_size: Arena size for augmentations
            fill_holes: Whether to interpolate missing keypoints
            data_augment: Return (aug_features, raw_features) instead of just features
            return_likelihoods: Return likelihood weights for loss weighting (NEW)
            likelihood_threshold: Threshold for likelihood filtering (NEW)
            include_testdata: Include test data in pretraining mode
        """
        self.mode = mode
        self.data_path = Path(data_path)
        self.keypoint_names = keypoint_names
        self.all_keypoints = all_keypoints
        self.center_keypoint = center_keypoint
        self.align_keypoints = align_keypoints
        self.scale_keypoints = scale_keypoints if scale else None
        self.grid_size = grid_size
        self.include_testdata = include_testdata
        
        # NEW: Likelihood parameters
        self.return_likelihoods = return_likelihoods
        self.likelihood_threshold = likelihood_threshold
        
        # Create mappings
        self.keypoint_name_to_idx = {name: i for i, name in enumerate(keypoint_names)}
        self.keypoint_indices = [all_keypoints.index(bp) for bp in keypoint_names]
        
        # Derived properties
        self.num_individuals = 1
        self.num_keypoints = len(keypoint_names)
        self.kpts_dimensions = 2
        self.keyframe_shape = (self.num_individuals, self.num_keypoints, self.kpts_dimensions)
        
        # Data parameters
        self.max_keypoints_len = num_frames
        self.sliding_window = sliding_window
        self.sampling_rate = sampling_rate
        self.centeralign = centeralign
        self.fill_holes_enabled = fill_holes
        self.data_augment = data_augment
        
        # Data storage (populated by load_data and preprocess)
        self.seq_keypoints = None  # List of arrays (variable lengths)
        self.seq_confidences = None  # NEW: List of confidence arrays
        self.keypoints_ids = None
        self.sequence_scales = None
        self.items = None
        self.n_frames = None
        self.raw_data = None
        
        # Setup augmentations
        self.augmentations = None
        if augmentations:
            gs = (self.grid_size, self.grid_size)
            self.augmentations = transforms.Compose([
                Rotation(grid_size=gs, p=augmentation_p),
                GaussianNoise(p=augmentation_p),
                Reflect(grid_size=gs, p=augmentation_p),
            ])
        
        # Load and preprocess
        self.load_data()
        self.preprocess()
    @property
    def tail_dim(self):
        # number of non-center keypoints * 2
        return (self.num_keypoints - 1) * 2

    def load_data(self) -> None:
        """Load data from .npy file."""
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
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'pretrain' or 'test'")

    def _restrict_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Restrict keypoints to selected subset.
        
        Args:
            keypoints: Array of shape (n_frames, n_individuals, n_all_keypoints, 2)
            
        Returns:
            Array of shape (n_frames, n_individuals, n_selected_keypoints, 2)
        """
        return keypoints[:, :, self.keypoint_indices, :]
    
    def _restrict_confidences(self, confidences: np.ndarray) -> np.ndarray:
        """
        Restrict confidences to selected subset.
        
        Args:
            confidences: Array of shape (n_frames, n_individuals, n_all_keypoints)
            
        Returns:
            Array of shape (n_frames, n_individuals, n_selected_keypoints)
        """
        return confidences[:, :, self.keypoint_indices]

    def compute_sequence_scale(self, keypoints: np.ndarray) -> float:
        """
        Compute scale for a sequence based on median body length.
        
        Args:
            keypoints: Shape (n_frames, n_individuals, n_keypoints, 2)
            
        Returns:
            Scale value (median distance between scale_keypoints)
        """
        if self.scale_keypoints is None:
            return 1.0
        
        start_name, end_name = self.scale_keypoints
        start_idx = self.keypoint_name_to_idx[start_name]
        end_idx = self.keypoint_name_to_idx[end_name]
        
        start_pts = keypoints[:, 0, start_idx, :]
        end_pts = keypoints[:, 0, end_idx, :]
        
        distances = np.linalg.norm(start_pts - end_pts, axis=1)
        scale = np.median(distances)
        
        return scale if scale > 0 else 1.0

    @staticmethod
    def fill_holes(vec_seq):
            if np.any(np.isnan(vec_seq)):
                # Simple forward fill for NaN values
                for ind in range(vec_seq.shape[1]):
                    for kpt in range(vec_seq.shape[2]):
                        for dim in range(vec_seq.shape[3]):
                            mask = np.isnan(vec_seq[:, ind, kpt, dim])
                            if np.any(mask):
                                # Forward fill
                                idx = np.where(~mask)[0]
                                if len(idx) > 0:
                                    vec_seq[:, ind, kpt, dim] = np.interp(
                                        np.arange(len(vec_seq)),
                                        idx,
                                        vec_seq[idx, ind, kpt, dim]
                                    )
                                else:
                                    # All NaN, use zeros
                                    vec_seq[:, ind, kpt, dim] = 0
            return vec_seq

    def transform_to_centered_data(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Center and rotate data.
        
        Returns:
            center: (T, 2) center positions
            rotation: (T, 2) encoded as (sin, cos)
            centered_data: (T, num_keypoints * 2) rotated keypoints
        """
        # Get indices
        center_idx = self.keypoint_name_to_idx[self.center_keypoint]
        align_start_idx = self.keypoint_name_to_idx[self.align_keypoints[0]]
        align_end_idx = self.keypoint_name_to_idx[self.align_keypoints[1]]
        
        # data shape: (T, num_individuals, num_keypoints, 2)
        # Extract center
        center = data[:, 0, center_idx, :]  # (T, 2)
        
        # Center the data
        centered = data - center[:, np.newaxis, np.newaxis, :]  # (T, num_individuals, num_keypoints, 2)åç
        
        # Compute rotation from align_keypoints
        align_vec = centered[:, 0, align_end_idx, :] - centered[:, 0, align_start_idx, :]  # (T, 2)
        angles = np.arctan2(align_vec[:, 1], align_vec[:, 0])  # (T,)
        
        # Create rotation matrices
        cos_theta = np.cos(-angles)
        sin_theta = np.sin(-angles)
        R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]]).transpose(2, 0, 1)  # (T, 2, 2)
        
        # Apply rotation
        rotated = np.matmul(R, centered.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)  # (T, num_individuals, num_keypoints, 2)
        
        # Flatten keypoints
        rotated_flat = rotated[:, 0].reshape(rotated.shape[0], -1)
        
        # Encode rotation as (sin, cos)
        rotation = np.stack([sin_theta, cos_theta], axis=1)  # (T, 2)

        # # --- [DEBUG START] ---
        # # Check for NaNs which often happen in rotation logic if align points overlap
        # if np.isnan(rotation).any():
        #     print(f"[DEBUG] FATAL: NaNs detected in rotation matrix!")
        #     print(f"  > Center shape: {center.shape}")
        #     print(f"  > Rotated flat shape: {rotated_flat.shape}")
        # # --- [DEBUG END] ---
        
        return center, rotation, rotated_flat

    def featurize_keypoints(self, keypoints: np.ndarray) -> torch.Tensor:
        """
        Convert keypoints to features WITHOUT redundant zero center keypoint.
        
        Returns:
            (T, 1, 4 + 2*(num_keypoints-1))
        """
        if not self.centeralign:
            features = keypoints.reshape(
                self.max_keypoints_len,
                self.num_individuals,
                self.num_keypoints * 2
            )
            return torch.from_numpy(features).float()

        keypoints_reshaped = keypoints.reshape(
            self.max_keypoints_len,
            self.num_individuals,
            self.num_keypoints,
            2
        )

        center_idx = self.keypoint_name_to_idx[self.center_keypoint]

        center, rotation, centered_kpts = self.transform_to_centered_data(keypoints_reshaped)

        # Remove center keypoint from centered_kpts
        centered_kpts = centered_kpts.reshape(
            self.max_keypoints_len, self.num_keypoints, 2
        )
        centered_kpts = np.delete(centered_kpts, center_idx, axis=1)
        centered_kpts = centered_kpts.reshape(self.max_keypoints_len, -1)

        features = np.concatenate(
            [center, rotation, centered_kpts],
            axis=1
        )

        features = features[:, np.newaxis, :]
        return torch.from_numpy(features).float()


    def process_likelihoods(self, confidences: np.ndarray, seq_idx: int) -> torch.Tensor:
        """
        Likelihoods aligned with reduced feature set.
        """
        T = confidences.shape[0]

        # Frame corruption rule (unchanged)
        frame_corrupted = np.any(confidences <= 0, axis=(1, 2))
        confidences[frame_corrupted] = 0.0

        confidences = np.where(
            confidences >= self.likelihood_threshold,
            confidences,
            0.0
        )

        if self.centeralign:
            center_idx = self.keypoint_name_to_idx[self.center_keypoint]
            a_idx = self.keypoint_name_to_idx[self.align_keypoints[0]]
            b_idx = self.keypoint_name_to_idx[self.align_keypoints[1]]

            # Center confidence
            center_conf = confidences[:, :, center_idx:center_idx+1]
            center_weights = np.repeat(center_conf, 2, axis=2)

            # Rotation confidence = min(conf(a), conf(b))
            rot_conf = np.minimum(
                confidences[:, :, a_idx:a_idx+1],
                confidences[:, :, b_idx:b_idx+1],
            )
            rotation_weights = np.repeat(rot_conf, 2, axis=2)

            # Remove center keypoint from remaining confidences
            kpt_conf = np.delete(confidences, center_idx, axis=2)
            kpt_weights = np.repeat(kpt_conf, 2, axis=2)

            weights = np.concatenate(
                [center_weights, rotation_weights, kpt_weights],
                axis=2
            )
        else:
            weights = np.repeat(confidences, 2, axis=2)

        return torch.from_numpy(weights).float()


    def scale_subsequence(self, subsequence: np.ndarray, seq_idx: int) -> np.ndarray:
        """
        Scale a subsequence using pre-computed sequence scale.
        
        Args:
            subsequence: Features to scale, shape (..., num_features)
            seq_idx: Index of the sequence
            
        Returns:
            Scaled features
        """
        if self.sequence_scales is None or seq_idx >= len(self.sequence_scales):
            return subsequence
        
        scale = self.sequence_scales[seq_idx]
        if scale > 0:
            return subsequence / scale
        return subsequence

    def prepare_sample(
        self,
        subsequence: np.ndarray,
        seq_idx: int
    ):
        """
        Prepare a sample for training.
        
        Args:
            subsequence: (max_keypoints_len, num_features)
                If with confidences: tuple of (keypoints, confidences)
            seq_idx: Sequence index
            
        Returns:
            Single concatenated tensor:
            - If augment + likelihoods: [raw, aug, confidences] shape (T, 1, 3*num_features)
            - If augment only: [raw, aug] shape (T, 1, 2*num_features)
            - If likelihoods only: [features, confidences] shape (T, 1, 2*num_features)
            - Otherwise: features shape (T, 1, num_features)
        """
        # Check if subsequence is a tuple (keypoints, confidences)
        if isinstance(subsequence, tuple):
            keypoints_subseq, confidences_subseq = subsequence
        else:
            keypoints_subseq = subsequence
            confidences_subseq = None
        
        # Reshape
        sequence_reshaped = keypoints_subseq.reshape(
            self.max_keypoints_len,
            self.num_individuals,
            self.num_keypoints,
            2
        )
        
        # Process likelihoods if available
        likelihoods = None
        if self.return_likelihoods and confidences_subseq is not None:
            confidences_reshaped = confidences_subseq.reshape(
                self.max_keypoints_len,
                self.num_individuals,
                self.num_keypoints
            )
            likelihoods = self.process_likelihoods(confidences_reshaped, seq_idx)
        
        # Handle augmentation
        if self.augmentations is not None and self.data_augment:
            # Create augmented version
            aug_sequence = self.augmentations(sequence_reshaped.copy())
            
            # Featurize both
            raw_features = self.featurize_keypoints(
                sequence_reshaped.reshape(self.max_keypoints_len, -1)
            )
            aug_features = self.featurize_keypoints(
                aug_sequence.reshape(self.max_keypoints_len, -1)
            )
            
            # Apply scaling
            raw_features_np = raw_features.numpy()
            aug_features_np = aug_features.numpy()
            
            if self.centeralign:
                # Only scale the keypoint part (skip center + rotation)
                # raw_features_np[:, :, 4:] = self.scale_subsequence(
                #     raw_features_np[:, :, 4:].reshape(-1, self.num_keypoints * 2),
                #     seq_idx
                # ).reshape(self.max_keypoints_len, self.num_individuals, -1)
                
                # aug_features_np[:, :, 4:] = self.scale_subsequence(
                #     aug_features_np[:, :, 4:].reshape(-1, self.num_keypoints * 2),
                #     seq_idx
                # ).reshape(self.max_keypoints_len, self.num_individuals, -1)
                raw_features_np[:, :, 4:] = self.scale_subsequence(
                    raw_features_np[:, :, 4:].reshape(-1, self.tail_dim),
                    seq_idx
                ).reshape(self.max_keypoints_len, self.num_individuals, self.tail_dim)

                aug_features_np[:, :, 4:] = self.scale_subsequence(
                    aug_features_np[:, :, 4:].reshape(-1, self.tail_dim),
                    seq_idx
                ).reshape(self.max_keypoints_len, self.num_individuals, self.tail_dim)

            else:
                # Scale entire feature vector
                raw_features_np = self.scale_subsequence(
                    raw_features_np.reshape(-1, raw_features_np.shape[-1]),
                    seq_idx
                ).reshape(raw_features.shape)
                
                aug_features_np = self.scale_subsequence(
                    aug_features_np.reshape(-1, aug_features_np.shape[-1]),
                    seq_idx
                ).reshape(aug_features.shape)
            
            raw_features = torch.tensor(raw_features_np, dtype=torch.float32)
            aug_features = torch.tensor(aug_features_np, dtype=torch.float32)
            
            # # --- [DEBUG START] ---
            # final_tensor = None
            # if self.return_likelihoods and likelihoods is not None:
            #     final_tensor = torch.cat([raw_features, aug_features, likelihoods], dim=-1)
            # else:
            #     final_tensor = torch.cat([raw_features, aug_features], dim=-1)

    
            # if seq_idx == 0:
            #     F = raw_features.shape[-1]
            #     print(f"\n[DEBUG] prepare_sample (Seq {seq_idx}) - Augmentation Path:")
            #     print(f"  > Raw features: {raw_features.shape}")
            #     print(f"  > Aug features: {aug_features.shape}")
            #     if likelihoods is not None:
            #         print(f"  > Likelihoods:  {likelihoods.shape}")
            #     print(f"  > Final Tensor: {final_tensor.shape}")
            #     print(f"  > Expected Feature Dim (F): {F}")
            #     print(f"  > Is Tensor Width == 2*F? {final_tensor.shape[-1] == 2*F}")
            #     print(f"  > Is Tensor Width == 3*F? {final_tensor.shape[-1] == 3*F}")
            # # --- [DEBUG END] ---




            # Concatenate: [raw, aug, (optionally) likelihoods]
            if self.return_likelihoods and likelihoods is not None:
                return torch.cat([raw_features, aug_features, likelihoods], dim=-1)
            return torch.cat([raw_features, aug_features], dim=-1)
        
        else:
            # No augmentation or not returning both
            features = self.featurize_keypoints(
                sequence_reshaped.reshape(self.max_keypoints_len, -1)
            )
            
            # Apply scaling
            features_np = features.numpy()
            
            if self.centeralign:
                # features_np[:, :, 4:] = self.scale_subsequence(
                #     features_np[:, :, 4:].reshape(-1, self.num_keypoints * 2),
                #     seq_idx
                # ).reshape(self.max_keypoints_len, self.num_individuals, -1)
                features_np[:, :, 4:] = self.scale_subsequence(
                    features_np[:, :, 4:].reshape(-1, self.tail_dim),
                    seq_idx
                ).reshape(self.max_keypoints_len, self.num_individuals, self.tail_dim)

            else:
                features_np = self.scale_subsequence(
                    features_np.reshape(-1, features_np.shape[-1]),
                    seq_idx
                ).reshape(features.shape)
            
            features = torch.tensor(features_np, dtype=torch.float32)
            
            # Concatenate: [features, (optionally) likelihoods]
            if self.return_likelihoods and likelihoods is not None:
                return torch.cat([features, likelihoods], dim=-1)
            return features

    def preprocess(self):
        """
        Preprocess dataset:
        - Computes per-sequence scales
        - Handles variable-length sequences
        - Creates sliding window samples
        - Stores confidences if return_likelihoods is True
        """
        sequences = self.raw_data["sequences"]
        
        seq_keypoints = []
        seq_confidences = [] if self.return_likelihoods else None
        keypoints_ids = []
        sequence_scales = []
        
        sub_seq_length = self.max_keypoints_len
        sliding_window = self.sliding_window
        
        for seq_ix, (seq_name, sequence) in enumerate(sequences.items()):
            # Get keypoints and restrict to subset
            vec_seq = sequence["keypoints"]
            vec_seq = self._restrict_keypoints(vec_seq)
            
            # Get confidences if needed
            conf_seq = None
            if self.return_likelihoods:
                conf_seq = sequence.get("confidences", None)
                if conf_seq is not None:
                    conf_seq = self._restrict_confidences(conf_seq)
            
            # Compute per-sequence scale
            scale = self.compute_sequence_scale(vec_seq)
            sequence_scales.append(scale)

            
            # # --- [DEBUG START] ---
            # if seq_ix == 0:
            #     print(f"\n[DEBUG] Preprocess Sequence 0:")
            #     print(f"  > Keypoints shape (restricted): {vec_seq.shape}")
            #     print(f"  > Scale computed: {scale:.4f}")
            #     if conf_seq is not None:
            #         print(f"  > Confidences found. Shape: {conf_seq.shape}")
            #         print(f"  > Confidences range: [{np.nanmin(conf_seq):.2f}, {np.nanmax(conf_seq):.2f}]")
            #     else:
            #         print(f"  > WARNING: No confidences found (return_likelihoods={self.return_likelihoods})")
            # # --- [DEBUG END] ---
            
            # Fill holes if requested ...
            
            # Fill holes if requested
            if self.fill_holes_enabled:
                vec_seq = self.fill_holes(vec_seq)
                # Note: We don't fill holes in confidences
            
            # Flatten keypoints
            vec_seq = vec_seq.reshape(vec_seq.shape[0], -1)
            
            # Flatten confidences
            if conf_seq is not None:
                conf_seq = conf_seq.reshape(conf_seq.shape[0], -1)
            
            # Temporal downsampling
            if self.sampling_rate > 1:
                vec_seq = vec_seq[:: self.sampling_rate]
                if conf_seq is not None:
                    conf_seq = conf_seq[:: self.sampling_rate]
            
            # Pad sequence edges
            pad_length = min(sub_seq_length, 120)
            pad_vec = np.pad(
                vec_seq,
                ((pad_length // 2, pad_length - 1 - pad_length // 2), (0, 0)),
                mode="edge",
            )
            
            # Pad confidences
            if conf_seq is not None:
                pad_conf = np.pad(
                    conf_seq,
                    ((pad_length // 2, pad_length - 1 - pad_length // 2), (0, 0)),
                    mode="edge",
                )
            
            # Store (as individual array to handle variable lengths)
            seq_keypoints.append(pad_vec.astype(np.float32))
            if conf_seq is not None:
                seq_confidences.append(pad_conf.astype(np.float32))
            
            # Create sliding window sample indices
            keypoints_ids.extend([
                (seq_ix, i)
                for i in np.arange(
                    0, len(pad_vec) - sub_seq_length + 1, sliding_window
                )
            ])
        
        # Store results
        self.seq_keypoints = seq_keypoints  # List of arrays
        self.seq_confidences = seq_confidences  # List of confidence arrays (or None)
        self.sequence_scales = np.array(sequence_scales, dtype=np.float32)
        self.keypoints_ids = keypoints_ids
        self.items = list(np.arange(len(keypoints_ids)))
        self.n_frames = len(self.keypoints_ids)
        
        # Clean up
        del self.raw_data

    def __len__(self):
        return len(self.keypoints_ids)

    def __getitem__(self, idx: int):
        """
        Get a sample.
        
        Returns:
            (concatenated_tensor, []) where concatenated_tensor is:
            - If augment + likelihoods: [raw, aug, confidences] (T, 1, 3*num_features)
            - If augment only: [raw, aug] (T, 1, 2*num_features)
            - If likelihoods only: [features, confidences] (T, 1, 2*num_features)
            - Otherwise: features (T, 1, num_features)
        """
        subseq_ix = self.keypoints_ids[idx]
        seq_idx = subseq_ix[0]
        start_idx = subseq_ix[1]
        
        # Get subsequence
        keypoints_subseq = self.seq_keypoints[seq_idx][
            start_idx : start_idx + self.max_keypoints_len
        ]
        
        # Get confidence subsequence if available
        confidences_subseq = None
        if self.seq_confidences is not None:
            confidences_subseq = self.seq_confidences[seq_idx][
                start_idx : start_idx + self.max_keypoints_len
            ]
        
        # Prepare sample (with or without confidences) - returns single concatenated tensor
        if confidences_subseq is not None:
            result = self.prepare_sample((keypoints_subseq, confidences_subseq), seq_idx=seq_idx)
        else:
            result = self.prepare_sample(keypoints_subseq, seq_idx=seq_idx)
        if idx == 0:
            print("DATASET __getitem__ result shape:", result.shape)

        return result, []

    # =========================================================================
    # Reconstruction Utilities
    # =========================================================================
    
    def inverse_transform(
        self,
        features: Union[np.ndarray, torch.Tensor],
        seq_idx: Optional[int] = None,
        unscale: bool = True
    ) -> np.ndarray:
        """
        Inverse transform with implicit center keypoint reconstruction.
        """
        if not self.centeralign:
            if isinstance(features, torch.Tensor):
                features = features.detach().cpu().numpy()
            return features.reshape(*features.shape[:-1], self.num_keypoints, 2)

        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        original_shape = features.shape[:-1]
        features_flat = features.reshape(-1, features.shape[-1])

        center = features_flat[:, 0:2]
        sin_cos = features_flat[:, 2:4]
        rotated_kpts = features_flat[:, 4:]

        if unscale and self.sequence_scales is not None and seq_idx is not None:
            scale = self.sequence_scales[seq_idx]
            if scale > 0:
                rotated_kpts = rotated_kpts * scale

        num_other_kpts = self.num_keypoints - 1
        rotated_kpts = rotated_kpts.reshape(-1, num_other_kpts, 2)

        angles = np.arctan2(sin_cos[:, 0], sin_cos[:, 1])
        cos_theta = np.cos(-angles)
        sin_theta = np.sin(-angles)
        R_inv = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ]).transpose(2, 0, 1)

        unrotated = np.matmul(
            R_inv,
            rotated_kpts.transpose(0, 2, 1)
        ).transpose(0, 2, 1)

        unrotated += center[:, None, :]

        # Reinsert center keypoint
        full_kpts = np.zeros((unrotated.shape[0], self.num_keypoints, 2))
        center_idx = self.keypoint_name_to_idx[self.center_keypoint]

        full_kpts[:, center_idx, :] = center
        mask = np.ones(self.num_keypoints, dtype=bool)
        mask[center_idx] = False
        full_kpts[:, mask, :] = unrotated

        final_shape = original_shape + (self.num_keypoints, 2)
        return full_kpts.reshape(final_shape)
