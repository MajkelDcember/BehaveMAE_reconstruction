# Unified Pose Trajectory Dataset
# Combines functionality from OFDMouseDataset and BasePoseTrajDataset
# Supports reconstruction with augmentation-aware training

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
        return_augmented: bool = True,
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
            return_augmented: Return (aug_features, raw_features) instead of just features
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
        self.return_augmented = return_augmented
        
        # Data storage (populated by load_data and preprocess)
        self.seq_keypoints = None  # List of arrays (variable lengths)
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
    def fill_holes(data: np.ndarray) -> np.ndarray:
        """Interpolate missing keypoints (zeros)."""
        clean_data = copy.deepcopy(data)
        num_individuals = clean_data.shape[1]
        
        # Fill holes in first frame
        for m in range(num_individuals):
            holes = np.where(clean_data[0, m, :, 0] == 0)
            if not holes:
                continue
            for h in holes[0]:
                sub = np.where(clean_data[:, m, h, 0] != 0)
                if sub and sub[0].size > 0:
                    clean_data[0, m, h, :] = clean_data[sub[0][0], m, h, :]

        # Fill holes in remaining frames
        for fr in range(1, clean_data.shape[0]):
            for m in range(num_individuals):
                holes = np.where(clean_data[fr, m, :, 0] == 0)
                if not holes:
                    continue
                for h in holes[0]:
                    clean_data[fr, m, h, :] = clean_data[fr - 1, m, h, :]
        
        return clean_data

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
        
        # Reshape: (seq_len, num_inds, num_kpts, 2) -> (seq_len * num_inds, num_kpts, 2)
        data = data.reshape(-1, *data.shape[2:])
        
        # Center the data
        center = data[:, center_idx, :]
        centered_data = data - center[:, np.newaxis, :]
        
        # Compute rotation angle (align tail_base -> neck to y-axis)
        rotation_angle = np.arctan2(
            data[:, align_start_idx, 0] - data[:, align_end_idx, 0],
            data[:, align_start_idx, 1] - data[:, align_end_idx, 1],
        )
        
        # Create rotation matrix
        R = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ]).transpose((2, 0, 1))
        
        # Encode rotation as (sin, cos)
        rotation = np.concatenate([
            np.sin(rotation_angle)[:, np.newaxis],
            np.cos(rotation_angle)[:, np.newaxis],
        ], axis=-1)
        
        # Apply rotation
        centered_data = np.matmul(R, centered_data.transpose(0, 2, 1))
        centered_data = centered_data.transpose((0, 2, 1))
        
        # Flatten keypoints
        centered_data = centered_data.reshape((-1, self.num_keypoints * 2))
        
        return center, rotation, centered_data

    def transform_to_centeralign_components(self, data: np.ndarray) -> np.ndarray:
        """
        Apply centeralign transformation.
        
        Args:
            data: Shape (seq_len, num_inds, num_kpts, 2)
            
        Returns:
            Features: Shape (seq_len, num_inds, 4 + num_kpts * 2)
                     [center_x, center_y, sin, cos, ...rotated_keypoints...]
        """
        seq_len, num_inds = data.shape[:2]
        
        center, rotation, centered_data = self.transform_to_centered_data(data)
        
        # Concatenate: [center, rotation, rotated_keypoints]
        features = np.concatenate([center, rotation, centered_data], axis=1)
        features = features.reshape(seq_len, num_inds, -1)
        
        return features

    def featurize_keypoints(self, keypoints: np.ndarray) -> torch.Tensor:
        """
        Convert keypoints to features.
        
        Args:
            keypoints: Shape (num_frames, num_features) or 
                      (num_frames, num_inds, num_kpts, 2)
        
        Returns:
            Features as torch.Tensor
        """
        if self.centeralign:
            keypoints = keypoints.reshape(self.max_keypoints_len, *self.keyframe_shape)
            keypoints = self.transform_to_centeralign_components(keypoints)
        
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        return keypoints

    def scale_subsequence(self, sequence: np.ndarray, seq_idx: int) -> np.ndarray:
        """
        Scale sequence by its pre-computed scale.
        
        Args:
            sequence: Keypoints array (after featurization if centeralign)
            seq_idx: Index of the source sequence
            
        Returns:
            Scaled sequence
        """
        if self.sequence_scales is not None and seq_idx < len(self.sequence_scales):
            scale = self.sequence_scales[seq_idx]
            if scale > 0:
                sequence = sequence / scale
        return sequence

    def prepare_sample(
        self,
        sequence: np.ndarray,
        seq_idx: int,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Prepare a training sample with optional augmentations.
        
        Strategy for augmentation-aware reconstruction:
        1. raw_features = featurize(raw_keypoints) 
        2. aug_features = featurize(augment(raw_keypoints))
        3. Model sees aug_features (with masking)
        4. Loss computed against raw_features
        
        Args:
            sequence: Keypoints array (num_frames, num_features)
            seq_idx: Sequence index for scaling
            
        Returns:
            If return_augmented: (aug_features, raw_features)
            Else: features
        """
        # Reshape to keyframe format
        sequence_reshaped = sequence.reshape(self.max_keypoints_len, *self.keyframe_shape)
        
        # Handle augmentations
        if self.augmentations is not None and self.return_augmented:
            # Create augmented version
            sequence_aug = self.augmentations(sequence_reshaped.copy())
            
            # Featurize both versions
            raw_features = self.featurize_keypoints(
                sequence_reshaped.reshape(self.max_keypoints_len, -1)
            )
            aug_features = self.featurize_keypoints(
                sequence_aug.reshape(self.max_keypoints_len, -1)
            )
            
            # Apply per-sequence scaling
            raw_features_np = raw_features.numpy()
            aug_features_np = aug_features.numpy()
            
            if self.centeralign:
                # Scale only rotated keypoint part (skip center and rotation: first 4 features)
                raw_features_np[:, :, 4:] = self.scale_subsequence(
                    raw_features_np[:, :, 4:].reshape(-1, self.num_keypoints * 2),
                    seq_idx
                ).reshape(self.max_keypoints_len, self.num_individuals, -1)
                
                aug_features_np[:, :, 4:] = self.scale_subsequence(
                    aug_features_np[:, :, 4:].reshape(-1, self.num_keypoints * 2),
                    seq_idx
                ).reshape(self.max_keypoints_len, self.num_individuals, -1)
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
            
            return aug_features, raw_features
        
        else:
            # No augmentation or not returning both
            features = self.featurize_keypoints(
                sequence_reshaped.reshape(self.max_keypoints_len, -1)
            )
            
            # Apply scaling
            features_np = features.numpy()
            
            if self.centeralign:
                features_np[:, :, 4:] = self.scale_subsequence(
                    features_np[:, :, 4:].reshape(-1, self.num_keypoints * 2),
                    seq_idx
                ).reshape(self.max_keypoints_len, self.num_individuals, -1)
            else:
                features_np = self.scale_subsequence(
                    features_np.reshape(-1, features_np.shape[-1]),
                    seq_idx
                ).reshape(features.shape)
            
            features = torch.tensor(features_np, dtype=torch.float32)
            
            return features

    def preprocess(self):
        """
        Preprocess dataset:
        - Computes per-sequence scales
        - Handles variable-length sequences
        - Creates sliding window samples
        """
        sequences = self.raw_data["sequences"]
        
        seq_keypoints = []
        keypoints_ids = []
        sequence_scales = []
        
        sub_seq_length = self.max_keypoints_len
        sliding_window = self.sliding_window
        
        for seq_ix, (seq_name, sequence) in enumerate(sequences.items()):
            # Get keypoints and restrict to subset
            vec_seq = sequence["keypoints"]
            vec_seq = self._restrict_keypoints(vec_seq)
            
            # Compute per-sequence scale
            scale = self.compute_sequence_scale(vec_seq)
            sequence_scales.append(scale)
            
            # Fill holes if requested
            if self.fill_holes_enabled:
                vec_seq = self.fill_holes(vec_seq)
            
            # Flatten keypoints
            vec_seq = vec_seq.reshape(vec_seq.shape[0], -1)
            
            # Temporal downsampling
            if self.sampling_rate > 1:
                vec_seq = vec_seq[:: self.sampling_rate]
            
            # Pad sequence edges
            pad_length = min(sub_seq_length, 120)
            pad_vec = np.pad(
                vec_seq,
                ((pad_length // 2, pad_length - 1 - pad_length // 2), (0, 0)),
                mode="edge",
            )
            
            # Store (as individual array to handle variable lengths)
            seq_keypoints.append(pad_vec.astype(np.float32))
            
            # Create sliding window sample indices
            keypoints_ids.extend([
                (seq_ix, i)
                for i in np.arange(
                    0, len(pad_vec) - sub_seq_length + 1, sliding_window
                )
            ])
        
        # Store results
        self.seq_keypoints = seq_keypoints  # List of arrays
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
            If return_augmented: ((aug_features, raw_features), [])
            Else: (features, [])
        """
        subseq_ix = self.keypoints_ids[idx]
        seq_idx = subseq_ix[0]
        start_idx = subseq_ix[1]
        
        # Get subsequence
        subsequence = self.seq_keypoints[seq_idx][
            start_idx : start_idx + self.max_keypoints_len
        ]
        
        # Prepare sample
        result = self.prepare_sample(subsequence, seq_idx=seq_idx)
        
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
        Inverse transform features back to original keypoint coordinates.
        
        Args:
            features: Transformed features, shape (..., num_features)
                     If centeralign: (..., 4 + num_keypoints * 2)
            seq_idx: Sequence index for unscaling
            unscale: Whether to apply inverse scaling
            
        Returns:
            Original keypoints, shape (..., num_keypoints, 2)
        """
        if not self.centeralign:
            # Just reshape if no centeralign
            if isinstance(features, torch.Tensor):
                features = features.detach().cpu().numpy()
            return features.reshape(*features.shape[:-1], self.num_keypoints, 2)
        
        # Handle centeralign inverse
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        
        original_shape = features.shape[:-1]
        features_flat = features.reshape(-1, features.shape[-1])
        
        # Extract components: [center_x, center_y, sin, cos, ...rotated_kpts...]
        center = features_flat[:, 0:2]
        sin_cos = features_flat[:, 2:4]
        rotated_kpts = features_flat[:, 4:]
        
        # Unscale rotated keypoints if requested
        if unscale and self.sequence_scales is not None and seq_idx is not None:
            scale = self.sequence_scales[seq_idx]
            if scale > 0:
                rotated_kpts = rotated_kpts * scale
        
        # Reshape rotated keypoints
        rotated_kpts = rotated_kpts.reshape(-1, self.num_keypoints, 2)
        
        # Compute rotation angles from sin/cos
        angles = np.arctan2(sin_cos[:, 0], sin_cos[:, 1])
        
        # Create inverse rotation matrices
        cos_theta = np.cos(-angles)
        sin_theta = np.sin(-angles)
        R_inv = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ]).transpose(2, 0, 1)
        
        # Apply inverse rotation
        unrotated_kpts = np.matmul(
            R_inv,
            rotated_kpts.transpose(0, 2, 1)
        ).transpose(0, 2, 1)
        
        # Add back center
        original_kpts = unrotated_kpts + center[:, np.newaxis, :]
        
        # Reshape to original
        final_shape = original_shape + (self.num_keypoints, 2)
        original_kpts = original_kpts.reshape(final_shape)
        
        return original_kpts


