# OFD Mouse Dataset for hBehaveMAE
# Adapted from MABeMouseDataset for single-mouse OFD behavioral data

import __future__

import os
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from .augmentations import GaussianNoise, Reflect, Rotation
from .pose_traj_dataset import BasePoseTrajDataset


class OFDMouseDataset(BasePoseTrajDataset):
    """
    OFD Mouse dataset for single-individual behavioral tracking.
    Uses a subset of keypoints defined in use_bodyparts.
    """

    DEFAULT_FRAME_RATE = 25
    DEFAULT_GRID_SIZE = 500  # Typical OFD arena size
    NUM_INDIVIDUALS = 1
    NUM_KEYPOINTS = 8  # Subset of bodyparts used for modeling
    KPTS_DIMENSIONS = 2
    KEYFRAME_SHAPE = (NUM_INDIVIDUALS, NUM_KEYPOINTS, KPTS_DIMENSIONS)
    NUM_TASKS = 1  # Single-task dataset

    # All available bodyparts in the full dataset
    ALL_BODYPARTS = [
        "nose",
        "left_ear",
        "right_ear",
        "left_ear_tip",
        "right_ear_tip",
        "left_eye",
        "right_eye",
        "neck",
        "mid_back",
        "mouse_center",
        "mid_backend",
        "mid_backend2",
        "mid_backend3",
        "tail_base",
        "tail1",
        "tail2",
        "tail3",
        "tail4",
        "tail5",
        "left_shoulder",
        "left_midside",
        "left_hip",
        "right_shoulder",
        "right_midside",
        "right_hip",
        "tail_end",
        "head_midpoint",
    ]

    # Subset of bodyparts used for modeling
    STR_BODY_PARTS = [
        "nose",
        "head_midpoint",
        "left_ear",
        "right_ear",
        "neck",
        "mouse_center",
        "mid_backend3",
        "tail_base",
    ]
    BODY_PART_2_INDEX = {w: i for i, w in enumerate(STR_BODY_PARTS)}

    def __init__(
        self,
        mode: str,
        path_to_data_dir: Path,
        scale: bool = True,
        sampling_rate: int = 1,
        num_frames: int = 80,
        sliding_window: int = 1,
        fill_holes: bool = False,
        augmentations: transforms.Compose = None,
        centeralign: bool = True,
        include_testdata: bool = False,
        **kwargs
    ):

        super().__init__(
            path_to_data_dir,
            scale,
            sampling_rate,
            num_frames,
            sliding_window,
            fill_holes,
            **kwargs
        )

        self.sample_frequency = self.DEFAULT_FRAME_RATE  # downsample frames if needed

        self.mode = mode

        self.centeralign = centeralign
        
        # Storage for per-sequence scale values
        self.sequence_scales = None

        # Create mapping from STR_BODY_PARTS to ALL_BODYPARTS indices
        self.bodypart_indices = [self.ALL_BODYPARTS.index(bp) for bp in self.STR_BODY_PARTS]

        if augmentations:
            gs = (self.DEFAULT_GRID_SIZE, self.DEFAULT_GRID_SIZE)
            self.augmentations = transforms.Compose(
                [
                    Rotation(grid_size=gs, p=0.5),
                    GaussianNoise(p=0.5),
                    Reflect(grid_size=gs, p=0.5),
                ]
            )
        else:
            self.augmentations = None

        self.load_data(include_testdata)

        self.preprocess()

    def load_data(self, include_testdata) -> None:
        """Loads dataset"""
        if self.mode == "pretrain":
            self.raw_data = np.load(self.path, allow_pickle=True).item()
            if include_testdata:
                raw_data_test = np.load(
                    self.path.replace("train", "test"), allow_pickle=True
                ).item()
                self.raw_data["sequences"].update(raw_data_test["sequences"])
        elif self.mode == "test":
            self.raw_data = np.load(
                self.path.replace("train", "test"), allow_pickle=True
            ).item()
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def _restrict_keypoints(self, keypoints):
        """
        Restrict keypoints array to use_bodyparts subset.
        
        Args:
            keypoints: Array of shape (n_frames, n_individuals, n_all_bodyparts, 2)
            
        Returns:
            Array of shape (n_frames, n_individuals, n_use_bodyparts, 2)
        """
        # Extract only the keypoints we want to use
        return keypoints[:, :, self.bodypart_indices, :]

    def featurise_keypoints(self, keypoints):
        if self.centeralign:
            keypoints = keypoints.reshape(self.max_keypoints_len, *self.KEYFRAME_SHAPE)
            keypoints = self.transform_to_centeralign_components(keypoints)
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        return keypoints

    def scale_subsequence(self, sequence: np.ndarray, seq_idx: int) -> np.ndarray:
        """
        Scale sequence by its pre-computed body length scale.
        
        Args:
            sequence: Keypoints array [num_frames, features] after augmentation
            seq_idx: Index of the source sequence
            
        Returns:
            Scaled sequence
        """
        if self.sequence_scales is not None and seq_idx < len(self.sequence_scales):
            scale = self.sequence_scales[seq_idx]
            if scale > 0:  # Avoid division by zero
                sequence = sequence / scale
        return sequence

    def preprocess(self):
        """
        Does initial preprocessing on entire dataset.
        Computes per-sequence scale based on median body length.
        """
        self.check_annotations()

        sequences = self.raw_data["sequences"]

        seq_keypoints = []
        keypoints_ids = []
        sequence_scales = []
        sub_seq_length = self.max_keypoints_len
        sliding_window = self.sliding_window
        
        # Get nose and tail_base indices for scale computation
        nose_idx = self.BODY_PART_2_INDEX["nose"]
        tail_idx = self.BODY_PART_2_INDEX["tail_base"]

        for seq_ix, (seq_name, sequence) in enumerate(sequences.items()):
            vec_seq = sequence["keypoints"]
            # Restrict to STR_BODY_PARTS subset
            vec_seq = self._restrict_keypoints(vec_seq)
            
            # Compute scale for this sequence based on body length
            # vec_seq shape: (n_frames, n_individuals, n_keypoints, 2)
            nose = vec_seq[:, 0, nose_idx, :]  # (T, 2)
            tail = vec_seq[:, 0, tail_idx, :]  # (T, 2)
            body_length = np.linalg.norm(nose - tail, axis=1)  # (T,)
            scale = np.median(body_length)  # robust scalar
            sequence_scales.append(scale)
            
            if self.interp_holes:
                vec_seq = self.fill_holes(vec_seq)
            # Preprocess sequences
            vec_seq = vec_seq.reshape(vec_seq.shape[0], -1)

            if self._sampling_rate > 1:
                vec_seq = vec_seq[:: self._sampling_rate]

            # Pads the beginning and end of the sequence with duplicate frames
            if sub_seq_length < 120:
                pad_length = sub_seq_length
            else:
                pad_length = 120
            pad_vec = np.pad(
                vec_seq,
                ((pad_length // 2, pad_length - 1 - pad_length // 2), (0, 0)),
                mode="edge",
            )

            seq_keypoints.append(pad_vec)

            keypoints_ids.extend(
                [
                    (seq_ix, i)
                    for i in np.arange(
                        0, len(pad_vec) - sub_seq_length + 1, sliding_window
                    )
                ]
            )

        seq_keypoints = np.array(seq_keypoints, dtype=np.float32)
        self.sequence_scales = np.array(sequence_scales, dtype=np.float32)

        self.items = list(np.arange(len(keypoints_ids)))

        self.seq_keypoints = seq_keypoints
        self.keypoints_ids = keypoints_ids
        self.n_frames = len(self.keypoints_ids)

        del self.raw_data
