"""
Keypoint-level augmentation for denoising MAE training.

Applies LOCAL per-frame perturbations (rotation, noise, scale) to keypoints
BEFORE featurization, enabling the model to learn to denoise corrupted inputs
rather than invert global geometric transforms.

Shape handling:
- Input:  numpy (T, I, K, 2)
- Output: numpy (T, I, K, 2)
"""

import numpy as np
import torch

from .simple_augment import SimpleAugment


class KeypointAugment:
    """
    Wrapper for applying per-frame keypoint augmentations.
    
    Converts numpy keypoints to torch, applies SimpleAugment (which does
    per-frame rotation/noise/scale), then converts back to numpy.
    
    Args:
        rotation_prob: Probability of applying rotation per frame
        noise_prob: Probability of applying noise per frame
        scale_prob: Probability of applying scale per frame
        min_rot_deg: Minimum rotation angle in degrees
        noise_var: Variance of Gaussian noise
        scale_factor: Scale multiplication factor
    """
    
    def __init__(
        self,
        rotation_prob: float = 0.3,
        noise_prob: float = 0.3,
        scale_prob: float = 0.3,
        min_rot_deg: float = 5.0,
        noise_var: float = 4.0,
        scale_factor: float = 1.05,
    ):
        self.augment = SimpleAugment(
            rotation_prob=rotation_prob,
            noise_prob=noise_prob,
            scale_prob=scale_prob,
            min_rot_deg=min_rot_deg,
            noise_var=noise_var,
            scale_factor=scale_factor,
        )
    
    def __call__(self, keypoints_np: np.ndarray) -> np.ndarray:
        """
        Apply per-frame augmentations to keypoints.
        
        Args:
            keypoints_np: (T, I, K, 2) numpy array
            
        Returns:
            Augmented keypoints with same shape (T, I, K, 2)
        """
        # Shape validation
        assert keypoints_np.ndim == 4, f"Expected 4D input, got {keypoints_np.ndim}D"
        assert keypoints_np.shape[-1] == 2, f"Expected last dim=2 (x,y), got {keypoints_np.shape[-1]}"
        
        original_shape = keypoints_np.shape
        T, I, K, _ = original_shape
        
        # Convert to torch tensor
        keypoints_torch = torch.from_numpy(keypoints_np).float()
        
        # Apply augmentation (SimpleAugment expects (T, I, K, 2))
        augmented_torch = self.augment(keypoints_torch)
        
        # Convert back to numpy
        augmented_np = augmented_torch.numpy()
        
        # Validate output shape
        assert augmented_np.shape == original_shape, (
            f"Shape mismatch: input {original_shape}, output {augmented_np.shape}"
        )
        
        return augmented_np
