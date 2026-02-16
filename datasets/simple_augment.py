import torch
import math


class SimpleAugment:
    def __init__(
        self,
        rotation_prob=0.3,
        noise_prob=0.3,
        scale_prob=0.3,
        min_rot_deg=5.0,
        noise_var=4.0,
        scale_factor=1.05,
    ):
        self.rotation_prob = rotation_prob
        self.noise_prob = noise_prob
        self.scale_prob = scale_prob
        self.min_rot_deg = min_rot_deg
        self.noise_var = noise_var
        self.scale_factor = scale_factor

    def __call__(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        keypoints: (T, I, K, 2)
        """
        T, I, K, _ = keypoints.shape
        coords = keypoints.reshape(T * I, K, 2)

        device = coords.device
        dtype = coords.dtype

        num_frames = coords.shape[0]

        rot_mask = torch.rand(num_frames, 1, device=device) < self.rotation_prob
        noise_mask = torch.rand(num_frames, 1, device=device) < self.noise_prob
        scale_mask = torch.rand(num_frames, 1, device=device) < self.scale_prob

        # ---------------- Rotation ----------------
        if self.rotation_prob > 0:
            angles_deg = (
                torch.rand(num_frames, 1, device=device, dtype=dtype)
                * (180.0 - self.min_rot_deg)
                + self.min_rot_deg
            )
            signs = torch.where(
                torch.rand(num_frames, 1, device=device) < 0.5,
                -1.0,
                1.0,
            ).to(dtype)
            angles_rad = angles_deg * signs * math.pi / 180.0

            cos_a = torch.cos(angles_rad)
            sin_a = torch.sin(angles_rad)

            center = coords.mean(dim=1, keepdim=True)

            x_c = coords[..., 0] - center[..., 0]
            y_c = coords[..., 1] - center[..., 1]

            x_r = cos_a * x_c - sin_a * y_c + center[..., 0]
            y_r = sin_a * x_c + cos_a * y_c + center[..., 1]

            coords[..., 0] = torch.where(rot_mask, x_r, coords[..., 0])
            coords[..., 1] = torch.where(rot_mask, y_r, coords[..., 1])

        # ---------------- Noise ----------------
        if self.noise_prob > 0:
            noise_std = math.sqrt(self.noise_var)
            noise = torch.randn_like(coords) * noise_std
            coords = torch.where(noise_mask.unsqueeze(-1), coords + noise, coords)

        # ---------------- Scale ----------------
        if self.scale_prob > 0:
            center = coords.mean(dim=1, keepdim=True)
            coords_scaled = (coords - center) * self.scale_factor + center
            coords = torch.where(scale_mask.unsqueeze(-1), coords_scaled, coords)

        return coords.reshape(T, I, K, 2)
