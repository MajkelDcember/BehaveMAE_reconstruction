# #!/usr/bin/env python3
# """
# Per-frame encoder embedding extraction for hBehaveMAE.

# Usage:
#     python extract_embeddings.py --ckpt checkpoint.pth --data raw_data.npy --output embeddings/
#     python extract_embeddings.py --ckpt checkpoint.pth --data raw_data.npy --output embeddings/ --strategy multi --sliding-window 300
# """

# import argparse
# import json
# import sys
# from pathlib import Path
# from typing import Dict, List

# import numpy as np
# import torch

# # Add parent directory to path for imports (same as reconstruction script)
# sys.path.insert(0, str(Path(__file__).parent.parent))

# from datasets.reconstruct_data2_utils import (
#     PoseGeometryConfig,
#     restrict_keypoints,
#     restrict_confidences,
#     fill_holes,
#     compute_sequence_scale,
#     featurize_full_sequence,
# )

# from models.models_defs import hbehavemae
# from models.general_hiera import GeneralizedHiera


# # -----------------------------------------------------------------------------
# # Constants
# # -----------------------------------------------------------------------------

# ALL_KEYPOINTS = [
#     "nose", "left_ear", "right_ear", "left_ear_tip", "right_ear_tip",
#     "left_eye", "right_eye", "neck", "mid_back", "mouse_center",
#     "mid_backend", "mid_backend2", "mid_backend3", "tail_base",
#     "tail1", "tail2", "tail3", "tail4", "tail5",
#     "left_shoulder", "left_midside", "left_hip",
#     "right_shoulder", "right_midside", "right_hip",
#     "tail_end", "head_midpoint",
# ]


# # -----------------------------------------------------------------------------
# # CLI
# # -----------------------------------------------------------------------------

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--ckpt", type=Path, required=True)
#     p.add_argument("--data", type=Path, required=True)
#     p.add_argument("--output", type=Path, default=Path("embedding_output"))
#     p.add_argument("--video-indices", type=int, nargs="+", default=None)
#     p.add_argument("--strategy", choices=["single", "multi"], default="multi")
#     p.add_argument("--sliding-window", type=int, default=300)
#     p.add_argument("--concat", action="store_true")
#     return p.parse_args()


# # -----------------------------------------------------------------------------
# # Model
# # -----------------------------------------------------------------------------

# def get_window_size(model) -> int:
#     return model.tokens_spatial_shape[0] * model.patch_stride[0]


# def get_encoder_intermediates(model, x: torch.Tensor) -> List[torch.Tensor]:
#     """
#     Run encoder with all tokens visible, return intermediates at each stage.
#     """
#     all_visible = torch.ones(
#         x.shape[0], np.prod(model.mask_spatial_shape),
#         device=x.device, dtype=torch.bool
#     )
    
#     _, intermediates = GeneralizedHiera.forward(
#         model, x, mask=all_visible, return_intermediates=True
#     )
#     return intermediates


# # -----------------------------------------------------------------------------
# # Data Loading (same as reconstruction script)
# # -----------------------------------------------------------------------------

# def load_and_featurize_sequence(seq_data: dict, geo_cfg) -> tuple:
#     keypoint_indices = list(range(len(ALL_KEYPOINTS)))
    
#     kpts_raw = restrict_keypoints(seq_data["keypoints"], keypoint_indices)
    
#     if "confidences" in seq_data:
#         conf_raw = restrict_confidences(seq_data["confidences"], keypoint_indices)
#     else:
#         conf_raw = np.ones(kpts_raw.shape[:-1], dtype=np.float32)
#         conf_raw[np.any(np.isnan(kpts_raw), axis=-1)] = 0.0
    
#     nan_mask = np.any(np.isnan(kpts_raw), axis=-1)
#     conf_raw[nan_mask] = 0.0
    
#     kpts_filled = fill_holes(kpts_raw)
#     scale = compute_sequence_scale(kpts_raw, geo_cfg)
    
#     feat = featurize_full_sequence(kpts_filled, geo_cfg)
#     feat[:, 4:] /= scale
    
#     return feat, conf_raw, scale


# # -----------------------------------------------------------------------------
# # Embedding Extraction
# # -----------------------------------------------------------------------------

# def tokens_to_frames(tokens: np.ndarray, window_frames: int) -> np.ndarray:
#     T_tokens = tokens.shape[0]
#     D = tokens.shape[-1]
    
#     assert window_frames % T_tokens == 0, \
#         f"window_frames ({window_frames}) must be divisible by T_tokens ({T_tokens})"
    
#     frames_per_token = window_frames // T_tokens
#     tokens_temporal = tokens.reshape(T_tokens, -1, D).mean(axis=1)
#     return np.repeat(tokens_temporal, frames_per_token, axis=0)


# def extract_embeddings(
#     model,
#     features: np.ndarray,
#     window_stride: int,
#     device: torch.device,
# ) -> Dict[str, np.ndarray]:
    
#     T_total = features.shape[0]
#     window_size = get_window_size(model)
#     num_stages = len(model.stage_ends)
    
#     embed_dims = [model.projections[i].out_features for i in range(num_stages)]
#     accumulators = [np.zeros((T_total, d), dtype=np.float32) for d in embed_dims]
#     counts = np.zeros(T_total, dtype=np.float32)
    
#     starts = list(range(0, T_total - window_size + 1, window_stride))
#     if len(starts) == 0 or starts[-1] + window_size < T_total:
#         starts.append(max(0, T_total - window_size))
    
#     print(f"  Processing {len(starts)} windows (size={window_size}, stride={window_stride})...")
    
#     for start in starts:
#         end = start + window_size
#         win = features[start:end]
        
#         x = torch.from_numpy(win).float().view(1, 1, window_size, 1, -1).to(device)
        
#         with torch.no_grad():
#             intermediates = get_encoder_intermediates(model, x)
        
#         for stage_idx, stage_out in enumerate(intermediates):
#             tokens = stage_out[0].cpu().numpy()
#             frame_emb = tokens_to_frames(tokens, window_size)
#             accumulators[stage_idx][start:end] += frame_emb
        
#         counts[start:end] += 1
    
#     for acc in accumulators:
#         acc /= counts[:, np.newaxis]
    
#     return {f"stage{i+1}": acc for i, acc in enumerate(accumulators)}


# # -----------------------------------------------------------------------------
# # Main
# # -----------------------------------------------------------------------------

# def main():
#     args = parse_args()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Load model
#     ckpt = torch.load(args.ckpt, map_location="cpu")
#     model = hbehavemae(**vars(ckpt["args"]))
#     model.load_state_dict(ckpt["model"])
#     model.to(device).eval()
    
#     window_size = get_window_size(model)
#     print(f"Model window_size: {window_size}")
    
#     if args.strategy == "single":
#         window_stride = window_size
#     else:
#         window_stride = args.sliding_window if args.sliding_window else window_size // 3
    
#     # Setup geometry config
#     geo_cfg = PoseGeometryConfig(
#         keypoint_names=tuple(ALL_KEYPOINTS),
#         center_keypoint="neck",
#         align_keypoints=("tail_base", "neck"),
#         scale_keypoints=("nose", "tail_base"),
#         grid_size=500,
#         centeralign=True,
#     )
    
#     # Load raw data
#     raw = np.load(args.data, allow_pickle=True).item()
#     names = list(raw["sequences"].keys())
#     indices = args.video_indices or range(len(names))
    
#     args.output.mkdir(parents=True, exist_ok=True)
    
#     for i in indices:
#         name = names[i]
#         seq = raw["sequences"][name]
        
#         print(f"[{i}/{len(indices)}] Processing {name}...")
        
#         # Featurize (same as reconstruction script)
#         feat, conf, scale = load_and_featurize_sequence(seq, geo_cfg)
        
#         # Truncate to model feature_dim if needed
#         if feat.shape[-1] > model.feature_dim:
#             feat = feat[..., :model.feature_dim]
        
#         # Extract embeddings
#         embeddings = extract_embeddings(model, feat, window_stride, device)
        
#         # Save
#         out_dir = args.output / name
#         out_dir.mkdir(parents=True, exist_ok=True)
        
#         if args.concat:
#             all_emb = np.concatenate(list(embeddings.values()), axis=1)
#             np.save(out_dir / "embeddings.npy", all_emb)
#         else:
#             np.savez(out_dir / "embeddings.npz", **embeddings)
        
#         # Save metadata
#         meta = {
#             "num_frames": int(feat.shape[0]),
#             "window_size": window_size,
#             "window_stride": window_stride,
#             "strategy": args.strategy,
#             "stages": {k: list(v.shape) for k, v in embeddings.items()},
#         }
#         with open(out_dir / "meta.json", "w") as f:
#             json.dump(meta, f, indent=2)
        
#         print(f"  Saved to {out_dir}")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Per-frame encoder embedding extraction for hBehaveMAE.

Extracts all 3 stage embeddings (stage1, stage2, stage3) for every frame.
Uses sliding windows with overlap averaging by default.

Usage:
    # Default: stride=300, overlap averaging
    python extract_embeddings.py --ckpt checkpoint.pth --data raw_data.npy --output embeddings/
    
    # Custom stride
    python extract_embeddings.py --ckpt checkpoint.pth --data raw_data.npy --output embeddings/ --window-stride 450
    
    # No overlap (stride = window_size)
    python extract_embeddings.py --ckpt checkpoint.pth --data raw_data.npy --output embeddings/ --no-overlap
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Add parent directory to path for imports (same as reconstruction script)
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.reconstruct_data2_utils import (
    PoseGeometryConfig,
    restrict_keypoints,
    restrict_confidences,
    fill_holes,
    compute_sequence_scale,
    featurize_full_sequence,
)

from models.models_defs import hbehavemae
from models.general_hiera import GeneralizedHiera


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

ALL_KEYPOINTS = [
    "nose", "left_ear", "right_ear", "left_ear_tip", "right_ear_tip",
    "left_eye", "right_eye", "neck", "mid_back", "mouse_center",
    "mid_backend", "mid_backend2", "mid_backend3", "tail_base",
    "tail1", "tail2", "tail3", "tail4", "tail5",
    "left_shoulder", "left_midside", "left_hip",
    "right_shoulder", "right_midside", "right_hip",
    "tail_end", "head_midpoint",
]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("embedding_output"))
    p.add_argument("--video-indices", type=int, nargs="+", default=None)
    p.add_argument("--window-stride", type=int, default=300,
                   help="Stride between windows. If < window_size (900), overlapping frames are averaged.")
    p.add_argument("--no-overlap", action="store_true",
                   help="Set stride = window_size (no overlap, no averaging)")
    p.add_argument("--concat", action="store_true",
                   help="Save as single .npy with stages concatenated instead of .npz with separate stages")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

def get_window_size(model) -> int:
    return model.tokens_spatial_shape[0] * model.patch_stride[0]


def get_encoder_intermediates(model, x: torch.Tensor) -> List[torch.Tensor]:
    """
    Run encoder with all tokens visible, return intermediates at each stage.
    """
    all_visible = torch.ones(
        x.shape[0], np.prod(model.mask_spatial_shape),
        device=x.device, dtype=torch.bool
    )
    
    _, intermediates = GeneralizedHiera.forward(
        model, x, mask=all_visible, return_intermediates=True
    )
    return intermediates


# -----------------------------------------------------------------------------
# Data Loading (same as reconstruction script)
# -----------------------------------------------------------------------------

def load_and_featurize_sequence(seq_data: dict, geo_cfg) -> tuple:
    keypoint_indices = list(range(len(ALL_KEYPOINTS)))
    
    kpts_raw = restrict_keypoints(seq_data["keypoints"], keypoint_indices)
    
    if "confidences" in seq_data:
        conf_raw = restrict_confidences(seq_data["confidences"], keypoint_indices)
    else:
        conf_raw = np.ones(kpts_raw.shape[:-1], dtype=np.float32)
        conf_raw[np.any(np.isnan(kpts_raw), axis=-1)] = 0.0
    
    nan_mask = np.any(np.isnan(kpts_raw), axis=-1)
    conf_raw[nan_mask] = 0.0
    
    kpts_filled = fill_holes(kpts_raw)
    scale = compute_sequence_scale(kpts_raw, geo_cfg)
    
    feat = featurize_full_sequence(kpts_filled, geo_cfg)
    feat[:, 4:] /= scale
    
    return feat, conf_raw, scale


# -----------------------------------------------------------------------------
# Embedding Extraction
# -----------------------------------------------------------------------------

def tokens_to_frames(tokens: np.ndarray, window_frames: int) -> np.ndarray:
    """
    Convert token-level embeddings to frame-level by replication.
    
    tokens shape: (num_mask_units, mu_t, mu_h, mu_w, D)
    Output shape: (window_frames, D)
    
    The temporal structure is: num_mask_units * mu_t tokens along time.
    Each token covers window_frames / (num_mask_units * mu_t) frames.
    """
    # tokens: (num_MUs, MU_t, MU_h, MU_w, D)
    num_mus, mu_t, mu_h, mu_w, D = tokens.shape
    
    # Total temporal tokens
    T_tokens = num_mus * mu_t
    
    # Reshape to (T_tokens, D) by merging mask units with their temporal dim
    # and averaging over spatial dims (mu_h, mu_w)
    # Shape: (num_MUs, mu_t, mu_h, mu_w, D) -> (num_MUs, mu_t, D)
    tokens_spatial_mean = tokens.mean(axis=(2, 3))  # (num_MUs, mu_t, D)
    
    # Reshape to (T_tokens, D) - interleave mask units properly
    # The mask units tile the temporal dimension, so we need to reshape correctly
    # tokens_spatial_mean[i, j] corresponds to mask_unit i, temporal position j within that unit
    # The full temporal order is: all mu_t positions of MU0, then all of MU1, etc.
    tokens_temporal = tokens_spatial_mean.reshape(T_tokens, D)
    
    assert window_frames % T_tokens == 0, \
        f"window_frames ({window_frames}) must be divisible by T_tokens ({T_tokens})"
    
    frames_per_token = window_frames // T_tokens
    return np.repeat(tokens_temporal, frames_per_token, axis=0)


def extract_embeddings(
    model,
    features: np.ndarray,
    window_stride: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    
    T_total = features.shape[0]
    window_size = get_window_size(model)
    num_stages = len(model.stage_ends)
    
    embed_dims = [model.projections[i].out_features for i in range(num_stages)]
    accumulators = [np.zeros((T_total, d), dtype=np.float32) for d in embed_dims]
    counts = np.zeros(T_total, dtype=np.float32)
    
    starts = list(range(0, T_total - window_size + 1, window_stride))
    if len(starts) == 0 or starts[-1] + window_size < T_total:
        starts.append(max(0, T_total - window_size))
    
    print(f"  Processing {len(starts)} windows (size={window_size}, stride={window_stride})...")
    
    for start in starts:
        end = start + window_size
        win = features[start:end]
        
        x = torch.from_numpy(win).float().view(1, 1, window_size, 1, -1).to(device)
        
        with torch.no_grad():
            intermediates = get_encoder_intermediates(model, x)
        
        for stage_idx, stage_out in enumerate(intermediates):
            tokens = stage_out[0].cpu().numpy()
            frame_emb = tokens_to_frames(tokens, window_size)
            accumulators[stage_idx][start:end] += frame_emb
        
        counts[start:end] += 1
    
    for acc in accumulators:
        acc /= counts[:, np.newaxis]
    
    return {f"stage{i+1}": acc for i, acc in enumerate(accumulators)}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = hbehavemae(**vars(ckpt["args"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    
    window_size = get_window_size(model)
    print(f"Model window_size: {window_size}")
    
    if args.no_overlap:
        window_stride = window_size
    else:
        window_stride = args.window_stride
    
    # Setup geometry config
    geo_cfg = PoseGeometryConfig(
        keypoint_names=tuple(ALL_KEYPOINTS),
        center_keypoint="neck",
        align_keypoints=("tail_base", "neck"),
        scale_keypoints=("nose", "tail_base"),
        grid_size=500,
        centeralign=True,
    )
    
    # Load raw data
    raw = np.load(args.data, allow_pickle=True).item()
    names = list(raw["sequences"].keys())
    indices = args.video_indices or range(len(names))
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    for i in indices:
        name = names[i]
        seq = raw["sequences"][name]
        
        print(f"[{i}/{len(indices)}] Processing {name}...")
        
        # Featurize (same as reconstruction script)
        feat, conf, scale = load_and_featurize_sequence(seq, geo_cfg)
        
        # Truncate to model feature_dim if needed
        if feat.shape[-1] > model.feature_dim:
            feat = feat[..., :model.feature_dim]
        
        # Extract embeddings
        embeddings = extract_embeddings(model, feat, window_stride, device)
        
        # Save
        out_dir = args.output / name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if args.concat:
            all_emb = np.concatenate(list(embeddings.values()), axis=1)
            np.save(out_dir / "embeddings.npy", all_emb)
        else:
            np.savez(out_dir / "embeddings.npz", **embeddings)
        
        # Save metadata
        meta = {
            "num_frames": int(feat.shape[0]),
            "window_size": window_size,
            "window_stride": window_stride,
            "stages": {k: list(v.shape) for k, v in embeddings.items()},
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        print(f"  Saved to {out_dir}")


if __name__ == "__main__":
    main()