#!/usr/bin/env python3
"""
Minimal HBehaveMAE Reconstruction Script.
- Uses PoseReconstructionDataset as single source of truth
- Hard mask (conf <= 0.01) + stochastic mask (10 passes, median)
- Stitches windows and inverse transforms to pixel coordinates
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

# ============================================================================
# Keypoints (hardcoded from training)
# ============================================================================
ALL_KEYPOINTS = [
    "nose", "left_ear", "right_ear", "left_ear_tip", "right_ear_tip",
    "left_eye", "right_eye", "neck", "mid_back", "mouse_center",
    "mid_backend", "mid_backend2", "mid_backend3", "tail_base",
    "tail1", "tail2", "tail3", "tail4", "tail5",
    "left_shoulder", "left_midside", "left_hip",
    "right_shoulder", "right_midside", "right_hip",
    "tail_end", "head_midpoint",
]



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seq_idxs", type=str, default="all")
    p.add_argument("--num_frames", type=int, default=900)
    p.add_argument("--sliding_window", type=int, default=300)
    p.add_argument("--data_augment", type=bool, default=True)
    p.add_argument("--return_likelihoods", type=bool, default=True)
    p.add_argument("--num_passes", type=int, default=10)
    p.add_argument("--target_visible_frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ============================================================================
# MASKING
# ============================================================================

def conf_to_hard_mask(conf: np.ndarray, num_mask_units: int, num_frames: int) -> np.ndarray:
    """
    Map frame-level confidence to mask-unit level hard mask.
    
    Args:
        conf: (T, F) confidence per frame/feature
        num_mask_units: number of mask units from model
        num_frames: window length
    
    Returns:
        (num_mask_units,) bool array. True = reliable, False = hard-masked (conf <= 0.01 or NaN)
    """
    frames_per_unit = max(1, num_frames // num_mask_units)
    unit_reliable = np.ones(num_mask_units, dtype=bool)
    
    for u in range(num_mask_units):
        start = u * frames_per_unit
        end = min(start + frames_per_unit, num_frames)
        if start >= num_frames:
            unit_reliable[u] = False
            continue
        
        segment = conf[start:end]
        # Mark as unreliable if ANY frame has low confidence or NaN
        if np.any(segment <= 0.01) or np.any(np.isnan(segment)):
            unit_reliable[u] = False
    
    return unit_reliable


def make_stochastic_masks(
    hard_mask: np.ndarray,
    target_visible_frac: float,
    num_passes: int,
    seed: int,
) -> list:
    """
    Generate random masks on top of hard mask.
    
    Convention: True = visible, False = masked
    (This matches forward_encoder/get_random_mask convention)
    """
    rng = np.random.default_rng(seed)
    num_units = len(hard_mask)
    target_visible = int(num_units * target_visible_frac)
    
    reliable_indices = np.where(hard_mask)[0]
    num_reliable = len(reliable_indices)
    
    masks = []
    for _ in range(num_passes):
        mask = np.zeros(num_units, dtype=bool)
        
        if num_reliable > 0:
            num_to_keep = min(target_visible, num_reliable)
            keep_indices = rng.choice(reliable_indices, size=num_to_keep, replace=False)
            mask[keep_indices] = True
        
        masks.append(mask)
    
    return masks


# ============================================================================
# MODEL
# ============================================================================

def load_model(ckpt_path: str, device: torch.device):
    """Load checkpoint and build model."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    ckpt_args = ckpt.get("args")
    
    if ckpt_args is None:
        raise ValueError("Checkpoint missing 'args'")
    
    from models.models_defs import hbehavemae
    model = hbehavemae(**vars(ckpt_args))
    
    # Handle DDP prefix
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    
    return model, ckpt_args


# ============================================================================
# RECONSTRUCTION
# ============================================================================

def reconstruct_window(model, x_features, hard_mask, num_passes, target_visible_frac, seed, device):
    """
    Reconstruct one window with multiple random masks, return median.
    
    Returns:
        pred_median: (num_tokens, pred_dim)
        stats: dict
    """
    x_5d = x_features.unsqueeze(1).to(device)  # (1, 1, T, I, F)
    
    masks = make_stochastic_masks(hard_mask, target_visible_frac, num_passes, seed)
    
    all_preds = []
    for mask_np in masks:
        # Convention: True = visible (matches model's get_random_mask)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)
        
        with torch.no_grad():
            latent, mask_out = model.forward_encoder(x_5d, mask_ratio=0.0, mask=mask_t)
            pred, _ = model.forward_decoder(latent, mask_out)
        
        all_preds.append(pred.cpu().numpy())
    
    # Median across passes
    all_preds = np.stack(all_preds, axis=0)  # (num_passes, 1, num_tokens, pred_dim)
    pred_median = np.median(all_preds, axis=0).squeeze(0)  # (num_tokens, pred_dim)
    
    stats = {
        "hard_masked_frac": 1.0 - hard_mask.mean(),
        "mean_visible_frac": np.mean([m.mean() for m in masks]),
    }
    
    return pred_median, stats


def stitch_windows(seq_len: int, num_features: int, window_preds: list, window_starts: list, window_len: int) -> np.ndarray:
    """
    Stitch overlapping window predictions using averaging.
    
    Args:
        seq_len: total sequence length
        num_features: feature dimension
        window_preds: list of (window_len, num_features) arrays
        window_starts: list of start indices
        window_len: window length
    
    Returns:
        (seq_len, num_features) stitched array
    """
    stitched = np.zeros((seq_len, num_features), dtype=np.float32)
    counts = np.zeros((seq_len, 1), dtype=np.float32)
    
    for pred, start in zip(window_preds, window_starts):
        end = min(start + window_len, seq_len)
        valid_len = end - start
        
        # pred might be shorter than window_len (edge case)
        pred_len = min(pred.shape[0], valid_len)
        
        stitched[start:start + pred_len] += pred[:pred_len]
        counts[start:start + pred_len] += 1.0
    
    # Avoid division by zero
    counts = np.maximum(counts, 1.0)
    return stitched / counts


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model from {args.ckpt}")
    model, ckpt_args = load_model(args.ckpt, device)
    
    F = model.feature_dim
    num_mask_units = math.prod(model.mask_spatial_shape)
    print(f"Model: feature_dim={F}, num_mask_units={num_mask_units}")
    
    # Load dataset
    from datasets.reconstruct_data import PoseReconstructionDataset
    
    dataset = PoseReconstructionDataset(
        mode="pretrain",
        data_path=args.dataset_path,
        keypoint_names=ALL_KEYPOINTS,
        all_keypoints=ALL_KEYPOINTS,
        center_keypoint="neck",
        align_keypoints=("tail_base", "neck"),
        scale_keypoints=("nose", "tail_base"),
        num_frames=args.num_frames,
        sliding_window=args.sliding_window,
        sampling_rate=1,
        fill_holes=True,
        centeralign=True,
        augmentations=False,
        data_augment=args.data_augment,
        return_likelihoods=args.return_likelihoods,
        nan_scattered_threshold=1.0,  # Disable filtering
        nan_concentrated_threshold=1.0,  # Disable filtering
    )
    
    num_seqs = len(dataset.seq_keypoints)
    print(f"Dataset: {num_seqs} sequences")
    
    # Parse sequence indices
    if args.seq_idxs.lower() == "all":
        seq_idxs = list(range(num_seqs))
    else:
        seq_idxs = [int(x.strip()) for x in args.seq_idxs.split(",")]
    
    print(f"Processing sequences: {seq_idxs}")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each sequence
    for seq_idx in seq_idxs:
        seq_name = f"seq_{seq_idx:04d}"
        print(f"\n=== {seq_name} ===")
        
        seq_out = out_dir / seq_name
        seq_out.mkdir(exist_ok=True)
        
        # Get raw data for this sequence
        seq_kpts = dataset.seq_keypoints[seq_idx]
        seq_conf = dataset.seq_confidences[seq_idx] if dataset.seq_confidences else None
        seq_len = len(seq_kpts)
        
        print(f"  Length: {seq_len}, window: {args.num_frames}, stride: {args.sliding_window}")
        
        window_preds = []
        window_starts = []
        all_stats = []
        
        # Sliding window
        for start in range(0, seq_len - args.num_frames + 1, args.sliding_window):
            kpts_win = seq_kpts[start:start + args.num_frames]
            conf_win = seq_conf[start:start + args.num_frames] if seq_conf is not None else None
            
            # Use dataset's prepare_sample for feature construction
            if conf_win is not None:
                sample = dataset.prepare_sample((kpts_win, conf_win), seq_idx=seq_idx)
            else:
                sample = dataset.prepare_sample(kpts_win, seq_idx=seq_idx)
            
            sample = sample.unsqueeze(0)  # (1, T, I, total_F)
            
            # Slice based on flags
            if args.data_augment and args.return_likelihoods:
                x_features = sample[..., F:2*F]
                x_conf = sample[..., 2*F:3*F]
            elif args.data_augment:
                x_features = sample[..., F:2*F]
                x_conf = None
            elif args.return_likelihoods:
                x_features = sample[..., 0:F]
                x_conf = sample[..., F:2*F]
            else:
                x_features = sample
                x_conf = None
            
            # Build hard mask from confidence
            if x_conf is not None:
                conf_np = x_conf.squeeze(0).squeeze(1).numpy()  # (T, F)
                hard_mask = conf_to_hard_mask(conf_np, num_mask_units, args.num_frames)
            else:
                hard_mask = np.ones(num_mask_units, dtype=bool)
            
            # Reconstruct
            pred, stats = reconstruct_window(
                model, x_features, hard_mask,
                args.num_passes, args.target_visible_frac,
                args.seed + start, device
            )
            
            # pred is (num_tokens, pred_dim) - need to reshape to (T_out, F)
            # The model predicts strided tokens. For now, store raw pred.
            window_preds.append(pred)
            window_starts.append(start)
            all_stats.append(stats)
            
            if start == 0:
                print(f"  Window pred shape: {pred.shape}")
        
        print(f"  Processed {len(window_preds)} windows")
        
        # Compute average stats
        mean_hard_frac = np.mean([s["hard_masked_frac"] for s in all_stats])
        mean_vis_frac = np.mean([s["mean_visible_frac"] for s in all_stats])
        print(f"  Hard masked: {mean_hard_frac:.1%}, Visible: {mean_vis_frac:.1%}")
        
        # === RESHAPE PREDICTIONS ===
        # Model output: (num_tokens, pred_stride * in_chans)
        # Need to reshape to (T, F) for stitching
        # pred_stride = temporal_patch_stride (for 1D pose data)
        pred_stride = model.pred_stride
        num_tokens = window_preds[0].shape[0]
        
        # Reshape each window prediction: (num_tokens, pred_stride) -> (num_tokens * pred_stride / F, F)
        # For pose data: pred_stride = temporal compression, so (num_tokens, pred_stride) -> (T_strided, F)
        # Actually the output is (num_tokens, pred_stride * in_chans) where in_chans=1
        # So it's (num_tokens, pred_stride) which reshapes to (num_tokens * pred_stride,) then to (T_strided, F)
        
        # The temporal stride means we only reconstruct every pred_stride-th frame
        T_strided = num_tokens  # Each token outputs pred_stride values flattened
        
        reshaped_preds = []
        for pred in window_preds:
            # pred: (num_tokens, pred_stride * in_chans) = (num_tokens, pred_stride)
            # Reshape to (num_tokens * pred_stride,) then to frame-feature format
            # But we need to match the feature dimension F
            pred_flat = pred.reshape(-1)  # Flatten
            
            # If pred_flat.size matches T_strided * F, reshape directly
            # Otherwise, this is strided output - need to handle carefully
            try:
                pred_frames = pred_flat.reshape(-1, F)  # (T_out, F)
                reshaped_preds.append(pred_frames)
            except ValueError:
                # Can't reshape cleanly - save raw predictions
                print(f"  Warning: Can't reshape pred {pred.shape} to frame format")
                reshaped_preds.append(pred)
        
        if len(reshaped_preds) > 0 and len(reshaped_preds[0].shape) == 2 and reshaped_preds[0].shape[1] == F:
            # Successfully reshaped - stitch predictions
            T_out_per_window = reshaped_preds[0].shape[0]
            
            # Calculate effective stride for strided predictions
            # The output is temporally subsampled, so window_starts need adjustment
            stride_ratio = args.num_frames / T_out_per_window
            adjusted_starts = [int(s / stride_ratio) for s in window_starts]
            adjusted_seq_len = int(seq_len / stride_ratio)
            
            stitched_pred = stitch_windows(adjusted_seq_len, F, reshaped_preds, adjusted_starts, T_out_per_window)
            
            # Inverse transform predictions
            keypoints_pred = dataset.inverse_transform(stitched_pred, seq_idx=seq_idx)
            print(f"  Reconstructed keypoints shape: {keypoints_pred.shape}")
        else:
            # Couldn't reshape - save raw
            stitched_pred = None
            keypoints_pred = None
            print(f"  Warning: Could not reshape predictions to feature space")
        
        # Stack raw window predictions for debugging
        window_preds_arr = np.stack(window_preds, axis=0)
        
        # Also get the input features (stitched) for comparison
        input_features = []
        for start in range(0, seq_len - args.num_frames + 1, args.sliding_window):
            kpts_win = seq_kpts[start:start + args.num_frames]
            conf_win = seq_conf[start:start + args.num_frames] if seq_conf is not None else None
            
            if conf_win is not None:
                sample = dataset.prepare_sample((kpts_win, conf_win), seq_idx=seq_idx)
            else:
                sample = dataset.prepare_sample(kpts_win, seq_idx=seq_idx)
            
            # Get raw features
            if args.data_augment:
                feat = sample[..., 0:F].squeeze(1).numpy()
            else:
                feat = sample.squeeze(1).numpy() if not args.return_likelihoods else sample[..., 0:F].squeeze(1).numpy()
            
            input_features.append(feat)
        
        # Stitch input features
        stitched_input = stitch_windows(seq_len, F, input_features, window_starts, args.num_frames)
        
        # Inverse transform input to get ground truth keypoints
        keypoints_input = dataset.inverse_transform(stitched_input, seq_idx=seq_idx)
        
        # Save
        save_dict = {
            # Raw window predictions (in model's output space)
            "window_preds": window_preds_arr,
            "window_starts": np.array(window_starts),
            # Input (ground truth)
            "stitched_input_features": stitched_input,
            "keypoints_input": keypoints_input,
            # Metadata
            "seq_idx": seq_idx,
            "seq_name": seq_name,
            "num_passes": args.num_passes,
            "hard_masked_frac": mean_hard_frac,
            "mean_visible_frac": mean_vis_frac,
        }
        
        # Add reconstructed keypoints if available
        if keypoints_pred is not None:
            save_dict["keypoints_pred"] = keypoints_pred
            save_dict["stitched_pred_features"] = stitched_pred
        
        np.savez(seq_out / "reconstruction.npz", **save_dict)
        
        print(f"  Saved to {seq_out}")
        print(f"  Input keypoints shape: {keypoints_input.shape}")
        if keypoints_pred is not None:
            print(f"  Pred keypoints shape: {keypoints_pred.shape}")
    
    print(f"\nDone! Results in {out_dir}")


if __name__ == "__main__":
    main()