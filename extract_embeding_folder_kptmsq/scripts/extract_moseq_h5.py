#!/usr/bin/env python3
"""
Per-frame encoder embedding extraction for hBehaveMAE using Keypoint-MoSeq data.
Saves outputs as a SINGLE .h5 file per model, with internal groups for each video.
"""

import argparse
import sys
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import h5py
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.kptmoseq_data import KeypointMoSeqDataset
from models.models_defs import hbehavemae
from models.general_hiera import GeneralizedHiera

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True, help="Path to model checkpoint (.pth)")
    p.add_argument("--data", type=Path, required=True, help="Path to the .pkl data file")
    p.add_argument("--keypoint-names", nargs="+", type=str, required=True, 
                   help="List of keypoints expected by the model")
    p.add_argument("--output-base", type=Path, default=Path("./data"),
                   help="Base folder where the <split>/<model>.h5 structure will be created.")
    p.add_argument("--model-name", type=str, required=True, 
                   help="Name of the model config (used as the .h5 filename)")
    p.add_argument("--scale-target", type=float, default=25.0)
    p.add_argument("--window-stride", type=int, default=300,
                   help="Stride between windows. Overlapping frames are averaged.")
    p.add_argument("--no-overlap", action="store_true",
                   help="Set stride = window_size (no overlap, no averaging)")
    return p.parse_args()

def get_window_size(model) -> int:
    return model.tokens_spatial_shape[0] * model.patch_stride[0]

def get_encoder_intermediates(model, x: torch.Tensor) -> List[torch.Tensor]:
    all_visible = torch.ones(
        x.shape[0], np.prod(model.mask_spatial_shape),
        device=x.device, dtype=torch.bool
    )
    _, intermediates = GeneralizedHiera.forward(
        model, x, mask=all_visible, return_intermediates=True
    )
    return intermediates

def tokens_to_frames(tokens: np.ndarray, window_frames: int) -> np.ndarray:
    num_mus, mu_t, mu_h, mu_w, D = tokens.shape
    T_tokens = num_mus * mu_t
    tokens_spatial_mean = tokens.mean(axis=(2, 3))
    tokens_temporal = tokens_spatial_mean.reshape(T_tokens, D)
    assert window_frames % T_tokens == 0
    frames_per_token = window_frames // T_tokens
    return np.repeat(tokens_temporal, frames_per_token, axis=0)

def extract_embeddings(
    model, features: np.ndarray, window_stride: int, device: torch.device,
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

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup the output directory and single .h5 file path
    split_name = args.data.stem
    out_dir = args.output_base / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    h5_path = out_dir / f"{args.model_name}.h5"
    print(f"Outputs will be saved to a single file: {h5_path}")
    
    # 2. Load model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = hbehavemae(**vars(ckpt["args"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    
    window_size = get_window_size(model)
    window_stride = window_size if args.no_overlap else args.window_stride
    
    # 3. Instantiate dataset (purely for preprocessing)
    dataset = KeypointMoSeqDataset(
        data_path=args.data,
        keypoint_names=args.keypoint_names,
        num_frames=window_size,
        sliding_window=window_stride,
        scale_target=args.scale_target,
        return_likelihoods=False
    )
    
    with open(args.data, "rb") as f:
        raw_data = pickle.load(f)
    video_names = list(raw_data[0].keys())
    
    # 4. Extract and write directly to the single HDF5 file
    with h5py.File(h5_path, 'w') as h5_file:
        # Save global metadata as file-level attributes
        h5_file.attrs["model_name"] = args.model_name
        h5_file.attrs["window_size"] = window_size
        h5_file.attrs["window_stride"] = window_stride
        h5_file.attrs["total_videos"] = len(video_names)
        
        for i, name in enumerate(tqdm(video_names, desc="Extracting Videos")):
            kpts_filled = dataset.video_keypoints[i]
            scale = dataset.video_scales[i]
            features = dataset._featurize(kpts_filled, scale)
            
            if features.shape[-1] > model.feature_dim:
                features = features[..., :model.feature_dim]
                
            embeddings = extract_embeddings(model, features, window_stride, device)
            
            # Create a group (like a folder) for this specific video
            video_group = h5_file.create_group(name)
            
            # Save video-specific metadata
            video_group.attrs["num_frames"] = int(features.shape[0])
            video_group.attrs["scale_factor_used"] = scale
            
            # Save the embeddings inside the video's group
            for stage_name, emb_data in embeddings.items():
                video_group.create_dataset(
                    name=stage_name, 
                    data=emb_data, 
                    compression="gzip", 
                    compression_opts=4
                )

if __name__ == "__main__":
    main()