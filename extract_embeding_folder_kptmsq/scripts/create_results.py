#!/usr/bin/env python3

import h5py
import numpy as np
import pickle
from tqdm import tqdm

# --- Configuration ---
PKL_PATH = "/scratch/michal/projects/dvc_ofd_2025/code/ofdpipe/tmp/data/interim/keypoint_moseq_data/shuffle-3/shuffle-3_split-train.pkl"
GMM_H5_PATH = "/scratch/michal/projects/dvc_ofd_2025/code/BehaveMAE_reconstruction/extract_embeding_folder_kptmsq/gmm_40_clustering_outputs/gmm_labels_tailhip_40.h5"
EMBEDDINGS_H5_PATH = "/scratch/michal/projects/dvc_ofd_2025/code/BehaveMAE_reconstruction/extract_embeding_folder_kptmsq/data/shuffle-3_split-train/ofd_tailhip_20260226-205043.h5"
OUTPUT_H5_PATH = "/scratch/michal/projects/dvc_ofd_2025/code/BehaveMAE_reconstruction/extract_embeding_folder_kptmsq/gmm_40_clustering_outputs/results_40_clusters.h5"

STAGE = "stage2"
K_CLUSTERS = 40

def fill_holes_simple(keypoints):
    """Linear interpolation to fill NaNs."""
    res = keypoints.copy()
    T = res.shape[0]
    x = np.arange(T)
    for k in range(res.shape[1]):
        for d in range(res.shape[2]):
            y = res[:, k, d]
            mask = np.isnan(y)
            if not np.any(mask): continue
            if np.all(mask):
                res[:, k, d] = 0.0
                continue
            res[:, k, d] = np.interp(x, x[~mask], y[~mask])
    return res

def main():
    print("Loading raw coordinates from pickle...")
    with open(PKL_PATH, "rb") as f:
        raw_data = pickle.load(f)
    keypoints_dict = raw_data[0]
    keypoint_names = list(raw_data[2])

    c_idx = keypoint_names.index("mouse_center")
    n_idx = keypoint_names.index("nose")
    t_idx = keypoint_names.index("tail_base")

    print(f"Creating KPMS-compatible results file:\n{OUTPUT_H5_PATH}")
    
    with h5py.File(GMM_H5_PATH, 'r') as f_gmm, \
         h5py.File(EMBEDDINGS_H5_PATH, 'r') as f_emb, \
         h5py.File(OUTPUT_H5_PATH, 'w') as f_out:
        
        video_names = list(f_gmm.keys())
        
        for vid in tqdm(video_names, desc="Processing videos"):
            gmm_path = f"{vid}/{STAGE}/k_{K_CLUSTERS}"
            # hBehaveMAE typically stores the embedding matrix directly under the stage group
            emb_path = f"{vid}/{STAGE}" 
            
            if gmm_path in f_gmm and emb_path in f_emb and vid in keypoints_dict:
                # 1. Get Labels & Embeddings
                labels = f_gmm[gmm_path][:]
                latent = f_emb[emb_path][:]
                kpts = keypoints_dict[vid]
                
                # Match lengths in case of sliding window drops during embedding extraction
                min_len = min(len(labels), len(kpts), len(latent))
                labels = labels[:min_len]
                latent = latent[:min_len]
                kpts = fill_holes_simple(kpts[:min_len])
                
                # 2. Calculate Centroid (mouse_center)
                centroid = kpts[:, c_idx, :].astype(np.float64)
                
                # 3. Calculate Heading (Nose - Tail)
                dy = kpts[:, n_idx, 1] - kpts[:, t_idx, 1]
                dx = kpts[:, n_idx, 0] - kpts[:, t_idx, 0]
                heading = np.arctan2(dy, dx).astype(np.float64)
                
                # Write to HDF5
                grp = f_out.create_group(vid)
                grp.create_dataset("syllable", data=labels.astype(np.int64))
                grp.create_dataset("centroid", data=centroid)
                grp.create_dataset("heading", data=heading)
                grp.create_dataset("latent_state", data=latent.astype(np.float64))

    print("\nDone! Your file is ready to drop right into Keypoint MoSeq.")

if __name__ == "__main__":
    main()