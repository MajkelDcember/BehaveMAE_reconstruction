#!/usr/bin/env python3
"""
Step 1: GMM Fitting (K=40)
Fits a StandardScaler and GaussianMixture model on a subsample of hBehaveMAE embeddings.
Saves the fitted models to disk for later prediction.
"""

import os
from pathlib import Path
import gc
import h5py
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- Cluster limits ---
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# --- Configuration ---
PROJECT_ROOT = "/scratch/michal/projects/dvc_ofd_2025"
H5_DIR = f"{PROJECT_ROOT}/code/BehaveMAE_reconstruction/extract_embeding_folder_kptmsq/data/shuffle-3_split-train"
OUT_DIR = Path(f"{PROJECT_ROOT}/code/BehaveMAE_reconstruction/extract_embeding_folder_kptmsq/gmm_40_clustering_outputs")

# Create directories to hold our saved models
MODELS_OUT_DIR = OUT_DIR / "fitted_models"
MODELS_OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "base": f"{H5_DIR}/ofd_base_20260226-204726.h5",
    "tail": f"{H5_DIR}/ofd_tail_20260226-204729.h5",
    "tailhip": f"{H5_DIR}/ofd_tailhip_20260226-205043.h5"
}

STAGES = ["stage1", "stage2", "stage3"]
K_CLUSTERS = 40
FIT_SUBSAMPLE_SIZE = 2_000_000  # Number of frames to fit the GMM


def fit_model_stage(model_name, h5_path):
    print(f"\n{'='*60}\nFitting GMMs for Model: {model_name}\n{'='*60}")
    
    with h5py.File(h5_path, 'r') as f_in:
        video_names = list(f_in.keys())
        
        for stage in STAGES:
            print(f"\n--- Processing {stage} ---")
            
            # 1. Load embeddings into memory
            # Note: We load all to RAM to allow fast random subsampling, 
            # then delete the massive array immediately to free memory.
            all_embs = []
            for vid in tqdm(video_names, desc=f"Loading {stage} frames"):
                all_embs.append(f_in[vid][stage][:])
                
            all_embs = np.vstack(all_embs)
            total_frames = all_embs.shape[0]
            
            # 2. Subsample
            print(f"Subsampling {FIT_SUBSAMPLE_SIZE} out of {total_frames} frames...")
            np.random.seed(42)
            # If total frames are less than subsample size, just use all of them
            actual_subsample = min(FIT_SUBSAMPLE_SIZE, total_frames)
            fit_idx = np.random.choice(total_frames, actual_subsample, replace=False)
            fit_data = all_embs[fit_idx]
            
            # Free up RAM aggressively before doing math
            del all_embs
            gc.collect()

            # 3. Fit StandardScaler
            print("Fitting StandardScaler...")
            scaler = StandardScaler()
            fit_data_scaled = scaler.fit_transform(fit_data)
            
            # Save the scaler so we can use exactly the same scaling during prediction
            scaler_path = MODELS_OUT_DIR / f"scaler_{model_name}_{stage}.pkl"
            joblib.dump(scaler, scaler_path)

            # 4. Fit GMM
            print(f"Fitting GMM (K={K_CLUSTERS}). This may take a moment...")
            gmm = GaussianMixture(
                n_components=K_CLUSTERS, 
                covariance_type='diag', 
                max_iter=200,      # Slightly higher max_iter to ensure convergence
                n_init=1,          # 1 initialization is fine for K=40 with 2M frames
                random_state=42,
                verbose=2,         # Let's us see the convergence progress in the logs
                verbose_interval=10
            )
            gmm.fit(fit_data_scaled)
            
            # Save the fitted GMM
            gmm_path = MODELS_OUT_DIR / f"gmm_{model_name}_{stage}.pkl"
            joblib.dump(gmm, gmm_path)
            
            print(f"-> Saved Scaler and GMM for {model_name} {stage}!")
            
            # Clean up before next stage
            del fit_data, fit_data_scaled, scaler, gmm
            gc.collect()

def main():
    for model_name, path in MODELS.items():
        if os.path.exists(path):
            fit_model_stage(model_name, path)
        else:
            print(f"Could not find H5 file for {model_name} at {path}")
            
    print("\nALL FITTING COMPLETE! Models saved to:", MODELS_OUT_DIR)

if __name__ == "__main__":
    main()