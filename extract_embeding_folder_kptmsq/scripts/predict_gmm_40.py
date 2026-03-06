#!/usr/bin/env python3
"""
Step 2: GMM Predicting
Loads the fitted K=40 GMMs and applies them to every frame of every video.
Saves the discrete labels to a clean HDF5 file.
"""

import os
from pathlib import Path
import h5py
import joblib
from tqdm import tqdm

# We don't need massive parallelization for prediction, but a few threads help
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# --- Configuration ---
PROJECT_ROOT = "/scratch/michal/projects/dvc_ofd_2025"
H5_DIR = f"{PROJECT_ROOT}/code/BehaveMAE_reconstruction/extract_embeding_folder_kptmsq/data/shuffle-3_split-train"
OUT_DIR = Path(f"{PROJECT_ROOT}/code/BehaveMAE_reconstruction/extract_embeding_folder_kptmsq/gmm_40_clustering_outputs")
MODELS_OUT_DIR = OUT_DIR / "fitted_models"

MODELS = {
    "base": f"{H5_DIR}/ofd_base_20260226-204726.h5",
    "tail": f"{H5_DIR}/ofd_tail_20260226-204729.h5",
    "tailhip": f"{H5_DIR}/ofd_tailhip_20260226-205043.h5"
}

STAGES = ["stage1", "stage2", "stage3"]
K_CLUSTERS = 40

def predict_model(model_name, h5_path):
    out_h5_path = OUT_DIR / f"gmm_labels_{model_name}_40.h5"
    print(f"\n{'='*60}\nPredicting labels for Model: {model_name}\nSaving to: {out_h5_path.name}\n{'='*60}")
    
    with h5py.File(h5_path, 'r') as f_in, h5py.File(out_h5_path, 'a') as f_out:
        video_names = list(f_in.keys())
        
        for stage in STAGES:
            print(f"\n--- Stage: {stage} ---")
            
            scaler_path = MODELS_OUT_DIR / f"scaler_{model_name}_{stage}.pkl"
            gmm_path = MODELS_OUT_DIR / f"gmm_{model_name}_{stage}.pkl"
            
            if not (scaler_path.exists() and gmm_path.exists()):
                print(f"  -> WARNING: Missing .pkl files for {model_name} {stage}. Skipping.")
                continue
                
            # Load the fitted models
            print("  -> Loading Scaler and GMM from disk...")
            scaler = joblib.load(scaler_path)
            gmm = joblib.load(gmm_path)
            
            # Predict video by video to keep memory footprint tiny
            for vid in tqdm(video_names, desc=f"Predicting"):
                # 1. Load raw embedding
                emb = f_in[vid][stage][:]
                
                # 2. Scale
                emb_scaled = scaler.transform(emb)
                
                # 3. Predict cluster labels (0 to 39)
                labels = gmm.predict(emb_scaled)
                
                # 4. Save to HDF5
                dataset_path = f"{vid}/{stage}/k_{K_CLUSTERS}"
                
                # Overwrite if it already exists (useful if you rerun the script)
                if dataset_path in f_out:
                    del f_out[dataset_path]
                    
                f_out.create_dataset(
                    dataset_path, 
                    data=labels, 
                    compression="gzip", 
                    compression_opts=4
                )

def main():
    for model_name, path in MODELS.items():
        if os.path.exists(path):
            predict_model(model_name, path)
        else:
            print(f"Could not find raw H5 file for {model_name}")
            
    print("\nALL PREDICTIONS COMPLETE! Your discrete syllables are ready.")

if __name__ == "__main__":
    main()