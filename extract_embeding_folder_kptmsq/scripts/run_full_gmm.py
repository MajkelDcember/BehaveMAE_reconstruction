#!/usr/bin/env python3
"""
Full Dataset GMM Clustering for hBehaveMAE Embeddings.
Fits a GMM on a large subsample, predicts on the entire dataset, 
evaluates against KPMS, and saves all labels to an HDF5 file.
"""

import os
import json
from pathlib import Path
import gc

# Maximize core usage for BLAS operations during GMM fitting
os.environ["OMP_NUM_THREADS"] = "24"
os.environ["OPENBLAS_NUM_THREADS"] = "24"
os.environ["MKL_NUM_THREADS"] = "24"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm

# --- Configuration ---
PROJECT_ROOT = "/scratch/michal/projects/dvc_ofd_2025"
H5_DIR = f"{PROJECT_ROOT}/code/BehaveMAE_reconstruction/extract_embeding_folder_kptmsq/data/shuffle-3_split-train"
KPMS_RESULTS = f"{PROJECT_ROOT}/data/interim/keypoint_moseq_project/shuffle-3_projset-0/shuffle-3_projset-0_modset-0/results.h5"

OUTPUT_DIR = Path(f"{PROJECT_ROOT}/code/BehaveMAE_reconstruction/extract_embeding_folder_kptmsq/gmm_clustering_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "base": f"{H5_DIR}/ofd_base_20260226-204726.h5",
    "tail": f"{H5_DIR}/ofd_tail_20260226-204729.h5",
    "tailhip": f"{H5_DIR}/ofd_tailhip_20260226-205043.h5"
}

K_RANGE = [10, 20, 30, 40, 50, 60, 80, 100]
STAGES = ["stage1", "stage2", "stage3"]
FIT_SUBSAMPLE_SIZE = 2_000_000  # Number of frames to fit the GMM on


def load_kpms_dict():
    """Load KPMS syllables into a dictionary in memory for fast lookup."""
    print("Loading KPMS labels into memory...")
    kpms_dict = {}
    with h5py.File(KPMS_RESULTS, 'r') as f:
        for vid in f.keys():
            if 'syllable' in f[vid]:
                kpms_dict[vid] = f[vid]['syllable'][:]
    return kpms_dict


def process_model(model_name: str, h5_path: str, kpms_dict: dict, metrics_dict: dict):
    print(f"\n{'='*60}\nProcessing Model: {model_name}\n{'='*60}")
    
    out_h5_path = OUTPUT_DIR / f"gmm_labels_{model_name}.h5"
    metrics_dict[model_name] = {}

    with h5py.File(h5_path, 'r') as f_in:
        video_names = list(f_in.keys())
        
        for stage in STAGES:
            print(f"\n--- Loading {stage} for all videos ---")
            metrics_dict[model_name][stage] = {"nmi": {}, "ari": {}}
            
            # 1. Load all embeddings for this stage to standardize globally
            all_embs = []
            video_boundaries = [0]
            
            for vid in tqdm(video_names, desc="Reading"):
                emb = f_in[vid][stage][:]
                all_embs.append(emb)
                video_boundaries.append(video_boundaries[-1] + emb.shape[0])
                
            all_embs = np.vstack(all_embs)
            total_frames = all_embs.shape[0]
            print(f"Total frames loaded: {total_frames} (Shape: {all_embs.shape})")

            # 2. Standardize
            print("Standardizing data...")
            scaler = StandardScaler()
            all_embs = scaler.fit_transform(all_embs)
            
            # 3. Subsample for fitting
            print(f"Subsampling {FIT_SUBSAMPLE_SIZE} frames for GMM fitting...")
            np.random.seed(42)
            fit_idx = np.random.choice(total_frames, min(FIT_SUBSAMPLE_SIZE, total_frames), replace=False)
            fit_data = all_embs[fit_idx]

            # 4. Sweep through K
            for k in K_RANGE:
                print(f"\nFitting GMM for K={k}...")
                # Diagonal covariance is critical for high-dim stability and speed
                gmm = GaussianMixture(
                    n_components=k, 
                    covariance_type='diag', 
                    max_iter=150, 
                    n_init=1, 
                    random_state=42,
                    verbose=1,
                    verbose_interval=10
                )
                gmm.fit(fit_data)
                
                print("Predicting across entire dataset...")
                global_labels = gmm.predict(all_embs)
                
                # 5. Calculate Metrics and Save to H5
                nmi_true, nmi_pred = [], []
                
                with h5py.File(out_h5_path, 'a') as f_out:
                    for i, vid in enumerate(video_names):
                        start = video_boundaries[i]
                        end = video_boundaries[i+1]
                        vid_labels = global_labels[start:end]
                        
                        # Save to H5
                        dataset_path = f"{vid}/{stage}/k_{k}"
                        if dataset_path in f_out:
                            del f_out[dataset_path]
                        f_out.create_dataset(dataset_path, data=vid_labels, compression="gzip", compression_opts=4)
                        
                        # Accumulate for metrics if KPMS exists
                        if vid in kpms_dict:
                            y_true = kpms_dict[vid]
                            min_len = min(len(y_true), len(vid_labels))
                            nmi_true.append(y_true[:min_len])
                            nmi_pred.append(vid_labels[:min_len])
                
                # Compute metrics
                if nmi_true:
                    nmi_true = np.concatenate(nmi_true)
                    nmi_pred = np.concatenate(nmi_pred)
                    nmi = normalized_mutual_info_score(nmi_true, nmi_pred)
                    ari = adjusted_rand_score(nmi_true, nmi_pred)
                    
                    metrics_dict[model_name][stage]["nmi"][k] = nmi
                    metrics_dict[model_name][stage]["ari"][k] = ari
                    print(f"  -> NMI: {nmi:.4f} | ARI: {ari:.4f}")
                    
            # Aggressive cleanup before loading next stage to prevent RAM buildup
            del all_embs, fit_data, global_labels
            gc.collect()

def main():
    kpms_dict = load_kpms_dict()
    all_metrics = {}
    
    for model_name, path in MODELS.items():
        process_model(model_name, path, kpms_dict, all_metrics)
        
        # Save progress after each model in case of crash
        with open(OUTPUT_DIR / "gmm_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=4)
            
    print("\nALL DONE! Have a great weekend!")

if __name__ == "__main__":
    main()