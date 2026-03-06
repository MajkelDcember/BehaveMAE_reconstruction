#!/usr/bin/env python3
import os
import json
from pathlib import Path

# Limit threads for cluster safety
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import umap

# --- Configuration ---
PROJECT_ROOT = "/scratch/michal/projects/dvc_ofd_2025"
H5_DIR = f"{PROJECT_ROOT}/code/BehaveMAE_reconstruction/extract_embeding_folder_kptmsq/data/shuffle-3_split-train"
KPMS_RESULTS = f"{PROJECT_ROOT}/data/interim/keypoint_moseq_project/shuffle-3_projset-0/shuffle-3_projset-0_modset-0/results.h5"
OUTPUT_DIR = Path(f"{PROJECT_ROOT}/code/BehaveMAE_reconstruction/analysis_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "base": f"{H5_DIR}/ofd_base_20260226-204726.h5",
    "tail": f"{H5_DIR}/ofd_tail_20260226-204729.h5",
    "tailhip": f"{H5_DIR}/ofd_tailhip_20260226-205043.h5"
}

N_VIDEOS = 100
N_FRAMES_PER_VID = 5000
K_RANGE = [10, 20, 30, 40, 50, 60, 70, 80, 100, 150]


def load_aligned_data(model_path, kpms_path):
    """Loads hBehaveMAE embeddings and KPMS syllables, perfectly aligned."""
    hbmae_data = {"stage1": [], "stage2": [], "stage3": []}
    kpms_labels = []
    
    with h5py.File(model_path, 'r') as f_hbmae, h5py.File(kpms_path, 'r') as f_kpms:
        # Find videos present in both
        common_vids = [v for v in f_hbmae.keys() if v in f_kpms and 'syllable' in f_kpms[v]]
        
        # Subsample videos
        np.random.seed(42)
        vids_to_sample = np.random.choice(common_vids, min(N_VIDEOS, len(common_vids)), replace=False)
        
        for vid in vids_to_sample:
            syls = f_kpms[vid]['syllable'][:]
            T_hbmae = f_hbmae[vid].attrs["num_frames"]
            
            # The maximum frames we can safely compare
            max_valid_frames = min(len(syls), T_hbmae)
            
            # Randomly sample frames from this video
            idx = np.sort(np.random.choice(max_valid_frames, min(N_FRAMES_PER_VID, max_valid_frames), replace=False))
            
            kpms_labels.append(syls[idx])
            for stage in hbmae_data.keys():
                hbmae_data[stage].append(f_hbmae[vid][stage][idx, :])
                
    # Concatenate globally
    y_kpms = np.concatenate(kpms_labels)
    X_stages = {}
    for stage, data in hbmae_data.items():
        X_stages[stage] = StandardScaler().fit_transform(np.vstack(data))
        
    return X_stages, y_kpms


def main():
    print(f"Starting Comprehensive Analysis...")
    
    all_results = {}
    
    for model_name, h5_path in MODELS.items():
        print(f"\n{'='*50}\nEvaluating Model: {model_name}\n{'='*50}")
        model_out_dir = OUTPUT_DIR / model_name
        model_out_dir.mkdir(exist_ok=True)
        
        # 1. Load Aligned Data
        print("Loading aligned data...")
        X_stages, y_kpms = load_aligned_data(h5_path, KPMS_RESULTS)
        print(f"Total frames extracted: {len(y_kpms)}")
        
        # 2. Dimensionality Reduction (PCA & UMAP)
        # We heavily subsample for UMAP to keep time reasonable
        umap_idx = np.random.choice(len(y_kpms), min(30000, len(y_kpms)), replace=False)
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 18))
        fig.suptitle(f"{model_name.upper()} - Dimensionality Reduction", fontsize=16)
        
        for i, (stage, X) in enumerate(X_stages.items()):
            print(f"Computing PCA & UMAP for {stage}...")
            # PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            axes[i, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_kpms, cmap='tab20', s=1, alpha=0.3)
            axes[i, 0].set_title(f"{stage} - PCA (Colored by KPMS Syllable)")
            
            # UMAP
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            X_umap = reducer.fit_transform(X[umap_idx])
            axes[i, 1].scatter(X_umap[:, 0], X_umap[:, 1], c=y_kpms[umap_idx], cmap='tab20', s=2, alpha=0.5)
            axes[i, 1].set_title(f"{stage} - UMAP (Colored by KPMS Syllable)")
            
        plt.tight_layout()
        plt.savefig(model_out_dir / f"{model_name}_dim_reduction.png", dpi=200)
        plt.close()
        
        # 3. K-Means Sweep & KPMS Alignment
        print("Sweeping K-Means to find optimal alignment with KPMS...")
        results = {"stage1": {"nmi": [], "ari": [], "inertia": []},
                   "stage2": {"nmi": [], "ari": [], "inertia": []},
                   "stage3": {"nmi": [], "ari": [], "inertia": []}}
                   
        for stage, X in X_stages.items():
            print(f"  Sweeping {stage}...")
            for k in K_RANGE:
                km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=8192, n_init=3)
                preds = km.fit_predict(X)
                
                results[stage]["inertia"].append(km.inertia_)
                results[stage]["nmi"].append(normalized_mutual_info_score(y_kpms, preds))
                results[stage]["ari"].append(adjusted_rand_score(y_kpms, preds))
                
        all_results[model_name] = results
        
        # Plot K-Sweep Results (NMI and Inertia)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for stage, metrics in results.items():
            axes[0].plot(K_RANGE, metrics["inertia"], marker='o', label=stage)
            axes[1].plot(K_RANGE, metrics["nmi"], marker='o', label=stage)
            
        axes[0].set_title(f"{model_name} - Elbow Method (Inertia)")
        axes[0].set_xlabel("Number of Clusters (k)")
        axes[0].legend()
        
        axes[1].set_title(f"{model_name} - KPMS Alignment (NMI)")
        axes[1].set_xlabel("Number of Clusters (k)")
        axes[1].set_ylabel("Normalized Mutual Information")
        axes[1].axvline(np.argmax(results["stage1"]["nmi"]), color='red', linestyle='--', alpha=0.5, label='Max NMI (Stage1)')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(model_out_dir / f"{model_name}_k_sweep.png", dpi=200)
        plt.close()

    # Save final JSON results
    with open(OUTPUT_DIR / "comprehensive_metrics.json", "w") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\nAnalysis complete! Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()