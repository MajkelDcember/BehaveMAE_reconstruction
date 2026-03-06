#!/usr/bin/env python3
"""
GMM Cluster Kinematic Analysis for hBehaveMAE

Extracts behavioral bouts from GMM cluster labels, computes kinematics from
raw keypoints, and generates jointplots to interpret each cluster's physical signature.

Kinematics computed:
    - Centroid velocity (mouse_center speed, px/s)
    - Head velocity (nose speed, px/s)
    - Tail velocity (tail_base speed, px/s)
    - Angular velocity (body orientation change, deg/s)
    - Head-body angle (nose-neck vs neck-tail_base, degrees)
    - Body elongation (nose to tail_base distance, px)

Usage:
    python analyze_gmm_kinematics.py

Author: Claude
Date: 2026-03-02
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class Config:
    """Analysis configuration."""
    pkl_path: str = "/scratch/michal/projects/dvc_ofd_2025/code/ofdpipe/tmp/data/interim/keypoint_moseq_data/shuffle-3/shuffle-3_split-train.pkl"
    gmm_h5_path: str = "/scratch/michal/projects/dvc_ofd_2025/code/BehaveMAE_reconstruction/extract_embeding_folder_kptmsq/gmm_40_clustering_outputs/gmm_labels_tailhip_40.h5"
    output_dir: str = "/scratch/michal/projects/dvc_ofd_2025/code/BehaveMAE_reconstruction/extract_embeding_folder_kptmsq/gmm_40_clustering_outputs/kinematic_analysis"
    
    stage: str = "stage2"
    k_clusters: int = 40
    min_bout_length: int = 5
    fps: float = 25.0
    
    tailhip_keypoints: Tuple[str, ...] = (
        "nose", "head_midpoint", "left_ear", "right_ear",
        "neck", "mouse_center", "mid_backend2", "tail_base",
        "left_midside", "right_midside", "left_hip", "right_hip",
        "tail2", "tail4", "tail_end"
    )


# ===========================================================================
# Data Loading
# ===========================================================================

def load_pickle_data(pkl_path: str) -> Tuple[Dict, Dict, List[str]]:
    """Load the Keypoint-MoSeq pickle file."""
    print(f"Loading pickle data from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        raw_data = pickle.load(f)
    keypoints_dict = raw_data[0]
    confidences_dict = raw_data[1]
    keypoint_names = list(raw_data[2])
    print(f"  Loaded {len(keypoints_dict)} videos")
    print(f"  Source keypoints ({len(keypoint_names)}): {keypoint_names[:5]}...")
    return keypoints_dict, confidences_dict, keypoint_names


def build_keypoint_index_map(
    source_names: List[str], 
    target_names: Tuple[str, ...]
) -> Dict[str, int]:
    """Build a mapping from keypoint name -> index in source array."""
    index_map = {}
    missing = []
    for name in target_names:
        if name in source_names:
            index_map[name] = source_names.index(name)
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"Missing keypoints in source data: {missing}")
    print(f"  Built index map for {len(index_map)} keypoints")
    return index_map


def fill_holes_simple(keypoints: np.ndarray) -> np.ndarray:
    """Fill NaN values using linear interpolation."""
    if not np.any(np.isnan(keypoints)):
        return keypoints
    result = keypoints.copy()
    T = result.shape[0]
    x_indices = np.arange(T)
    for kpt in range(result.shape[1]):
        for dim in range(result.shape[2]):
            data = result[:, kpt, dim]
            mask = np.isnan(data)
            if not np.any(mask):
                continue
            if np.all(mask):
                result[:, kpt, dim] = 0
                continue
            valid_mask = ~mask
            result[:, kpt, dim] = np.interp(x_indices, x_indices[valid_mask], data[valid_mask])
    return result


# ===========================================================================
# Bout Detection
# ===========================================================================

@dataclass
class Bout:
    """A contiguous sequence of frames with the same cluster label."""
    video_name: str
    cluster_id: int
    start_frame: int
    end_frame: int
    
    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame


def extract_bouts(
    labels: np.ndarray, 
    video_name: str, 
    min_length: int = 5
) -> List[Bout]:
    """Extract contiguous bouts from cluster labels."""
    bouts = []
    T = len(labels)
    if T == 0:
        return bouts
    start = 0
    current_cluster = labels[0]
    for i in range(1, T):
        if labels[i] != current_cluster:
            if i - start >= min_length:
                bouts.append(Bout(
                    video_name=video_name,
                    cluster_id=int(current_cluster),
                    start_frame=start,
                    end_frame=i
                ))
            start = i
            current_cluster = labels[i]
    if T - start >= min_length:
        bouts.append(Bout(
            video_name=video_name,
            cluster_id=int(current_cluster),
            start_frame=start,
            end_frame=T
        ))
    return bouts


# ===========================================================================
# Kinematic Calculations
# ===========================================================================

def compute_velocity(positions: np.ndarray, fps: float) -> np.ndarray:
    """Compute instantaneous speed from position time series (px/s)."""
    displacements = np.diff(positions, axis=0)
    return np.linalg.norm(displacements, axis=1) * fps


def compute_angular_velocity(
    neck_pos: np.ndarray, 
    tail_base_pos: np.ndarray, 
    fps: float
) -> np.ndarray:
    """Compute angular velocity of body orientation (deg/s)."""
    body_vec = tail_base_pos - neck_pos
    angles = np.arctan2(body_vec[:, 1], body_vec[:, 0])
    angle_diff = np.diff(angles)
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
    return np.abs(np.degrees(angle_diff)) * fps


def compute_head_body_angle(
    nose_pos: np.ndarray,
    neck_pos: np.ndarray,
    tail_base_pos: np.ndarray
) -> np.ndarray:
    """Compute angle between head direction and body axis (degrees)."""
    head_vec = nose_pos - neck_pos
    body_vec = neck_pos - tail_base_pos
    head_norm = np.maximum(np.linalg.norm(head_vec, axis=1, keepdims=True), 1e-6)
    body_norm = np.maximum(np.linalg.norm(body_vec, axis=1, keepdims=True), 1e-6)
    cos_angle = np.sum((head_vec / head_norm) * (body_vec / body_norm), axis=1)
    return np.abs(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def compute_body_elongation(
    nose_pos: np.ndarray, 
    tail_base_pos: np.ndarray
) -> np.ndarray:
    """Compute body length (nose to tail_base distance in pixels)."""
    return np.linalg.norm(nose_pos - tail_base_pos, axis=1)


def compute_bout_kinematics(
    keypoints: np.ndarray,
    kpt_idx: Dict[str, int],
    bout: Bout,
    fps: float
) -> Optional[Dict[str, float]]:
    """Compute all kinematic features for a single bout."""
    kpts = keypoints[bout.start_frame:bout.end_frame]
    if len(kpts) < 2:
        return None
    
    nose = kpts[:, kpt_idx["nose"], :]
    neck = kpts[:, kpt_idx["neck"], :]
    center = kpts[:, kpt_idx["mouse_center"], :]
    tail_base = kpts[:, kpt_idx["tail_base"], :]
    
    try:
        return {
            "cluster_id": bout.cluster_id,
            "bout_length": bout.length,
            "centroid_velocity": float(np.nanmean(compute_velocity(center, fps))),
            "head_velocity": float(np.nanmean(compute_velocity(nose, fps))),
            "tail_velocity": float(np.nanmean(compute_velocity(tail_base, fps))),
            "angular_velocity": float(np.nanmean(compute_angular_velocity(neck, tail_base, fps))),
            "head_body_angle": float(np.nanmean(compute_head_body_angle(nose, neck, tail_base))),
            "body_elongation": float(np.nanmean(compute_body_elongation(nose, tail_base))),
        }
    except Exception as e:
        print(f"  Warning: Failed to compute kinematics: {e}")
        return None


# ===========================================================================
# Main Analysis Pipeline
# ===========================================================================

def run_analysis(cfg: Config) -> pd.DataFrame:
    """Main analysis pipeline."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    keypoints_dict, _, keypoint_names = load_pickle_data(cfg.pkl_path)
    kpt_idx = build_keypoint_index_map(keypoint_names, cfg.tailhip_keypoints)
    
    all_bout_kinematics = []
    print(f"\nProcessing videos from {cfg.gmm_h5_path}...")
    print(f"  Stage: {cfg.stage}, K={cfg.k_clusters}")
    
    with h5py.File(cfg.gmm_h5_path, 'r') as f:
        video_names = list(f.keys())
        for vid_name in tqdm(video_names, desc="Analyzing videos"):
            if vid_name not in keypoints_dict:
                continue
            dataset_path = f"{vid_name}/{cfg.stage}/k_{cfg.k_clusters}"
            if dataset_path not in f:
                continue
            labels = f[dataset_path][:]
            raw_kpts = keypoints_dict[vid_name]
            if len(labels) != len(raw_kpts):
                continue
            kpts_filled = fill_holes_simple(raw_kpts)
            bouts = extract_bouts(labels, vid_name, min_length=cfg.min_bout_length)
            for bout in bouts:
                kin = compute_bout_kinematics(kpts_filled, kpt_idx, bout, cfg.fps)
                if kin is not None:
                    kin["video_name"] = vid_name
                    all_bout_kinematics.append(kin)
    
    df = pd.DataFrame(all_bout_kinematics)
    print(f"\n{'='*60}")
    print(f"Analysis Complete!")
    print(f"  Total bouts analyzed: {len(df)}")
    if len(df) > 0:
        print(f"  Bouts per cluster: ~{len(df) // cfg.k_clusters}")
    print(f"{'='*60}")
    return df


# ===========================================================================
# Visualization (CORRECTED)
# ===========================================================================

def plot_cluster_jointplots(
    df: pd.DataFrame, 
    output_dir: Path,
    x_var: str = "centroid_velocity",
    y_var: str = "head_velocity",
    n_cols: int = 8
) -> None:
    """Create a grid of hexbin jointplots, one per cluster."""
    if df.empty:
        print(f"  No data: skipping jointplot {x_var} vs {y_var}")
        return
    
    k_clusters = df["cluster_id"].nunique()
    n_rows = (k_clusters + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten()
    
    x_min, x_max = df[x_var].quantile(0.01), df[x_var].quantile(0.99)
    y_min, y_max = df[y_var].quantile(0.01), df[y_var].quantile(0.99)
    
    for cluster_id in range(k_clusters):
        ax = axes[cluster_id]
        cluster_df = df[df["cluster_id"] == cluster_id]
        if len(cluster_df) < 10:
            ax.text(0.5, 0.5, f"C{cluster_id}\n(n<10)", ha='center', va='center', transform=ax.transAxes)
        else:
            ax.hexbin(
                cluster_df[x_var], 
                cluster_df[y_var], 
                gridsize=20, 
                cmap='viridis', 
                mincnt=1, 
                extent=[x_min, x_max, y_min, y_max]
            )
        ax.set_title(f"C{cluster_id} (n={len(cluster_df):,})", fontsize=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if cluster_id % n_cols == 0:
            ax.set_ylabel(y_var.replace("_", " ").title(), fontsize=8)
        else:
            ax.set_yticklabels([])
        if cluster_id >= (n_rows - 1) * n_cols:
            ax.set_xlabel(x_var.replace("_", " ").title(), fontsize=8)
        else:
            ax.set_xticklabels([])
    
    for i in range(k_clusters, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f"GMM Cluster Kinematics: {y_var} vs {x_var}", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = output_dir / f"jointplot_grid_{x_var}_vs_{y_var}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_cluster_summary(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar summaries + CSV of per-cluster means (CORRECTED)."""
    if df.empty:
        print("  No data: skipping summary")
        return

    cluster_means = (
        df.groupby("cluster_id")
        .agg(
            centroid_velocity=("centroid_velocity", "mean"),
            head_velocity=("head_velocity", "mean"),
            tail_velocity=("tail_velocity", "mean"),
            angular_velocity=("angular_velocity", "mean"),
            head_body_angle=("head_body_angle", "mean"),
            body_elongation=("body_elongation", "mean"),
            mean_bout_length=("bout_length", "mean"),
            bout_count=("bout_length", "count"),
        )
        .reset_index()
    )

    cluster_means = cluster_means.sort_values("centroid_velocity", ascending=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics = [
        ("centroid_velocity", "Centroid Velocity (px/s)"),
        ("head_velocity", "Head Velocity (px/s)"),
        ("tail_velocity", "Tail Velocity (px/s)"),
        ("angular_velocity", "Angular Velocity (deg/s)"),
        ("head_body_angle", "Head-Body Angle (deg)"),
        ("body_elongation", "Body Elongation (px)"),
    ]

    for ax, (col, title) in zip(axes.flatten(), metrics):
        x_pos = np.arange(len(cluster_means))
        ax.bar(x_pos, cluster_means[col].values)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cluster_means["cluster_id"].values, fontsize=7, rotation=90)
        ax.set_xlabel("Cluster ID (sorted by speed)")
        ax.set_ylabel(title)
        ax.set_title(title)

    plt.suptitle("Mean Kinematics per GMM Cluster", fontsize=16)
    plt.tight_layout()

    png_path = output_dir / "cluster_summary_bars.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {png_path.name}")

    csv_path = output_dir / "cluster_kinematics_summary.csv"
    cluster_means.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path.name}")


def plot_kinematic_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Create a heatmap showing normalized kinematic profiles per cluster."""
    if df.empty:
        print("  No data: skipping heatmap")
        return
    
    metrics = [
        "centroid_velocity", "head_velocity", "tail_velocity",
        "angular_velocity", "head_body_angle", "body_elongation"
    ]
    
    cluster_means = df.groupby("cluster_id")[metrics].mean()
    cluster_means_norm = (cluster_means - cluster_means.mean()) / cluster_means.std()
    sort_order = cluster_means["centroid_velocity"].sort_values(ascending=False).index
    cluster_means_norm = cluster_means_norm.loc[sort_order]
    
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(
        cluster_means_norm,
        cmap="RdBu_r",
        center=0,
        xticklabels=[m.replace("_", "\n") for m in metrics],
        yticklabels=cluster_means_norm.index,
        ax=ax
    )
    ax.set_xlabel("Kinematic Feature", fontsize=12)
    ax.set_ylabel("Cluster ID (sorted by centroid velocity)", fontsize=12)
    ax.set_title("Z-scored Kinematic Profiles per GMM Cluster", fontsize=14)
    plt.tight_layout()
    
    save_path = output_dir / "cluster_kinematics_heatmap.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def generate_all_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate all visualization outputs (CORRECTED)."""
    print("\nGenerating visualizations...")
    
    # 1. Generate the bar charts and CSV
    plot_cluster_summary(df, output_dir)
    
    # 2. Generate the Z-scored heatmap
    plot_kinematic_heatmap(df, output_dir)

    # 3. Generate all the Hexbin Jointplot Grids
    jointplot_pairs = [
        ("centroid_velocity", "head_velocity"),
        ("centroid_velocity", "angular_velocity"),
        ("centroid_velocity", "body_elongation"),
        ("angular_velocity", "head_body_angle"),
        ("head_velocity", "tail_velocity"),
        ("body_elongation", "head_body_angle"),
    ]
    
    for x_var, y_var in jointplot_pairs:
        plot_cluster_jointplots(df, output_dir, x_var=x_var, y_var=y_var)

    print("\nAll visualizations complete!")


# ===========================================================================
# Main
# ===========================================================================

# ===========================================================================
# Main
# ===========================================================================

def main():
    cfg = Config()
    
    print("="*60)
    print("GMM Cluster Kinematic Analysis")
    print("="*60)
    print(f"Pickle path: {cfg.pkl_path}")
    print(f"GMM H5 path: {cfg.gmm_h5_path}")
    print(f"Output dir:  {cfg.output_dir}")
    print(f"Stage: {cfg.stage}, K={cfg.k_clusters}")
    print(f"Min bout length: {cfg.min_bout_length} frames")
    print(f"Frame rate: {cfg.fps} fps")
    print("="*60)
    
    # Run analysis
    df = run_analysis(cfg)
    
    # Save raw data with a Safety Net
    output_dir = Path(cfg.output_dir)
    parquet_path = output_dir / "bout_kinematics.parquet"
    csv_fallback_path = output_dir / "bout_kinematics_fallback.csv"
    
    if not df.empty:
        try:
            df.to_parquet(parquet_path, index=False)
            print(f"\nSaved bout kinematics to: {parquet_path}")
        except Exception as e:
            print(f"\nWarning: Failed to save Parquet ({e}). Falling back to CSV...")
            df.to_csv(csv_fallback_path, index=False)
            print(f"Saved bout kinematics to: {csv_fallback_path}")
    
    # Generate visualizations
    try:
        generate_all_visualizations(df, output_dir)
    except Exception as e:
        print(f"\nError generating visualizations: {e}")
    
    # Print quick stats
    print("\n" + "="*60)
    print("Quick Stats per Cluster:")
    print("="*60)
    if len(df) > 0:
        summary = df.groupby("cluster_id").agg({
            "centroid_velocity": ["mean", "std"],
            "angular_velocity": ["mean", "std"],
            "bout_length": "count"
        })
        print(summary.head(10))
        print("...")
    else:
        print("No bouts found!")

if __name__ == "__main__":
    main()