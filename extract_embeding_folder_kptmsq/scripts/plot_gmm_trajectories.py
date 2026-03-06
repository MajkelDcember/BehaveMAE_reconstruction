#!/usr/bin/env python3
"""
GMM Cluster Trajectory Plots

Extracts aligned, onset-triggered average trajectories for each GMM cluster.
Centers the mouse at (0,0) and rotates it to face East at bout onset (t=0).
Generates a 40-cluster grid showing the average skeletal movement over time.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tqdm import tqdm
import matplotlib.animation as animation


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class Config:
    pkl_path: str = "/scratch/michal/projects/dvc_ofd_2025/code/ofdpipe/tmp/data/interim/keypoint_moseq_data/shuffle-3/shuffle-3_split-train.pkl"
    gmm_h5_path: str = "/scratch/michal/projects/dvc_ofd_2025/code/BehaveMAE_reconstruction/extract_embeding_folder_kptmsq/gmm_40_clustering_outputs/gmm_labels_tailhip_40.h5"
    output_dir: str = "/scratch/michal/projects/dvc_ofd_2025/code/BehaveMAE_reconstruction/extract_embeding_folder_kptmsq/gmm_40_clustering_outputs/kinematic_analysis"
    
    stage: str = "stage2"
    k_clusters: int = 40
    min_bout_length: int = 5
    
    # Trajectory Window Parameters (in frames)
    # E.g., pre=5, post=15 means we look at the 5 frames before onset and 15 frames after.
    pre_frames: int = 5  
    post_frames: int = 15 
    
    tailhip_keypoints: Tuple[str, ...] = (
        "nose", "head_midpoint", "left_ear", "right_ear",
        "neck", "mouse_center", "mid_backend2", "tail_base",
        "left_midside", "right_midside", "left_hip", "right_hip",
        "tail2", "tail4", "tail_end"
    )

    # Define the skeleton for plotting lines between keypoints
    skeleton_edges: Tuple[Tuple[str, str], ...] = (
        ("nose", "head_midpoint"), ("head_midpoint", "neck"),
        ("left_ear", "head_midpoint"), ("right_ear", "head_midpoint"),
        ("neck", "mouse_center"), ("mouse_center", "mid_backend2"), 
        ("mid_backend2", "tail_base"),
        ("left_midside", "mouse_center"), ("right_midside", "mouse_center"),
        ("left_hip", "mid_backend2"), ("right_hip", "mid_backend2"),
        ("tail_base", "tail2"), ("tail2", "tail4"), ("tail4", "tail_end")
    )


# ===========================================================================
# Core Logic
# ===========================================================================

def load_data(cfg: Config):
    """Loads Keypoints and creates the Index Map."""
    print("Loading raw pickle data...")
    with open(cfg.pkl_path, "rb") as f:
        raw_data = pickle.load(f)
    keypoints_dict = raw_data[0]
    source_names = list(raw_data[2])
    
    kpt_idx = {name: source_names.index(name) for name in cfg.tailhip_keypoints}
    edge_idx = [(kpt_idx[src], kpt_idx[dst]) for src, dst in cfg.skeleton_edges]
    
    return keypoints_dict, kpt_idx, edge_idx

def fill_holes(keypoints: np.ndarray) -> np.ndarray:
    """Linear interpolation for missing coordinates."""
    res = keypoints.copy()
    T = res.shape[0]
    x = np.arange(T)
    for k in range(res.shape[1]):
        for d in range(res.shape[2]):
            y = res[:, k, d]
            mask = np.isnan(y)
            if not np.any(mask): continue
            if np.all(mask):
                res[:, k, d] = 0
                continue
            res[:, k, d] = np.interp(x, x[~mask], y[~mask])
    return res

def align_trajectories(windows: np.ndarray, center_idx: int, neck_idx: int, tail_idx: int, onset_frame: int):
    """
    Translates and rotates all windows so that at `onset_frame`:
    - The `mouse_center` is at (0,0)
    - The vector from `tail_base` to `neck` points perfectly East (0 degrees).
    """
    if len(windows) == 0:
        return np.array([])
        
    # 1. Translate (Center at t=onset)
    origin = windows[:, onset_frame, center_idx, :] # (N, 2)
    centered = windows - origin[:, None, None, :]   # (N, T, K, 2)
    
    # 2. Rotate (Align Body Axis at t=onset)
    neck = centered[:, onset_frame, neck_idx, :]
    tail = centered[:, onset_frame, tail_idx, :]
    body_vec = neck - tail
    angles = np.arctan2(body_vec[:, 1], body_vec[:, 0]) # (N,)
    
    # We want to rotate by -angle to align with X-axis
    c = np.cos(-angles)[:, None, None]
    s = np.sin(-angles)[:, None, None]
    
    x = centered[..., 0]
    y = centered[..., 1]
    
    rot_x = x * c - y * s
    rot_y = x * s + y * c
    
    aligned = np.stack([rot_x, rot_y], axis=-1)
    return aligned

def extract_and_align_bouts(cfg: Config, keypoints_dict: dict, kpt_idx: dict) -> dict:
    """Finds bouts, extracts temporal windows, and computes aligned averages per cluster."""
    cluster_windows = {i: [] for i in range(cfg.k_clusters)}
    window_size = cfg.pre_frames + cfg.post_frames
    
    print("Extracting and aligning bout trajectories...")
    with h5py.File(cfg.gmm_h5_path, 'r') as f:
        for vid_name in tqdm(list(f.keys())):
            if vid_name not in keypoints_dict: continue
            
            dataset_path = f"{vid_name}/{cfg.stage}/k_{cfg.k_clusters}"
            if dataset_path not in f: continue
            
            labels = f[dataset_path][:]
            kpts = fill_holes(keypoints_dict[vid_name])
            
            # --- Scale to normalized units (Target: 25.0) ---
            neck = kpts[:, kpt_idx["neck"], :]
            tail = kpts[:, kpt_idx["tail_base"], :]
            median_dist = float(np.median(np.linalg.norm(neck - tail, axis=1)))
            if median_dist > 0:
                kpts = kpts * (25.0 / median_dist)
            # -------------------------------------------------

            # Find bout onsets
            T = len(labels)
            current = labels[0]
            start = 0
            
            for i in range(1, T):
                if labels[i] != current:
                    if (i - start) >= cfg.min_bout_length:
                        # We found a valid bout. Extract the window around onset `start`.
                        w_start = start - cfg.pre_frames
                        w_end = start + cfg.post_frames
                        
                        # Only grab windows that fit fully inside the video
                        if w_start >= 0 and w_end <= T:
                            cluster_windows[int(current)].append(kpts[w_start:w_end])
                            
                    start = i
                    current = labels[i]

    # Align and average
    print("Averaging trajectories...")
    avg_trajectories = {}
    for c_id, windows_list in cluster_windows.items():
        if len(windows_list) > 10:
            windows = np.stack(windows_list) # (N, T, K, 2)
            aligned = align_trajectories(
                windows, 
                center_idx=kpt_idx["mouse_center"], 
                neck_idx=kpt_idx["neck"], 
                tail_idx=kpt_idx["tail_base"], 
                onset_frame=cfg.pre_frames
            )
            # Take the mean across all bouts for this cluster
            avg_trajectories[c_id] = np.nanmean(aligned, axis=0) 
            
    return avg_trajectories

# ===========================================================================
# Plotting
# ===========================================================================

def plot_trajectory_grid(cfg: Config, avg_trajectories: dict, edge_idx: list):
    """Draws a grid of skeleton trajectories colored by time."""
    print("Generating trajectory plot grid...")
    
    n_cols = 8
    n_rows = (cfg.k_clusters + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten()
    
    T = cfg.pre_frames + cfg.post_frames
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.2, 1, T)) # Color gradient representing time
    
    # Use a fixed axis limit so all plots share the same physical scale
    lim = 45 

    for c_id in range(cfg.k_clusters):
        ax = axes[c_id]
        ax.set_aspect('equal')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Cluster {c_id}", fontsize=10)
        
        if c_id not in avg_trajectories:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue
            
        traj = avg_trajectories[c_id] # (T, K, 2)
        
        # Plot the skeleton at each time step
        for t in range(T):
            alpha = 0.3 if t < cfg.pre_frames else 0.8 # Pre-onset is faint, post-onset is bold
            color = colors[t]
            
            # Plot edges
            lines = [[traj[t, src], traj[t, dst]] for src, dst in edge_idx]
            lc = LineCollection(lines, colors=[color]*len(lines), alpha=alpha, linewidths=1.5)
            ax.add_collection(lc)
            
            # Plot Nose as a distinct dot to show facing direction
            nose_pos = traj[t, 0] # Index 0 is nose
            ax.scatter(nose_pos[0], nose_pos[1], color=color, s=10, alpha=alpha, zorder=5)

    # Clean up empty axes
    for i in range(cfg.k_clusters, len(axes)):
        axes[i].set_visible(False)
        
    plt.suptitle(f"Average Kinematic Trajectories per Cluster (Aligned at Onset)\nColor = Time (Pre-onset to Post-onset)", fontsize=16, y=1.02)
    plt.tight_layout()
    
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_trajectories_grid.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


    
def save_trajectory_gifs(cfg: Config, avg_trajectories: dict, edge_idx: list):
    """Generates an animated GIF for the average trajectory of each cluster."""
    print("Generating animated GIFs for each cluster...")
    
    # Create a subfolder specifically for the gifs
    gif_dir = Path(cfg.output_dir) / "trajectory_gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    
    T = cfg.pre_frames + cfg.post_frames
    lim = 45 # Same physical scale as the grid plot

    for c_id, traj in tqdm(avg_trajectories.items(), desc="Saving GIFs"):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.set_aspect('equal')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Cluster {c_id}", fontsize=12)

        # 1. Draw a faint background trail of the whole movement
        for t in range(T):
            lines = [[traj[t, src], traj[t, dst]] for src, dst in edge_idx]
            lc = LineCollection(lines, colors='lightgray', alpha=0.15, linewidths=1)
            ax.add_collection(lc)

        # 2. Initialize the active skeleton lines and nose point
        active_lines = [ax.plot([], [], '-', lw=2, color='teal')[0] for _ in edge_idx]
        nose_pt, = ax.plot([], [], 'ro', ms=5, zorder=10) # Red nose to show facing direction

        def update(t):
            # Update the skeleton lines for frame t
            for line, (src, dst) in zip(active_lines, edge_idx):
                line.set_data(
                    [traj[t, src, 0], traj[t, dst, 0]], 
                    [traj[t, src, 1], traj[t, dst, 1]]
                )
            # Update the nose dot
            nose_pt.set_data([traj[t, 0, 0]], [traj[t, 0, 1]])
            return active_lines + [nose_pt]

        # 3. Animate and save
        anim = animation.FuncAnimation(fig, update, frames=T, interval=1000/25.0, blit=True)
        
        gif_path = gif_dir / f"cluster_{c_id:02d}.gif"
        # We use 'pillow' because it doesn't require ffmpeg to be installed on your cluster
        anim.save(gif_path, writer='pillow', fps=25) 
        plt.close(fig)
        
    print(f"All GIFs saved to: {gif_dir}")

def main():
    cfg = Config()
    keypoints_dict, kpt_idx, edge_idx = load_data(cfg)
    avg_trajectories = extract_and_align_bouts(cfg, keypoints_dict, kpt_idx)
    plot_trajectory_grid(cfg, avg_trajectories, edge_idx)
    save_trajectory_gifs(cfg, avg_trajectories, edge_idx)
if __name__ == "__main__":
    main()