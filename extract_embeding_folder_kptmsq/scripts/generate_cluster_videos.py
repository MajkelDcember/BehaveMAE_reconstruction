# #!/usr/bin/env python3
# """
# Generate videos for each cluster showing consecutive frames belonging to that cluster.
# Creates 15 videos per stage (3 stages × 15 clusters = 45 videos total).
# """

# import argparse
# import json
# from pathlib import Path
# from typing import List, Tuple

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FFMpegWriter
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler


# # -----------------------------------------------------------------------------
# # Keypoint visualization
# # -----------------------------------------------------------------------------

# # Skeleton connections for mouse
# SKELETON = [
#     ("nose", "head_midpoint"),
#     ("head_midpoint", "left_ear"),
#     ("head_midpoint", "right_ear"),
#     ("left_ear", "left_ear_tip"),
#     ("right_ear", "right_ear_tip"),
#     ("head_midpoint", "left_eye"),
#     ("head_midpoint", "right_eye"),
#     ("head_midpoint", "neck"),
#     ("neck", "mid_back"),
#     ("mid_back", "mouse_center"),
#     ("mouse_center", "mid_backend"),
#     ("mid_backend", "mid_backend2"),
#     ("mid_backend2", "mid_backend3"),
#     ("mid_backend3", "tail_base"),
#     ("tail_base", "tail1"),
#     ("tail1", "tail2"),
#     ("tail2", "tail3"),
#     ("tail3", "tail4"),
#     ("tail4", "tail5"),
#     ("tail5", "tail_end"),
#     ("neck", "left_shoulder"),
#     ("left_shoulder", "left_midside"),
#     ("left_midside", "left_hip"),
#     ("left_hip", "tail_base"),
#     ("neck", "right_shoulder"),
#     ("right_shoulder", "right_midside"),
#     ("right_midside", "right_hip"),
#     ("right_hip", "tail_base"),
# ]

# ALL_KEYPOINTS = [
#     "nose", "left_ear", "right_ear", "left_ear_tip", "right_ear_tip",
#     "left_eye", "right_eye", "neck", "mid_back", "mouse_center",
#     "mid_backend", "mid_backend2", "mid_backend3", "tail_base",
#     "tail1", "tail2", "tail3", "tail4", "tail5",
#     "left_shoulder", "left_midside", "left_hip",
#     "right_shoulder", "right_midside", "right_hip",
#     "tail_end", "head_midpoint",
# ]

# KPT_TO_IDX = {name: i for i, name in enumerate(ALL_KEYPOINTS)}


# def get_skeleton_indices():
#     """Convert skeleton connections to index pairs."""
#     indices = []
#     for start, end in SKELETON:
#         if start in KPT_TO_IDX and end in KPT_TO_IDX:
#             indices.append((KPT_TO_IDX[start], KPT_TO_IDX[end]))
#     return indices


# def draw_pose(ax, keypoints, skeleton_idx, color='blue', alpha=1.0):
#     """Draw a single pose on an axis."""
#     # keypoints: (num_keypoints, 2)
#     ax.scatter(keypoints[:, 0], keypoints[:, 1], s=10, c=color, alpha=alpha, zorder=2)
    
#     for i, j in skeleton_idx:
#         if not np.any(np.isnan(keypoints[[i, j]])):
#             ax.plot(
#                 [keypoints[i, 0], keypoints[j, 0]],
#                 [keypoints[i, 1], keypoints[j, 1]],
#                 c=color, alpha=alpha * 0.7, linewidth=1, zorder=1
#             )


# # -----------------------------------------------------------------------------
# # Data loading
# # -----------------------------------------------------------------------------

# def load_raw_keypoints(data_path: Path) -> Tuple[dict, List[str]]:
#     """Load raw keypoints from .npy file."""
#     raw = np.load(data_path, allow_pickle=True).item()
#     sequences = raw["sequences"]
#     names = list(sequences.keys())
#     return sequences, names


# def load_all_embeddings(emb_root: Path) -> Tuple[dict, List[str], List[int]]:
#     """Load all embeddings and track video boundaries."""
#     video_dirs = sorted([d for d in emb_root.iterdir() if d.is_dir()])
    
#     all_stage1, all_stage2, all_stage3 = [], [], []
#     video_names = []
#     video_boundaries = [0]
    
#     for vdir in video_dirs:
#         emb_file = vdir / "embeddings.npz"
#         if not emb_file.exists():
#             continue
        
#         data = np.load(emb_file)
#         all_stage1.append(data["stage1"])
#         all_stage2.append(data["stage2"])
#         all_stage3.append(data["stage3"])
#         video_names.append(vdir.name)
#         video_boundaries.append(video_boundaries[-1] + len(data["stage1"]))
    
#     stages = {
#         "stage1": np.concatenate(all_stage1, axis=0),
#         "stage2": np.concatenate(all_stage2, axis=0),
#         "stage3": np.concatenate(all_stage3, axis=0),
#     }
    
#     return stages, video_names, video_boundaries


# def global_to_local_frame(global_idx: int, video_boundaries: List[int]) -> Tuple[int, int]:
#     """Convert global frame index to (video_idx, local_frame_idx)."""
#     for vid_idx in range(len(video_boundaries) - 1):
#         if video_boundaries[vid_idx] <= global_idx < video_boundaries[vid_idx + 1]:
#             local_idx = global_idx - video_boundaries[vid_idx]
#             return vid_idx, local_idx
#     raise ValueError(f"Global index {global_idx} out of bounds")


# # -----------------------------------------------------------------------------
# # Clustering
# # -----------------------------------------------------------------------------

# def cluster_embeddings(stages: dict, n_clusters: int, subsample: int = 100) -> dict:
#     """Cluster each stage separately."""
#     labels = {}
    
#     for stage_name, emb in stages.items():
#         print(f"Clustering {stage_name}...")
#         scaler = StandardScaler()
#         emb_scaled = scaler.fit_transform(emb)
        
#         # Subsample for fitting
#         idx = np.arange(0, len(emb), subsample)
#         emb_sub = emb_scaled[idx]
        
#         kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#         kmeans.fit(emb_sub)
        
#         # Predict on all
#         labels[stage_name] = kmeans.predict(emb_scaled)
    
#     return labels


# # -----------------------------------------------------------------------------
# # Video generation
# # -----------------------------------------------------------------------------

# def find_consecutive_segments(
#     cluster_labels: np.ndarray,
#     cluster_id: int,
#     video_boundaries: List[int],
#     max_frames: int = 1800,
# ) -> List[Tuple[int, int, int]]:
#     """
#     Find consecutive segments belonging to a cluster.
#     Returns list of (video_idx, start_local, end_local) tuples.
#     Stops when total frames reach max_frames.
#     """
#     segments = []
#     total_frames = 0
    
#     in_segment = False
#     seg_start = None
    
#     for i, label in enumerate(cluster_labels):
#         # Check if we crossed a video boundary
#         for vid_idx, boundary in enumerate(video_boundaries[1:], 1):
#             if i == boundary and in_segment:
#                 # End segment at video boundary
#                 vid_idx_seg, local_start = global_to_local_frame(seg_start, video_boundaries)
#                 _, local_end = global_to_local_frame(i - 1, video_boundaries)
#                 segments.append((vid_idx_seg, local_start, local_end + 1))
#                 total_frames += (local_end + 1 - local_start)
#                 in_segment = False
#                 break
        
#         if total_frames >= max_frames:
#             break
        
#         if label == cluster_id:
#             if not in_segment:
#                 seg_start = i
#                 in_segment = True
#         else:
#             if in_segment:
#                 vid_idx, local_start = global_to_local_frame(seg_start, video_boundaries)
#                 _, local_end = global_to_local_frame(i - 1, video_boundaries)
#                 segments.append((vid_idx, local_start, local_end + 1))
#                 total_frames += (local_end + 1 - local_start)
#                 in_segment = False
    
#     # Handle final segment
#     if in_segment and total_frames < max_frames:
#         vid_idx, local_start = global_to_local_frame(seg_start, video_boundaries)
#         _, local_end = global_to_local_frame(len(cluster_labels) - 1, video_boundaries)
#         segments.append((vid_idx, local_start, local_end + 1))
    
#     return segments


# def create_cluster_video(
#     sequences: dict,
#     video_names: List[str],
#     segments: List[Tuple[int, int, int]],
#     output_path: Path,
#     fps: int = 30,
#     max_frames: int = 1800,
# ):
#     """Create video from consecutive segments."""
#     skeleton_idx = get_skeleton_indices()
    
#     # Collect frames
#     frames_data = []
#     for vid_idx, start, end in segments:
#         if len(frames_data) >= max_frames:
#             break
        
#         vid_name = video_names[vid_idx]
#         if vid_name not in sequences:
#             continue
        
#         kpts = sequences[vid_name]["keypoints"]  # (T, 1, K, 2)
        
#         for t in range(start, min(end, kpts.shape[0])):
#             if len(frames_data) >= max_frames:
#                 break
#             frames_data.append((vid_name, t, kpts[t, 0]))  # (vid_name, frame_idx, keypoints)
    
#     if len(frames_data) == 0:
#         print(f"  No frames found, skipping")
#         return
    
#     print(f"  Creating video with {len(frames_data)} frames...")
    
#     # Create video
#     fig, ax = plt.subplots(figsize=(6, 6))
#     writer = FFMpegWriter(fps=fps, metadata=dict(artist='hBehaveMAE'))
    
#     output_path.parent.mkdir(parents=True, exist_ok=True)
    
#     with writer.saving(fig, str(output_path), dpi=100):
#         for i, (vid_name, frame_idx, kpts) in enumerate(frames_data):
#             ax.clear()
            
#             draw_pose(ax, kpts, skeleton_idx, color='blue')
            
#             ax.set_xlim(0, 500)
#             ax.set_ylim(500, 0)  # Flip y for image coordinates
#             ax.set_aspect('equal')
#             ax.set_title(f"Frame {i+1}/{len(frames_data)} | {vid_name}:{frame_idx}")
#             ax.axis('off')
            
#             writer.grab_frame()
            
#             if (i + 1) % 100 == 0:
#                 print(f"    {i+1}/{len(frames_data)} frames written")
    
#     plt.close(fig)
#     print(f"  Saved: {output_path}")


# # -----------------------------------------------------------------------------
# # Main
# # -----------------------------------------------------------------------------

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data", type=Path, required=True, help="Raw keypoints .npy file")
#     parser.add_argument("--emb-root", type=Path, default=Path("output_embedings"))
#     parser.add_argument("--output", type=Path, default=Path("cluster_videos"))
#     parser.add_argument("--n-clusters", type=int, default=15)
#     parser.add_argument("--max-frames", type=int, default=1800)
#     parser.add_argument("--fps", type=int, default=30)
#     parser.add_argument("--stages", nargs="+", default=["stage1", "stage2", "stage3"])
#     parser.add_argument("--clusters", type=int, nargs="+", default=None, help="Specific clusters to generate")
#     args = parser.parse_args()
    
#     print("Loading raw keypoints...")
#     sequences, raw_names = load_raw_keypoints(args.data)
#     print(f"  Loaded {len(raw_names)} sequences")
    
#     print("Loading embeddings...")
#     stages, emb_names, video_boundaries = load_all_embeddings(args.emb_root)
#     print(f"  Loaded {len(emb_names)} videos, {stages['stage1'].shape[0]} total frames")
    
#     print("Clustering...")
#     labels = cluster_embeddings(stages, args.n_clusters)
    
#     # Generate videos
#     clusters_to_process = args.clusters if args.clusters else list(range(args.n_clusters))
    
#     for stage_name in args.stages:
#         stage_labels = labels[stage_name]
#         stage_dir = args.output / stage_name
        
#         for cluster_id in clusters_to_process:
#             print(f"\n{stage_name} - Cluster {cluster_id}")
            
#             segments = find_consecutive_segments(
#                 stage_labels, cluster_id, video_boundaries, args.max_frames
#             )
            
#             if not segments:
#                 print(f"  No segments found")
#                 continue
            
#             output_path = stage_dir / f"cluster_{cluster_id:02d}.mp4"
#             create_cluster_video(
#                 sequences, emb_names, segments, output_path,
#                 fps=args.fps, max_frames=args.max_frames
#             )
    
#     # Save cluster stats
#     stats = {}
#     for stage_name, stage_labels in labels.items():
#         unique, counts = np.unique(stage_labels, return_counts=True)
#         stats[stage_name] = {int(u): int(c) for u, c in zip(unique, counts)}
    
#     with open(args.output / "cluster_stats.json", "w") as f:
#         json.dump(stats, f, indent=2)
    
#     print(f"\nDone! Videos saved to {args.output}")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Generate videos for each cluster showing consecutive frames belonging to that cluster.
Creates 15 videos per stage (3 stages × 15 clusters = 45 videos total).
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Keypoint visualization
# -----------------------------------------------------------------------------

# Skeleton connections for mouse
SKELETON = [
    ("nose", "head_midpoint"),
    ("head_midpoint", "left_ear"),
    ("head_midpoint", "right_ear"),
    ("left_ear", "left_ear_tip"),
    ("right_ear", "right_ear_tip"),
    ("head_midpoint", "left_eye"),
    ("head_midpoint", "right_eye"),
    ("head_midpoint", "neck"),
    ("neck", "mid_back"),
    ("mid_back", "mouse_center"),
    ("mouse_center", "mid_backend"),
    ("mid_backend", "mid_backend2"),
    ("mid_backend2", "mid_backend3"),
    ("mid_backend3", "tail_base"),
    ("tail_base", "tail1"),
    ("tail1", "tail2"),
    ("tail2", "tail3"),
    ("tail3", "tail4"),
    ("tail4", "tail5"),
    ("tail5", "tail_end"),
    ("neck", "left_shoulder"),
    ("left_shoulder", "left_midside"),
    ("left_midside", "left_hip"),
    ("left_hip", "tail_base"),
    ("neck", "right_shoulder"),
    ("right_shoulder", "right_midside"),
    ("right_midside", "right_hip"),
    ("right_hip", "tail_base"),
]

ALL_KEYPOINTS = [
    "nose", "left_ear", "right_ear", "left_ear_tip", "right_ear_tip",
    "left_eye", "right_eye", "neck", "mid_back", "mouse_center",
    "mid_backend", "mid_backend2", "mid_backend3", "tail_base",
    "tail1", "tail2", "tail3", "tail4", "tail5",
    "left_shoulder", "left_midside", "left_hip",
    "right_shoulder", "right_midside", "right_hip",
    "tail_end", "head_midpoint",
]

KPT_TO_IDX = {name: i for i, name in enumerate(ALL_KEYPOINTS)}


def get_skeleton_indices():
    """Convert skeleton connections to index pairs."""
    indices = []
    for start, end in SKELETON:
        if start in KPT_TO_IDX and end in KPT_TO_IDX:
            indices.append((KPT_TO_IDX[start], KPT_TO_IDX[end]))
    return indices


def draw_pose(ax, keypoints, skeleton_idx, color='blue', alpha=1.0):
    """Draw a single pose on an axis."""
    # keypoints: (num_keypoints, 2)
    ax.scatter(keypoints[:, 0], keypoints[:, 1], s=10, c=color, alpha=alpha, zorder=2)
    
    for i, j in skeleton_idx:
        if not np.any(np.isnan(keypoints[[i, j]])):
            ax.plot(
                [keypoints[i, 0], keypoints[j, 0]],
                [keypoints[i, 1], keypoints[j, 1]],
                c=color, alpha=alpha * 0.7, linewidth=1, zorder=1
            )


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_raw_keypoints(data_path: Path) -> Tuple[dict, List[str]]:
    """Load raw keypoints from .npy file."""
    raw = np.load(data_path, allow_pickle=True).item()
    sequences = raw["sequences"]
    names = list(sequences.keys())
    return sequences, names


def load_all_embeddings(emb_root: Path) -> Tuple[dict, List[str], List[int]]:
    """Load all embeddings and track video boundaries."""
    video_dirs = sorted([d for d in emb_root.iterdir() if d.is_dir()])
    
    all_stage1, all_stage2, all_stage3 = [], [], []
    video_names = []
    video_boundaries = [0]
    
    for vdir in video_dirs:
        emb_file = vdir / "embeddings.npz"
        if not emb_file.exists():
            continue
        
        data = np.load(emb_file)
        all_stage1.append(data["stage1"])
        all_stage2.append(data["stage2"])
        all_stage3.append(data["stage3"])
        video_names.append(vdir.name)
        video_boundaries.append(video_boundaries[-1] + len(data["stage1"]))
    
    stages = {
        "stage1": np.concatenate(all_stage1, axis=0),
        "stage2": np.concatenate(all_stage2, axis=0),
        "stage3": np.concatenate(all_stage3, axis=0),
    }
    
    return stages, video_names, video_boundaries


def global_to_local_frame(global_idx: int, video_boundaries: List[int]) -> Tuple[int, int]:
    """Convert global frame index to (video_idx, local_frame_idx)."""
    for vid_idx in range(len(video_boundaries) - 1):
        if video_boundaries[vid_idx] <= global_idx < video_boundaries[vid_idx + 1]:
            local_idx = global_idx - video_boundaries[vid_idx]
            return vid_idx, local_idx
    raise ValueError(f"Global index {global_idx} out of bounds")


# -----------------------------------------------------------------------------
# Clustering
# -----------------------------------------------------------------------------

def cluster_embeddings(stages: dict, n_clusters: int, subsample: int = 100) -> Tuple[dict, dict, dict]:
    """Cluster each stage separately. Returns labels, scaled embeddings, and kmeans models."""
    labels = {}
    scaled_embeddings = {}
    kmeans_models = {}
    
    for stage_name, emb in stages.items():
        print(f"Clustering {stage_name}...")
        scaler = StandardScaler()
        emb_scaled = scaler.fit_transform(emb)
        scaled_embeddings[stage_name] = emb_scaled
        
        # Subsample for fitting
        idx = np.arange(0, len(emb), subsample)
        emb_sub = emb_scaled[idx]
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(emb_sub)
        kmeans_models[stage_name] = kmeans
        
        # Predict on all
        labels[stage_name] = kmeans.predict(emb_scaled)
    
    return labels, scaled_embeddings, kmeans_models


# -----------------------------------------------------------------------------
# Video generation
# -----------------------------------------------------------------------------

def find_all_segments(
    cluster_labels: np.ndarray,
    cluster_id: int,
    video_boundaries: List[int],
    min_segment_length: int = 10,
) -> List[Tuple[int, int, int, int, int]]:
    """
    Find ALL consecutive segments belonging to a cluster.
    Returns list of (video_idx, start_local, end_local, start_global, end_global) tuples.
    Filters out segments shorter than min_segment_length.
    """
    segments = []
    in_segment = False
    seg_start = None
    
    for i, label in enumerate(cluster_labels):
        # Check if we crossed a video boundary
        at_boundary = i in video_boundaries[1:]
        
        if at_boundary and in_segment:
            # End segment at video boundary
            vid_idx, local_start = global_to_local_frame(seg_start, video_boundaries)
            _, local_end = global_to_local_frame(i - 1, video_boundaries)
            seg_len = local_end + 1 - local_start
            if seg_len >= min_segment_length:
                segments.append((vid_idx, local_start, local_end + 1, seg_start, i))
            in_segment = False
        
        if label == cluster_id:
            if not in_segment:
                seg_start = i
                in_segment = True
        else:
            if in_segment:
                vid_idx, local_start = global_to_local_frame(seg_start, video_boundaries)
                _, local_end = global_to_local_frame(i - 1, video_boundaries)
                seg_len = local_end + 1 - local_start
                if seg_len >= min_segment_length:
                    segments.append((vid_idx, local_start, local_end + 1, seg_start, i))
                in_segment = False
    
    # Handle final segment
    if in_segment:
        vid_idx, local_start = global_to_local_frame(seg_start, video_boundaries)
        _, local_end = global_to_local_frame(len(cluster_labels) - 1, video_boundaries)
        seg_len = local_end + 1 - local_start
        if seg_len >= min_segment_length:
            segments.append((vid_idx, local_start, local_end + 1, seg_start, len(cluster_labels)))
    
    return segments


def find_most_representative_segments(
    cluster_labels: np.ndarray,
    cluster_id: int,
    embeddings: np.ndarray,
    cluster_center: np.ndarray,
    video_boundaries: List[int],
    max_frames: int = 1800,
    min_segment_length: int = 10,
) -> List[Tuple[int, int, int]]:
    """
    Find consecutive segments closest to the cluster center.
    Returns list of (video_idx, start_local, end_local) tuples.
    Segments are sorted by distance to cluster center (closest first).
    """
    # Find all segments
    all_segments = find_all_segments(
        cluster_labels, cluster_id, video_boundaries, min_segment_length
    )
    
    if not all_segments:
        return []
    
    # Compute mean distance to cluster center for each segment
    segment_distances = []
    for vid_idx, local_start, local_end, global_start, global_end in all_segments:
        # Get embeddings for this segment
        seg_emb = embeddings[global_start:global_end]
        # Mean distance to cluster center
        distances = np.linalg.norm(seg_emb - cluster_center, axis=1)
        mean_dist = distances.mean()
        segment_distances.append((mean_dist, vid_idx, local_start, local_end, global_end - global_start))
    
    # Sort by distance (closest first)
    segment_distances.sort(key=lambda x: x[0])
    
    # Select segments until we reach max_frames
    selected = []
    total_frames = 0
    
    for dist, vid_idx, local_start, local_end, seg_len in segment_distances:
        if total_frames + seg_len > max_frames:
            # Take partial segment if it fits
            remaining = max_frames - total_frames
            if remaining >= min_segment_length:
                selected.append((vid_idx, local_start, local_start + remaining))
                total_frames += remaining
            break
        
        selected.append((vid_idx, local_start, local_end))
        total_frames += seg_len
        
        if total_frames >= max_frames:
            break
    
    return selected


def create_cluster_video(
    sequences: dict,
    video_names: List[str],
    segments: List[Tuple[int, int, int]],
    output_path: Path,
    fps: int = 30,
    max_frames: int = 1800,
):
    """Create video from consecutive segments."""
    skeleton_idx = get_skeleton_indices()
    
    # Collect frames
    frames_data = []
    for vid_idx, start, end in segments:
        if len(frames_data) >= max_frames:
            break
        
        vid_name = video_names[vid_idx]
        if vid_name not in sequences:
            continue
        
        kpts = sequences[vid_name]["keypoints"]  # (T, 1, K, 2)
        
        for t in range(start, min(end, kpts.shape[0])):
            if len(frames_data) >= max_frames:
                break
            frames_data.append((vid_name, t, kpts[t, 0]))  # (vid_name, frame_idx, keypoints)
    
    if len(frames_data) == 0:
        print(f"  No frames found, skipping")
        return
    
    print(f"  Creating video with {len(frames_data)} frames...")
    
    # Create video
    fig, ax = plt.subplots(figsize=(6, 6))
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='hBehaveMAE'))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with writer.saving(fig, str(output_path), dpi=100):
        for i, (vid_name, frame_idx, kpts) in enumerate(frames_data):
            ax.clear()
            
            draw_pose(ax, kpts, skeleton_idx, color='blue')
            
            ax.set_xlim(0, 500)
            ax.set_ylim(500, 0)  # Flip y for image coordinates
            ax.set_aspect('equal')
            ax.set_title(f"Frame {i+1}/{len(frames_data)} | {vid_name}:{frame_idx}")
            ax.axis('off')
            
            writer.grab_frame()
            
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(frames_data)} frames written")
    
    plt.close(fig)
    print(f"  Saved: {output_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Raw keypoints .npy file")
    parser.add_argument("--emb-root", type=Path, default=Path("output_embedings"))
    parser.add_argument("--output", type=Path, default=Path("cluster_videos"))
    parser.add_argument("--n-clusters", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=1800)
    parser.add_argument("--min-segment-length", type=int, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stages", nargs="+", default=["stage1", "stage2", "stage3"])
    parser.add_argument("--clusters", type=int, nargs="+", default=None, help="Specific clusters to generate")
    args = parser.parse_args()
    
    print("Loading raw keypoints...")
    sequences, raw_names = load_raw_keypoints(args.data)
    print(f"  Loaded {len(raw_names)} sequences")
    
    print("Loading embeddings...")
    stages, emb_names, video_boundaries = load_all_embeddings(args.emb_root)
    print(f"  Loaded {len(emb_names)} videos, {stages['stage1'].shape[0]} total frames")
    
    print("Clustering...")
    labels, scaled_embeddings, kmeans_models = cluster_embeddings(stages, args.n_clusters)
    
    # Generate videos
    clusters_to_process = args.clusters if args.clusters else list(range(args.n_clusters))
    
    for stage_name in args.stages:
        stage_labels = labels[stage_name]
        stage_emb = scaled_embeddings[stage_name]
        stage_kmeans = kmeans_models[stage_name]
        stage_dir = args.output / stage_name
        
        for cluster_id in clusters_to_process:
            print(f"\n{stage_name} - Cluster {cluster_id}")
            
            # Get cluster center
            cluster_center = stage_kmeans.cluster_centers_[cluster_id]
            
            # Find most representative segments
            segments = find_most_representative_segments(
                stage_labels, 
                cluster_id, 
                stage_emb,
                cluster_center,
                video_boundaries, 
                args.max_frames,
                args.min_segment_length,
            )
            
            if not segments:
                print(f"  No segments found")
                continue
            
            output_path = stage_dir / f"cluster_{cluster_id:02d}.mp4"
            create_cluster_video(
                sequences, emb_names, segments, output_path,
                fps=args.fps, max_frames=args.max_frames
            )
    
    # Save cluster stats
    stats = {}
    for stage_name, stage_labels in labels.items():
        unique, counts = np.unique(stage_labels, return_counts=True)
        stats[stage_name] = {int(u): int(c) for u, c in zip(unique, counts)}
    
    with open(args.output / "cluster_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDone! Videos saved to {args.output}")


if __name__ == "__main__":
    main()