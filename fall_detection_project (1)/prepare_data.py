"""
Step 2 & 3: Extract physics features from labelled videos and save as training data.
===============================================================================

This script is the bridge between your raw video data and the LSTM training.
It runs the full detection → tracking → pose → physics pipeline on each video,
then saves the 5-feature sequences with labels as .npz files.

Directory structure expected:
    data/
    ├── fall/          ← videos of people falling
    │   ├── fall_001.mp4
    │   ├── fall_002.mp4
    │   └── ...
    └── normal/        ← videos of normal activity (walking, sitting, standing)
        ├── normal_001.mp4
        ├── normal_002.mp4
        └── ...

Usage:
    python prepare_data.py --data_dir data/ --output data/training_data.npz --seq_len 36

What it does for each video:
    1. YOLOv8 detects all persons per frame
    2. ByteTrack assigns persistent track IDs
    3. MediaPipe extracts 33 keypoints per person
    4. Physics engine computes 5 features per person per frame
    5. Once a track has seq_len frames buffered, it saves that sequence
    6. Label comes from the folder name (fall/ or normal/)
"""

import argparse
import os
import sys
import glob
import numpy as np
import cv2
from pathlib import Path

# These imports require ultralytics, mediapipe, filterpy installed
from modules.detector import PersonDetector
from modules.tracker import ByteTracker
from modules.pose_extractor import PoseExtractor


def extract_sequences_from_video(
    video_path: str,
    detector: PersonDetector,
    tracker: ByteTracker,
    pose_extractor: PoseExtractor,
    seq_len: int = 36,
    frame_skip: int = 1,
    max_frames: int = -1,
) -> list:
    """Run the full pipeline on one video and extract all complete feature sequences.

    Returns:
        List of np.ndarray, each of shape (seq_len, 5).
        Each array is one person's complete temporal feature sequence.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sequences = []
    collected_tracks = set()  # Track IDs we already saved a sequence for
    fc = 0

    # Reset tracker for each video
    tracker.reset()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fc += 1
        if 0 < max_frames < fc:
            break
        if fc % frame_skip != 0:
            continue

        # Detection + Tracking
        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        active_ids = {t.track_id for t in tracks if t.is_active}

        # Pose + Physics Features
        for track in tracks:
            if not track.is_active:
                continue
            pose_extractor.extract_for_track(frame, track)

            # Check if this track has a full sequence buffered
            buf_len = pose_extractor.get_buffer_length(track.track_id)
            if buf_len >= seq_len and track.track_id not in collected_tracks:
                seq = pose_extractor.get_feature_sequence(track.track_id)
                if seq is not None:
                    sequences.append(seq)
                    collected_tracks.add(track.track_id)

            # Also collect overlapping windows (every seq_len//2 frames)
            # This increases training data via sliding window
            elif buf_len >= seq_len and buf_len % (seq_len // 2) == 0:
                seq = pose_extractor.get_feature_sequence(track.track_id)
                if seq is not None:
                    sequences.append(seq)

        pose_extractor.cleanup(active_ids)

    cap.release()
    return sequences


def main():
    parser = argparse.ArgumentParser(
        description="Extract training data from labelled fall/normal videos"
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Root directory with fall/ and normal/ subdirectories"
    )
    parser.add_argument(
        "--output", default="data/training_data.npz",
        help="Output .npz file path"
    )
    parser.add_argument("--seq_len", type=int, default=36, help="Sequence length")
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument(
        "--model_size", default="medium",
        choices=["nano", "small", "medium", "large", "xlarge"]
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Verify directory structure
    fall_dir = os.path.join(args.data_dir, "fall")
    normal_dir = os.path.join(args.data_dir, "normal")

    if not os.path.isdir(fall_dir) or not os.path.isdir(normal_dir):
        print("ERROR: data_dir must contain 'fall/' and 'normal/' subdirectories.")
        print(f"  Expected: {fall_dir}")
        print(f"  Expected: {normal_dir}")
        print("\nStructure your videos like this:")
        print("  data/")
        print("  ├── fall/")
        print("  │   ├── fall_001.mp4")
        print("  │   └── ...")
        print("  └── normal/")
        print("      ├── normal_001.mp4")
        print("      └── ...")
        sys.exit(1)

    # Find all video files
    video_exts = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI")
    fall_videos = []
    normal_videos = []
    for ext in video_exts:
        fall_videos.extend(glob.glob(os.path.join(fall_dir, ext)))
        normal_videos.extend(glob.glob(os.path.join(normal_dir, ext)))

    print(f"Found {len(fall_videos)} fall videos, {len(normal_videos)} normal videos")

    if not fall_videos and not normal_videos:
        print("ERROR: No video files found. Supported formats: mp4, avi, mov, mkv")
        sys.exit(1)

    # Initialise pipeline modules
    print("\nInitialising pipeline...")
    detector = PersonDetector(
        model_size=args.model_size,
        confidence_threshold=0.5,
        device=args.device
    )
    tracker = ByteTracker()
    pose_extractor = PoseExtractor(sequence_length=args.seq_len)

    all_sequences = []
    all_labels = []

    # Process fall videos (label = 0)
    print(f"\n--- Processing fall videos ({len(fall_videos)} files) ---")
    for i, vpath in enumerate(sorted(fall_videos)):
        print(f"  [{i+1}/{len(fall_videos)}] {os.path.basename(vpath)}...", end=" ")
        seqs = extract_sequences_from_video(
            vpath, detector, tracker, pose_extractor,
            args.seq_len, args.frame_skip, args.max_frames
        )
        print(f"→ {len(seqs)} sequences")
        all_sequences.extend(seqs)
        all_labels.extend([0] * len(seqs))  # 0 = fall

    # Process normal videos (label = 1)
    print(f"\n--- Processing normal videos ({len(normal_videos)} files) ---")
    for i, vpath in enumerate(sorted(normal_videos)):
        print(f"  [{i+1}/{len(normal_videos)}] {os.path.basename(vpath)}...", end=" ")
        seqs = extract_sequences_from_video(
            vpath, detector, tracker, pose_extractor,
            args.seq_len, args.frame_skip, args.max_frames
        )
        print(f"→ {len(seqs)} sequences")
        all_sequences.extend(seqs)
        all_labels.extend([1] * len(seqs))  # 1 = normal

    pose_extractor.release()

    if not all_sequences:
        print("\nERROR: No sequences extracted. Check your videos contain visible people.")
        sys.exit(1)

    # Convert to numpy and save
    seqs_array = np.array(all_sequences, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int64)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        seqs=seqs_array,
        labels=labels_array
    )

    n_fall = sum(labels_array == 0)
    n_normal = sum(labels_array == 1)
    print(f"\n{'='*50}")
    print(f"Dataset saved to: {args.output}")
    print(f"  Total sequences: {len(labels_array)}")
    print(f"  Fall sequences:  {n_fall}")
    print(f"  Normal sequences: {n_normal}")
    print(f"  Sequence shape:  ({args.seq_len}, 5)")
    print(f"  Feature order:   [ratio_bbox, log_angle, re, ratio_deriv, gf]")
    print(f"{'='*50}")

    if n_fall == 0 or n_normal == 0:
        print("\nWARNING: Dataset is missing one class! You need both fall and normal videos.")

    if abs(n_fall - n_normal) > max(n_fall, n_normal) * 0.5:
        print(f"\nWARNING: Dataset is imbalanced ({n_fall} falls vs {n_normal} normal).")
        print("Consider adding more videos for the underrepresented class,")
        print("or use class weights during training.")

    print(f"\nNext step: python train.py --data {args.output} --epochs 50")


if __name__ == "__main__":
    main()
