"""
Evaluation Script: Analyse pipeline output CSV and generate metrics + plots.
=============================================================================

Usage:
    python evaluate_results.py --csv outputs/log_VIDEO_NAME.csv --output outputs/evaluation/
    python evaluate_results.py --csv outputs/log_VIDEO_NAME.csv --labels spot_check.json --output outputs/evaluation/

Spot-check JSON format (create by watching output video):
{
    "video_fps": 24,
    "notes": "Crowd ADL video, no actual falls",
    "spot_checks": [
        {"frame": 50,  "actual_persons": 12, "actual_falls": 0},
        {"frame": 100, "actual_persons": 15, "actual_falls": 0},
        {"frame": 200, "actual_persons": 20, "actual_falls": 0}
    ]
}
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path):
    """Load pipeline output CSV and normalise column names."""
    df = pd.read_csv(path)

    # Normalise column names to handle both logger formats
    rename_map = {}
    cols = df.columns.tolist()
    # Map: actual logger name -> standard name used in this script
    if "frame" in cols and "frame_number" not in cols:
        rename_map["frame"] = "frame_number"
    if "confidence" in cols and "detection_confidence" not in cols:
        rename_map["confidence"] = "detection_confidence"
    if "x1" in cols and "bbox_x1" not in cols:
        rename_map["x1"] = "bbox_x1"
        rename_map["y1"] = "bbox_y1"
        rename_map["x2"] = "bbox_x2"
        rename_map["y2"] = "bbox_y2"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure detection_confidence is numeric
    if "detection_confidence" in df.columns:
        df["detection_confidence"] = pd.to_numeric(df["detection_confidence"], errors="coerce").fillna(0)

    print(f"Loaded {len(df)} detection records from {path}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Frames: {df['frame_number'].min()} to {df['frame_number'].max()}")
    print(f"  Unique tracks: {df['track_id'].nunique()}")
    return df


# =====================================================
# Mode 1: Automatic Analysis
# =====================================================

def analyse_detection(df):
    per_frame = df.groupby("frame_number").agg(
        num_persons=("track_id", "count"),
        avg_confidence=("detection_confidence", "mean"),
    ).reset_index()

    stats = {
        "total_frames": int(per_frame["frame_number"].max()),
        "frames_with_detections": len(per_frame),
        "total_detections": len(df),
        "unique_tracks": int(df["track_id"].nunique()),
        "max_simultaneous": int(per_frame["num_persons"].max()),
        "avg_persons_per_frame": round(float(per_frame["num_persons"].mean()), 1),
        "avg_detection_confidence": round(float(df["detection_confidence"].mean()), 3),
        "min_detection_confidence": round(float(df["detection_confidence"].min()), 3),
    }
    return stats, per_frame


def analyse_tracking(df):
    track_stats = df.groupby("track_id").agg(
        first_frame=("frame_number", "min"),
        last_frame=("frame_number", "max"),
        num_detections=("frame_number", "count"),
    ).reset_index()
    track_stats["duration_frames"] = track_stats["last_frame"] - track_stats["first_frame"] + 1
    track_stats["detection_rate"] = track_stats["num_detections"] / track_stats["duration_frames"]

    stats = {
        "total_tracks": len(track_stats),
        "avg_track_duration_frames": round(float(track_stats["duration_frames"].mean()), 1),
        "max_track_duration_frames": int(track_stats["duration_frames"].max()),
        "median_track_duration_frames": int(track_stats["duration_frames"].median()),
        "avg_detection_rate": round(float(track_stats["detection_rate"].mean()), 3),
        "tracks_lasting_1_frame": int((track_stats["duration_frames"] == 1).sum()),
        "tracks_lasting_10plus_frames": int((track_stats["duration_frames"] >= 10).sum()),
    }
    return stats, track_stats


def analyse_falls(df):
    fall_df = df[df["activity"].str.contains("FALL", case=False, na=False)]
    if len(fall_df) == 0:
        return {"total_fall_detections": 0, "confirmed_falls": 0, "fall_warnings": 0,
                "tracks_with_falls": 0, "fall_rate_per_track": 0.0}, fall_df

    confirmed = df[df["activity"] == "FALL"]
    warnings = df[df["activity"] == "FALL WARNING"]
    fall_tracks = fall_df["track_id"].nunique()

    stats = {
        "total_fall_detections": len(fall_df),
        "confirmed_falls": len(confirmed),
        "fall_warnings": len(warnings),
        "tracks_with_falls": fall_tracks,
        "fall_rate_per_track": round(fall_tracks / max(df["track_id"].nunique(), 1), 3),
    }
    return stats, fall_df


def analyse_performance(df):
    timestamps = df.groupby("frame_number")["timestamp"].first().sort_index()
    if len(timestamps) < 2:
        return {"note": "Not enough frames"}
    frame_range = timestamps.index.max() - timestamps.index.min()
    time_range = float(timestamps.iloc[-1] - timestamps.iloc[0])
    return {
        "total_frames_processed": int(frame_range),
        "video_duration_seconds": round(time_range, 1),
        "estimated_video_fps": round(frame_range / max(time_range, 0.001), 1),
    }


# =====================================================
# Mode 2: Spot-Check Evaluation
# =====================================================

def evaluate_spot_checks(df, labels_path):
    with open(labels_path) as f:
        labels = json.load(f)

    results = []
    total_tp = total_fp = total_fn = 0
    fall_tp = fall_fp = fall_fn = 0

    for check in labels["spot_checks"]:
        frame = check["frame"]
        actual_persons = check["actual_persons"]
        actual_falls = check["actual_falls"]

        frame_df = df[df["frame_number"] == frame]
        pred_persons = len(frame_df)
        pred_falls = len(frame_df[frame_df["activity"].str.contains("FALL", case=False, na=False)])

        tp_p = min(pred_persons, actual_persons)
        fp_p = max(0, pred_persons - actual_persons)
        fn_p = max(0, actual_persons - pred_persons)
        total_tp += tp_p; total_fp += fp_p; total_fn += fn_p

        tp_f = min(pred_falls, actual_falls)
        fp_f = max(0, pred_falls - actual_falls)
        fn_f = max(0, actual_falls - pred_falls)
        fall_tp += tp_f; fall_fp += fp_f; fall_fn += fn_f

        results.append({
            "frame": frame, "actual_persons": actual_persons, "predicted_persons": pred_persons,
            "person_error": pred_persons - actual_persons,
            "actual_falls": actual_falls, "predicted_falls": pred_falls,
            "fall_error": pred_falls - actual_falls,
        })

    det_prec = total_tp / max(total_tp + total_fp, 1)
    det_rec = total_tp / max(total_tp + total_fn, 1)
    det_f1 = 2 * det_prec * det_rec / max(det_prec + det_rec, 1e-6)
    fall_prec = fall_tp / max(fall_tp + fall_fp, 1)
    fall_rec = fall_tp / max(fall_tp + fall_fn, 1)
    fall_f1 = 2 * fall_prec * fall_rec / max(fall_prec + fall_rec, 1e-6)

    return {
        "spot_check_count": len(results),
        "detection_precision": round(det_prec, 4),
        "detection_recall": round(det_rec, 4),
        "detection_f1": round(det_f1, 4),
        "fall_precision": round(fall_prec, 4),
        "fall_recall": round(fall_rec, 4),
        "fall_f1": round(fall_f1, 4),
        "avg_person_count_error": round(np.mean([abs(r["person_error"]) for r in results]), 2),
        "false_fall_rate": round(fall_fp / max(len(results), 1), 3),
        "details": results,
    }


# =====================================================
# Plotting
# =====================================================

def generate_plots(df, per_frame, track_stats, fall_stats, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Persons over time
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(per_frame["frame_number"], per_frame["num_persons"], linewidth=0.8, color="#2196F3")
    ax.fill_between(per_frame["frame_number"], per_frame["num_persons"], alpha=0.2, color="#2196F3")
    ax.set_xlabel("Frame Number"); ax.set_ylabel("Detected Persons")
    ax.set_title("Person Count Over Time"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "persons_over_time.png"), dpi=150); plt.close()
    print(f"  Saved: persons_over_time.png")

    # 2. Detection confidence distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["detection_confidence"], bins=50, color="#4CAF50", edgecolor="white", alpha=0.8)
    ax.axvline(df["detection_confidence"].mean(), color="red", linestyle="--",
               label=f"Mean: {df['detection_confidence'].mean():.3f}")
    ax.set_xlabel("Detection Confidence"); ax.set_ylabel("Count")
    ax.set_title("Detection Confidence Distribution"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "confidence_distribution.png"), dpi=150); plt.close()
    print(f"  Saved: confidence_distribution.png")

    # 3. Track duration histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    durations = track_stats["duration_frames"]
    ax.hist(durations, bins=min(50, len(durations)), color="#9C27B0", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Track Duration (frames)"); ax.set_ylabel("Number of Tracks")
    ax.set_title("Tracking Persistence Distribution")
    ax.axvline(durations.median(), color="red", linestyle="--", label=f"Median: {durations.median():.0f} frames")
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "track_duration.png"), dpi=150); plt.close()
    print(f"  Saved: track_duration.png")

    # 4. Activity distribution pie chart
    activity_counts = df["activity"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 7))
    colors_map = {"normal": "#4CAF50", "FALL": "#F44336", "FALL WARNING": "#FF9800"}
    pie_colors = [colors_map.get(a, "#607D8B") for a in activity_counts.index]
    ax.pie(activity_counts.values, labels=activity_counts.index, colors=pie_colors,
           autopct="%1.1f%%", startangle=90)
    ax.set_title("Activity Classification Distribution")
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "activity_distribution.png"), dpi=150); plt.close()
    print(f"  Saved: activity_distribution.png")

    # 5. Falls timeline
    fall_df = df[df["activity"].str.contains("FALL", case=False, na=False)]
    if len(fall_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 4))
        fall_per_frame = fall_df.groupby("frame_number")["track_id"].count()
        ax.bar(fall_per_frame.index, fall_per_frame.values, width=1.0, color="#F44336", alpha=0.7)
        ax.set_xlabel("Frame Number"); ax.set_ylabel("Fall Detections")
        ax.set_title("Fall Events Over Time"); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, "falls_timeline.png"), dpi=150); plt.close()
        print(f"  Saved: falls_timeline.png")


# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline results")
    parser.add_argument("--csv", required=True, help="Path to pipeline output CSV")
    parser.add_argument("--labels", default=None, help="Path to spot-check JSON (optional)")
    parser.add_argument("--output", default="outputs/evaluation/")
    args = parser.parse_args()

    df = load_csv(args.csv)
    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*60}")
    print("EVALUATION REPORT")
    print(f"{'='*60}")

    print("\n--- Detection Statistics ---")
    det_stats, per_frame = analyse_detection(df)
    for k, v in det_stats.items(): print(f"  {k}: {v}")

    print("\n--- Tracking Analysis ---")
    trk_stats, track_df = analyse_tracking(df)
    for k, v in trk_stats.items(): print(f"  {k}: {v}")

    print("\n--- Fall Detection Analysis ---")
    fall_stats, fall_df = analyse_falls(df)
    for k, v in fall_stats.items(): print(f"  {k}: {v}")

    print("\n--- Performance ---")
    perf_stats = analyse_performance(df)
    for k, v in perf_stats.items(): print(f"  {k}: {v}")

    spot_metrics = None
    if args.labels and os.path.exists(args.labels):
        print(f"\n--- Spot-Check Evaluation ---")
        spot_metrics = evaluate_spot_checks(df, args.labels)
        print(f"  Detection Precision: {spot_metrics['detection_precision']:.4f}")
        print(f"  Detection Recall:    {spot_metrics['detection_recall']:.4f}")
        print(f"  Detection F1:        {spot_metrics['detection_f1']:.4f}")
        print(f"  Fall Precision:      {spot_metrics['fall_precision']:.4f}")
        print(f"  Fall Recall:         {spot_metrics['fall_recall']:.4f}")
        print(f"  Fall F1:             {spot_metrics['fall_f1']:.4f}")
        print(f"  Avg person count error: {spot_metrics['avg_person_count_error']}")
        print(f"\n  Per-frame details:")
        for d in spot_metrics["details"]:
            print(f"    Frame {d['frame']}: persons {d['actual_persons']}→{d['predicted_persons']} "
                  f"(err={d['person_error']:+d}), falls {d['actual_falls']}→{d['predicted_falls']} "
                  f"(err={d['fall_error']:+d})")

    print(f"\n--- Generating Plots ---")
    generate_plots(df, per_frame, track_df, fall_stats, args.output)

    report = {"detection": det_stats, "tracking": trk_stats, "falls": fall_stats, "performance": perf_stats}
    if spot_metrics: report["spot_check_evaluation"] = spot_metrics
    report_path = os.path.join(args.output, "evaluation_report.json")
    with open(report_path, "w") as f: json.dump(report, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Persons: {det_stats['unique_tracks']} unique, {det_stats['max_simultaneous']} max simultaneous")
    print(f"  Avg confidence: {det_stats['avg_detection_confidence']}")
    print(f"  Track persistence: {trk_stats['avg_track_duration_frames']} avg frames")
    print(f"  Falls: {fall_stats.get('confirmed_falls',0)} confirmed, {fall_stats.get('fall_warnings',0)} warnings")
    if spot_metrics:
        print(f"  Detection F1: {spot_metrics['detection_f1']:.4f}")
        print(f"  Fall F1:      {spot_metrics['fall_f1']:.4f}")
    print(f"  Report: {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()