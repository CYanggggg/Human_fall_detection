"""
Fall detection on a single image.
Uses YOLOv8 detection + MediaPipe pose + physics-based rule analysis.

Note: Since a single image has no temporal data (no previous frames),
the LSTM cannot be used. This script uses rule-based fall detection
based on single-frame features: body angle and bounding box ratio.

Usage:
    python test_image_fall.py image.jpg
    python test_image_fall.py image.jpg --confidence 0.6 --model_size medium
"""
import cv2
import sys
import math
import argparse
import os
import numpy as np
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from modules.detector import PersonDetector

# MediaPipe landmark indices
L_EYE, R_EYE, L_EAR, R_EAR = 2, 5, 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24

# Skeleton for drawing
SKELETON = [
    (11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28)
]

# Colors (BGR)
COLOR_FALL = (0, 0, 255)       # Red
COLOR_WARNING = (0, 165, 255)  # Orange
COLOR_NORMAL = (0, 200, 0)     # Green

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
MODEL_FILE = "models/pose_landmarker_heavy.task"


def ensure_model():
    if not os.path.exists(MODEL_FILE):
        os.makedirs("models", exist_ok=True)
        print(f"Downloading pose model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print(f"Done.")
    return MODEL_FILE


def analyse_person(landmarks, bbox):
    """Analyse single-frame pose features for fall indicators.

    Returns: (label, confidence, features_dict)
      label: "FALL", "FALL WARNING", or "normal"
    """
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]

    # Feature 1: Bounding box ratio
    ratio = bw / max(bh, 1)

    # Feature 2: Body angle from vertical
    ls, rs = landmarks[L_SHOULDER], landmarks[R_SHOULDER]
    lh, rh = landmarks[L_HIP], landmarks[R_HIP]

    if ls.visibility < 0.3 or rs.visibility < 0.3 or lh.visibility < 0.3 or rh.visibility < 0.3:
        return "unknown", 0.0, {"ratio": ratio, "angle": 0, "note": "low visibility"}

    neck = np.array([(ls.x + rs.x) / 2, (ls.y + rs.y) / 2])
    base = np.array([(lh.x + rh.x) / 2, (lh.y + rh.y) / 2])
    body_vec = neck - base  # In normalised coords (0-1)
    angle = math.atan2(-body_vec[0], -body_vec[1])
    angle_deg = abs(math.degrees(angle))

    features = {
        "bbox_ratio": round(ratio, 3),
        "body_angle_deg": round(angle_deg, 1),
        "log_angle": round(math.log(1 + abs(angle)), 3),
    }

    # Rule-based classification
    if ratio > 1.5 and angle_deg > 60:
        return "FALL", 0.9, features
    elif ratio > 1.2 or angle_deg > 50:
        return "FALL WARNING", 0.6, features
    elif angle_deg > 35 and ratio > 0.8:
        return "FALL WARNING", 0.5, features
    else:
        return "normal", 0.9, features


def draw_skeleton(img, landmarks, bbox, color):
    """Draw pose skeleton on the image."""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pts = []
    for i in range(33):
        lm = landmarks[i]
        px = int(x1 + lm.x * bw)
        py = int(y1 + lm.y * bh)
        pts.append((px, py, lm.visibility))

    for s, e in SKELETON:
        if pts[s][2] > 0.5 and pts[e][2] > 0.5:
            cv2.line(img, (pts[s][0], pts[s][1]), (pts[e][0], pts[e][1]), color, 2, cv2.LINE_AA)
    for px, py, v in pts:
        if v > 0.5:
            cv2.circle(img, (px, py), 4, color, -1)


def main():
    parser = argparse.ArgumentParser(description="Fall detection on a single image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--model_size", default="medium", choices=["nano","small","medium","large","xlarge"])
    parser.add_argument("--no_skeleton", action="store_true", help="Don't draw skeleton")
    args = parser.parse_args()

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Cannot read: {args.image}")
        sys.exit(1)
    h, w = img.shape[:2]
    print(f"Image: {args.image} ({w}x{h})")

    # Detect persons
    detector = PersonDetector(model_size=args.model_size, confidence_threshold=args.confidence)
    detections = detector.detect(img)
    print(f"Detected {len(detections)} persons")

    # Setup MediaPipe PoseLandmarker
    model_path = ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    # Analyse each person
    fall_count = 0
    warning_count = 0
    normal_count = 0

    for i, det in enumerate(detections):
        bbox = det.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1

        # Crop and run pose estimation
        pad = 0.1
        cx1 = max(0, int(x1 - bw * pad))
        cy1 = max(0, int(y1 - bh * pad))
        cx2 = min(w, int(x2 + bw * pad))
        cy2 = min(h, int(y2 + bh * pad))
        crop = img[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)

        try:
            result = landmarker.detect(mp_image)
        except Exception:
            # No pose detected, draw detection only
            cv2.rectangle(img, (x1,y1), (x2,y2), COLOR_NORMAL, 2)
            cv2.putText(img, f"{det.confidence:.0%}", (x1,y1-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_NORMAL, 2)
            normal_count += 1
            continue

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            cv2.rectangle(img, (x1,y1), (x2,y2), COLOR_NORMAL, 2)
            cv2.putText(img, f"{det.confidence:.0%}", (x1,y1-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_NORMAL, 2)
            normal_count += 1
            continue

        landmarks = result.pose_landmarks[0]

        # Analyse for fall
        label, conf, features = analyse_person(landmarks, det.bbox)

        # Set color based on label
        if label == "FALL":
            color = COLOR_FALL
            fall_count += 1
        elif label == "FALL WARNING":
            color = COLOR_WARNING
            warning_count += 1
        else:
            color = COLOR_NORMAL
            normal_count += 1

        # Draw bounding box
        thickness = 3 if "FALL" in label else 2
        cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)

        # Draw label
        text = f"#{i+1} {label} {det.confidence:.0%}"
        if label != "normal":
            text += f" angle:{features['body_angle_deg']}° ratio:{features['bbox_ratio']}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1-th-10), (x1+tw+4, y1), color, -1)
        cv2.putText(img, text, (x1+2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # Draw skeleton
        if not args.no_skeleton:
            draw_skeleton(img, landmarks, det.bbox, color)

        # Print details
        print(f"  Person #{i+1}: {label} (det:{det.confidence:.0%}) "
              f"angle={features.get('body_angle_deg','?')}° ratio={features.get('bbox_ratio','?')}")

    # Summary panel
    overlay = img.copy()
    cv2.rectangle(overlay, (10,10), (300,100), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    y = 30
    for line in [f"Persons: {len(detections)}", f"Normal: {normal_count}",
                 f"Warnings: {warning_count}", f"Falls: {fall_count}"]:
        col = (255,255,255)
        if "Falls" in line and fall_count > 0: col = COLOR_FALL
        cv2.putText(img, line, (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
        y += 22

    # Save
    out_path = args.image.rsplit(".", 1)[0] + "_fall_result.jpg"
    cv2.imwrite(out_path, img)
    print(f"\nSummary: {normal_count} normal, {warning_count} warnings, {fall_count} falls")
    print(f"Saved: {out_path}")

    landmarker.close()


if __name__ == "__main__":
    main()