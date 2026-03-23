"""MediaPipe pose extraction + physics-based fall features.

Updated for MediaPipe 0.10.30+ (new Task API):
  - Uses mp.tasks.vision.PoseLandmarker instead of mp.solutions.pose.Pose
  - Auto-downloads the pose_landmarker_heavy.task model on first run
  - Uses detect_for_video() with monotonic timestamps

The 5 physics features (from HumanFallDetection reference):
  1. ratio_bbox    — width/height of bounding box
  2. log_angle     — log(1 + |torso angle from vertical|)
  3. re            — rotational energy of head+torso
  4. ratio_deriv   — rate of change of bbox ratio
  5. gf            — generalised force (inverted pendulum model)
"""
import numpy as np
import cv2
import math
import os
import urllib.request
from typing import Optional, Dict
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# MediaPipe landmark indices (same 33 as before)
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 2, 5, 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26

SKELETON_DRAW = [
    (11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28)
]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
MODEL_FILENAME = "pose_landmarker_heavy.task"


def _ensure_model(models_dir="models"):
    """Download the pose landmarker model if not present."""
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, MODEL_FILENAME)
    if not os.path.exists(model_path):
        print(f"[PoseExtractor] Downloading pose model to {model_path}...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print(f"[PoseExtractor] Download complete.")
    return model_path


class PoseExtractor:
    """Extracts MediaPipe keypoints and computes physics-based fall features.
    Uses the new MediaPipe Tasks API (0.10.30+).
    """

    def __init__(self, model_complexity=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, sequence_length=36,
                 models_dir="models"):
        model_path = _ensure_model(models_dir)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self.sequence_length = sequence_length
        self._state: Dict[int, dict] = {}
        self._timestamp_ms = 0
        print(f"[PoseExtractor] MediaPipe Tasks API | model={MODEL_FILENAME} | seq_len={sequence_length}")

    def _extract_inverted_pendulum(self, landmarks, w, h) -> Optional[dict]:
        """Extract H (head), N (neck/mid-shoulders), B (base/mid-hips)."""
        kp = {}
        head_indices = [L_EAR, R_EAR, L_EYE, R_EYE]
        numx = sum(landmarks[i].visibility * landmarks[i].x * w for i in head_indices)
        numy = sum(landmarks[i].visibility * landmarks[i].y * h for i in head_indices)
        den = sum(landmarks[i].visibility for i in head_indices)
        kp['H'] = np.array([numx/den, numy/den]) if den > 0.01 else None

        ls, rs = landmarks[L_SHOULDER], landmarks[R_SHOULDER]
        if ls.visibility > 0.3 and rs.visibility > 0.3:
            kp['N'] = np.array([(ls.x+rs.x)/2*w, (ls.y+rs.y)/2*h])
        else:
            kp['N'] = None

        lh, rh = landmarks[L_HIP], landmarks[R_HIP]
        if lh.visibility > 0.3 and rh.visibility > 0.3:
            kp['B'] = np.array([(lh.x+rh.x)/2*w, (lh.y+rh.y)/2*h])
        else:
            kp['B'] = None

        if kp['N'] is None or kp['B'] is None:
            return None
        return kp

    def _compute_physics_features(self, curr_kp, prev_kp, prev2_kp, bbox, dt=1/30) -> np.ndarray:
        features = np.zeros(5, dtype=np.float32)
        bw, bh = bbox[2]-bbox[0], bbox[3]-bbox[1]
        ratio = bw / max(bh, 1)
        features[0] = ratio

        body_vec = curr_kp['N'] - curr_kp['B']
        angle_vert = math.atan2(-body_vec[0], -body_vec[1])
        features[1] = np.log(1 + abs(angle_vert))

        if prev_kp is not None and prev_kp.get('N') is not None and prev_kp.get('B') is not None:
            N1 = curr_kp['N'] - curr_kp['B']
            N0 = prev_kp['N'] - prev_kp['B']
            d2sq = N1.dot(N1)
            angle_N = math.atan2(np.linalg.det([N0, N1]), np.dot(N0, N1))
            w2sq = (angle_N / dt) ** 2
            energy = 5 * d2sq * w2sq
            if curr_kp['H'] is not None and prev_kp.get('H') is not None:
                H1 = curr_kp['H'] - curr_kp['B']
                H0 = prev_kp['H'] - prev_kp['B']
                d1sq = H1.dot(H1)
                angle_H = math.atan2(np.linalg.det([H0, H1]), np.dot(H0, H1))
                energy += 1 * d1sq * (angle_H/dt)**2
                den = 5*d2sq + 1*d1sq
            else:
                den = 5*d2sq
            features[2] = energy / (2*den + 1e-6)
            prev_ratio = prev_kp.get('bbox_w', bw) / max(prev_kp.get('bbox_h', bh), 1)
            features[3] = (ratio - prev_ratio) / dt
        if (prev_kp is not None and prev2_kp is not None and
                all(prev_kp.get(k) is not None for k in ['H','N','B']) and
                all(prev2_kp.get(k) is not None for k in ['H','N','B'])):
            features[4] = self._compute_gf(prev2_kp, prev_kp, curr_kp, dt)
        return features

    def _compute_gf(self, ip0, ip1, ip2, dt):
        m1, m2, g = 1, 15, 10
        H2=ip2['H']-ip2['N']; H1=ip1['H']-ip1['N']; H0=ip0['H']-ip0['N']
        N2=ip2['N']-ip2['B']; N1=ip1['N']-ip1['B']; N0=ip0['N']-ip0['B']
        d1=np.sqrt(H1.dot(H1)); d2=np.sqrt(N1.dot(N1))
        if d2<1e-6: return 0.0
        d1_n=d1/d2; d2_n=1.0
        def av(v): return math.atan2(-v[0],-v[1])
        def ga(v0,v1): return math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
        theta1=av(H1)-av(N1); theta2=av(N1)
        dt1_0,dt1_1=ga(H0,H1)/dt, ga(H1,H2)/dt
        dt2_0,dt2_1=ga(N0,N1)/dt, ga(N1,N2)/dt
        dt1=0.5*(dt1_0+dt1_1); dt2=0.5*(dt2_0+dt2_1)
        ddt1=(dt1_1-dt1_0)/(dt); ddt2=(dt2_1-dt2_0)/(dt)
        Q1=m1*d1_n*ddt1**2+(m1*d1_n**2+m1*d1_n*d2_n*np.cos(theta1))*ddt2
        Q1+=m1*d1_n*d2_n*np.sin(theta1)*dt2**2-m1*g*d2_n*np.sin(theta1+theta2)
        Q2=(m1*d1_n**2+m1*d1_n*d2_n*np.cos(theta1))*ddt1
        Q2+=((m1+m2)*d2_n**2+m1*d1_n**2+2*m1*d1_n*d2_n*np.cos(theta1))*ddt2
        Q2-=2*m1*d1_n*d2_n*np.sin(theta1)*dt2*dt1+m1*d1_n*d2_n*np.sin(theta1)*dt1**2
        Q2-=(m1+m2)*g*d2_n*np.sin(theta2)+m1*g*d1_n*np.sin(theta1+theta2)
        return Q1+Q2

    def extract_for_track(self, frame, track) -> Optional[np.ndarray]:
        """Extract pose and compute 5 physics features for one tracked person."""
        h, w = frame.shape[:2]
        bbox = track.bbox
        bw, bh = bbox[2]-bbox[0], bbox[3]-bbox[1]
        pad = 0.1
        x1, y1 = max(0, int(bbox[0]-bw*pad)), max(0, int(bbox[1]-bh*pad))
        x2, y2 = min(w, int(bbox[2]+bw*pad)), min(h, int(bbox[3]+bh*pad))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        self._timestamp_ms += 33

        try:
            result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)
        except Exception:
            return None

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        landmarks = result.pose_landmarks[0]
        ch, cw = crop.shape[:2]
        kp = self._extract_inverted_pendulum(landmarks, cw, ch)
        if kp is None: return None

        for key in ['H', 'N', 'B']:
            if kp[key] is not None:
                kp[key] = kp[key] + np.array([x1, y1])
        kp['bbox_w'] = bw
        kp['bbox_h'] = bh

        tid = track.track_id
        if tid not in self._state:
            self._state[tid] = {"kp_history": deque(maxlen=3), "feat_buffer": deque(maxlen=self.sequence_length)}
        state = self._state[tid]
        prev_kp = state["kp_history"][-1] if len(state["kp_history"]) >= 1 else None
        prev2_kp = state["kp_history"][-2] if len(state["kp_history"]) >= 2 else None
        state["kp_history"].append(kp)

        features = self._compute_physics_features(kp, prev_kp, prev2_kp, bbox)
        state["feat_buffer"].append(features)

        raw_kp = np.zeros(99, dtype=np.float32)
        for i, lm in enumerate(landmarks):
            raw_kp[i*3], raw_kp[i*3+1], raw_kp[i*3+2] = lm.x, lm.y, lm.visibility
        state["raw_kp"] = raw_kp

        return features

    def get_feature_sequence(self, track_id) -> Optional[np.ndarray]:
        if track_id not in self._state: return None
        buf = self._state[track_id]["feat_buffer"]
        if len(buf) < self.sequence_length: return None
        return np.array(list(buf), dtype=np.float32)

    def get_raw_keypoints(self, track_id) -> Optional[np.ndarray]:
        if track_id not in self._state: return None
        return self._state[track_id].get("raw_kp")

    def get_buffer_length(self, track_id):
        if track_id not in self._state: return 0
        return len(self._state[track_id]["feat_buffer"])

    def cleanup(self, active_ids: set):
        for sid in set(self._state) - active_ids: del self._state[sid]

    def release(self):
        if hasattr(self, 'landmarker'): self.landmarker.close()
        self._state.clear()