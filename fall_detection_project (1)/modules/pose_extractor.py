"""MediaPipe pose extraction + physics-based fall features (adapted from HumanFallDetection).

Combines your design's MediaPipe 33-keypoint extraction with the reference project's
5 hand-crafted physics features:
  1. ratio_bbox    — width/height ratio of bounding box (>1 = lying down)
  2. log_angle     — log(1 + |torso angle from vertical|)
  3. re            — rotational energy of head+torso (rapid rotation = falling)
  4. ratio_deriv   — rate of change of bbox ratio (sudden change = fall)
  5. gf            — generalised force on the inverted pendulum model
"""
import numpy as np
import cv2
import math
from typing import Optional, Dict, List, Tuple
from collections import deque

try:
    import mediapipe as mp
except ImportError:
    mp = None

# MediaPipe landmark indices
NOSE, L_SHOULDER, R_SHOULDER = 0, 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_EAR, R_EAR, L_EYE, R_EYE = 7, 8, 1, 2

SKELETON_DRAW = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),
                 (23,25),(25,27),(24,26),(26,28)]


class PoseExtractor:
    """Extracts MediaPipe keypoints and computes physics-based fall features per person."""

    def __init__(self, model_complexity=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, sequence_length=36):
        if mp is None: raise ImportError("pip install mediapipe")
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self.sequence_length = sequence_length
        # Per-track state: {track_id: {"keypoints": deque, "features": deque, "prev_kp": dict}}
        self._state: Dict[int, dict] = {}
        print(f"[PoseExtractor] MediaPipe + physics features | seq_len={sequence_length}")

    def _get_keypoint(self, landmarks, idx, w, h):
        """Get pixel coordinates and visibility for a landmark."""
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h]), lm.visibility

    def _extract_inverted_pendulum(self, landmarks, w, h) -> Optional[dict]:
        """Extract the inverted pendulum model points (H, N, B) from MediaPipe landmarks.
        Adapted from reference project's get_kp() function.
        H = head center, N = neck (mid-shoulders), B = base (mid-hips).
        """
        kp = {}
        # Head: weighted average of ears and eyes
        points = [(L_EAR, landmarks[L_EAR]), (R_EAR, landmarks[R_EAR]),
                  (L_EYE, landmarks[L_EYE]), (R_EYE, landmarks[R_EYE])]
        numx = sum(p.visibility * p.x * w for _, p in points)
        numy = sum(p.visibility * p.y * h for _, p in points)
        den = sum(p.visibility for _, p in points)
        kp['H'] = np.array([numx/den, numy/den]) if den > 0.01 else None

        # Neck: midpoint of shoulders
        ls, rs = landmarks[L_SHOULDER], landmarks[R_SHOULDER]
        if ls.visibility > 0.3 and rs.visibility > 0.3:
            kp['N'] = np.array([(ls.x + rs.x) / 2 * w, (ls.y + rs.y) / 2 * h])
        else:
            kp['N'] = None

        # Base: midpoint of hips
        lh, rh = landmarks[L_HIP], landmarks[R_HIP]
        if lh.visibility > 0.3 and rh.visibility > 0.3:
            kp['B'] = np.array([(lh.x + rh.x) / 2 * w, (lh.y + rh.y) / 2 * h])
        else:
            kp['B'] = None

        if kp['N'] is None or kp['B'] is None:
            return None
        return kp

    def _compute_physics_features(self, curr_kp: dict, prev_kp: Optional[dict],
                                   prev2_kp: Optional[dict], bbox, dt=1/30) -> np.ndarray:
        """Compute the 5 physics features from the reference project.
        Returns: np.array([ratio_bbox, log_angle, re, ratio_deriv, gf])
        """
        features = np.zeros(5, dtype=np.float32)

        # 1. Bounding box ratio (width / height)
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ratio = bw / max(bh, 1)
        features[0] = ratio

        # 2. Log angle — body vector angle from vertical
        body_vec = curr_kp['N'] - curr_kp['B']
        angle_vert = math.atan2(-body_vec[0], -body_vec[1])
        features[1] = np.log(1 + abs(angle_vert))

        # 3. Rotational energy (requires previous frame)
        if prev_kp is not None and prev_kp.get('N') is not None and prev_kp.get('B') is not None:
            # RE from reference: weighted angular velocity of head and torso segments
            N1 = curr_kp['N'] - curr_kp['B']
            N0 = prev_kp['N'] - prev_kp['B']
            d2sq = N1.dot(N1)
            angle_N = math.atan2(np.linalg.det([N0, N1]), np.dot(N0, N1))
            w2sq = (angle_N / dt) ** 2
            energy = 5 * d2sq * w2sq  # mass=5 for torso

            if curr_kp['H'] is not None and prev_kp.get('H') is not None:
                H1 = curr_kp['H'] - curr_kp['B']
                H0 = prev_kp['H'] - prev_kp['B']
                d1sq = H1.dot(H1)
                angle_H = math.atan2(np.linalg.det([H0, H1]), np.dot(H0, H1))
                w1sq = (angle_H / dt) ** 2
                energy += 1 * d1sq * w1sq  # mass=1 for head
                den = 5 * d2sq + 1 * d1sq
            else:
                den = 5 * d2sq

            features[2] = energy / (2 * den + 1e-6)

            # 4. Ratio derivative
            prev_bw = prev_kp.get('bbox_w', bw)
            prev_bh = prev_kp.get('bbox_h', bh)
            prev_ratio = prev_bw / max(prev_bh, 1)
            features[3] = (ratio - prev_ratio) / dt
        else:
            features[2] = 0.0
            features[3] = 0.0

        # 5. Generalised force (requires 2 previous frames)
        if (prev_kp is not None and prev2_kp is not None and
                all(prev_kp.get(k) is not None for k in ['H','N','B']) and
                all(prev2_kp.get(k) is not None for k in ['H','N','B'])):
            features[4] = self._compute_gf(prev2_kp, prev_kp, curr_kp, dt)
        else:
            features[4] = 0.0

        return features

    def _compute_gf(self, ip0, ip1, ip2, dt):
        """Generalised force from inverted double pendulum model (from reference)."""
        m1, m2, g = 1, 15, 10
        H2 = ip2['H'] - ip2['N']; H1 = ip1['H'] - ip1['N']; H0 = ip0['H'] - ip0['N']
        N2 = ip2['N'] - ip2['B']; N1 = ip1['N'] - ip1['B']; N0 = ip0['N'] - ip0['B']
        d1 = np.sqrt(H1.dot(H1)); d2 = np.sqrt(N1.dot(N1))
        if d2 < 1e-6: return 0.0
        d1_n = d1 / d2; d2_n = 1.0

        def av(v): return math.atan2(-v[0], -v[1])
        def ga(v0, v1): return math.atan2(np.linalg.det([v0,v1]), np.dot(v0,v1))

        t2_1, t2_0 = av(N1), av(N0)
        t12_2, t12_1, t12_0 = av(H2), av(H1), av(H0)
        t1_1 = t12_1 - t2_1; theta1, theta2 = t1_1, t2_1

        dt1_0, dt1_1 = ga(H0,H1)/dt, ga(H1,H2)/dt
        dt2_0, dt2_1 = ga(N0,N1)/dt, ga(N1,N2)/dt
        dt1 = 0.5*(dt1_0+dt1_1); dt2 = 0.5*(dt2_0+dt2_1)
        ddt1 = (dt1_1-dt1_0)/(0.5*2*dt); ddt2 = (dt2_1-dt2_0)/(0.5*2*dt)

        Q1 = m1*d1_n*ddt1**2 + (m1*d1_n**2+m1*d1_n*d2_n*np.cos(theta1))*ddt2
        Q1 += m1*d1_n*d2_n*np.sin(theta1)*dt2**2 - m1*g*d2_n*np.sin(theta1+theta2)
        Q2 = (m1*d1_n**2+m1*d1_n*d2_n*np.cos(theta1))*ddt1
        Q2 += ((m1+m2)*d2_n**2+m1*d1_n**2+2*m1*d1_n*d2_n*np.cos(theta1))*ddt2
        Q2 -= 2*m1*d1_n*d2_n*np.sin(theta1)*dt2*dt1+m1*d1_n*d2_n*np.sin(theta1)*dt1**2
        Q2 -= (m1+m2)*g*d2_n*np.sin(theta2)+m1*g*d1_n*np.sin(theta1+theta2)
        return Q1 + Q2

    def extract_for_track(self, frame, track) -> Optional[np.ndarray]:
        """Extract pose and compute 5 physics features for one tracked person.
        Returns: np.array of shape (5,) or None.
        """
        h, w = frame.shape[:2]
        bbox = track.bbox
        bw, bh = bbox[2]-bbox[0], bbox[3]-bbox[1]
        pad = 0.1
        x1, y1 = max(0, int(bbox[0]-bw*pad)), max(0, int(bbox[1]-bh*pad))
        x2, y2 = min(w, int(bbox[2]+bw*pad)), min(h, int(bbox[3]+bh*pad))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return None

        results = self.pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks is None: return None

        # Extract inverted pendulum keypoints in full-frame coordinates
        landmarks = results.pose_landmarks.landmark
        # Scale MediaPipe normalised coords to crop, then offset to full frame
        ch, cw = crop.shape[:2]
        kp = self._extract_inverted_pendulum(landmarks, cw, ch)
        if kp is None: return None
        # Offset to full frame
        for key in ['H', 'N', 'B']:
            if kp[key] is not None:
                kp[key] = kp[key] + np.array([x1, y1])
        kp['bbox_w'] = bw
        kp['bbox_h'] = bh

        # Get previous keypoints for this track
        tid = track.track_id
        if tid not in self._state:
            self._state[tid] = {"kp_history": deque(maxlen=3), "feat_buffer": deque(maxlen=self.sequence_length)}
        state = self._state[tid]

        prev_kp = state["kp_history"][-1] if len(state["kp_history"]) >= 1 else None
        prev2_kp = state["kp_history"][-2] if len(state["kp_history"]) >= 2 else None
        state["kp_history"].append(kp)

        # Compute 5 physics features
        features = self._compute_physics_features(kp, prev_kp, prev2_kp, bbox)
        state["feat_buffer"].append(features)

        # Also store raw 33 keypoints for skeleton drawing
        raw_kp = np.zeros(99, dtype=np.float32)
        for i, lm in enumerate(landmarks):
            raw_kp[i*3], raw_kp[i*3+1], raw_kp[i*3+2] = lm.x, lm.y, lm.visibility
        state["raw_kp"] = raw_kp

        return features

    def get_feature_sequence(self, track_id) -> Optional[np.ndarray]:
        """Get full temporal feature sequence (seq_len, 5) for LSTM."""
        if track_id not in self._state: return None
        buf = self._state[track_id]["feat_buffer"]
        if len(buf) < self.sequence_length: return None
        return np.array(list(buf), dtype=np.float32)

    def get_raw_keypoints(self, track_id) -> Optional[np.ndarray]:
        """Get latest raw 99-dim keypoints for skeleton drawing."""
        if track_id not in self._state: return None
        return self._state[track_id].get("raw_kp")

    def get_buffer_length(self, track_id):
        if track_id not in self._state: return 0
        return len(self._state[track_id]["feat_buffer"])

    def cleanup(self, active_ids: set):
        """Remove state for tracks no longer active."""
        stale = set(self._state) - active_ids
        for sid in stale: del self._state[sid]

    def release(self):
        if hasattr(self, 'pose'): self.pose.close()
        self._state.clear()
