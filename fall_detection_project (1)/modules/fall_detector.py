"""Fall detection classifier: LSTM on 5 physics features + false-positive dampening.

Architecture follows the reference project's LSTMModel but adapted for your pipeline:
  - Input: sequence of 5 physics features per frame
  - Output: 2 classes (fall / no-fall)
  - False-positive dampening: requires consecutive fall predictions + angle check
"""
import numpy as np
import math
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class FallLSTM(nn.Module):
    """LSTM classifier matching the reference project's architecture.
    Input: (batch, seq_len, 5) -> (batch, 2)
    """
    def __init__(self, input_dim=5, hidden_size=48, num_layers=2, dropout=0.1, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, h_s=None):
        out, h_s = self.lstm(x, h_s)
        logits = self.fc(out[:, -1, :])  # Last time step
        return logits, h_s


class FallDetector:
    """Fall detection with false-positive dampening logic from reference project.

    The dampening works as follows (adapted from reference):
    1. LSTM predicts fall/no-fall each frame
    2. A 'fall_count' counter tracks consecutive fall predictions
    3. A fall is only confirmed if fall_count exceeds a threshold
    4. Additional checks: torso angle must be > 45 degrees from vertical
    5. Height EMA tracks normal standing height to catch false falls
    """

    def __init__(self, hidden_size=48, num_layers=2, dropout=0.1,
                 fall_consec_frames=9, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = FallLSTM(input_dim=5, hidden_size=hidden_size,
                              num_layers=num_layers, dropout=dropout, num_classes=2).to(self.device)
        self.model.eval()
        self._loaded = False
        self.fall_consec_thresh = fall_consec_frames

        # Per-track LSTM hidden states
        self._hidden_states: Dict[int, tuple] = {}

        # EMA parameters (from reference)
        self.ema_frames = fall_consec_frames * 3
        self.ema_beta = 1 / (self.ema_frames + 1)

        print(f"[FallDetector] LSTM(5→{hidden_size}→2) | consec={fall_consec_frames} | device={self.device}")

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.model.eval()
        self._loaded = True
        print(f"[FallDetector] Loaded: {path}")

    @torch.no_grad()
    def predict(self, track, features: np.ndarray, feature_sequence: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """Predict fall/no-fall for a tracked person with dampening.

        Args:
            track: Track object (has fall_count, height_ema, etc.)
            features: Current frame's 5 physics features.
            feature_sequence: Full temporal sequence (seq_len, 5) if available.

        Returns:
            (label, confidence) — "FALL", "FALL WARNING", or "normal"
        """
        if not self._loaded:
            return self._rule_based_fallback(track, features)

        # Use sequence if available, otherwise single-frame with hidden state
        if feature_sequence is not None:
            x = torch.FloatTensor(feature_sequence).unsqueeze(0).to(self.device)
            logits, h_s = self.model(x)
            self._hidden_states[track.track_id] = h_s
        else:
            x = torch.FloatTensor(features).view(1, 1, 5).to(self.device)
            h_s = self._hidden_states.get(track.track_id)
            logits, h_s = self.model(x, h_s)
            self._hidden_states[track.track_id] = h_s

        probs = F.softmax(logits, dim=1)
        fall_prob = probs[0, 0].item()   # class 0 = fall
        normal_prob = probs[0, 1].item()
        raw_pred = 0 if fall_prob > normal_prob else 1

        # --- False-positive dampening (from reference) ---
        return self._dampen_prediction(track, features, raw_pred, fall_prob)

    def _dampen_prediction(self, track, features, raw_pred, confidence) -> Tuple[str, float]:
        """Apply false-positive dampening logic from the reference project."""
        ratio_bbox = features[0]
        log_angle = features[1]

        # Recover approximate angle from log_angle: log(1+|angle|)
        angle_abs = math.exp(log_angle) - 1

        if raw_pred == 0:  # LSTM says fall
            # Check 1: angle must be > pi/4 (45 degrees from vertical)
            if angle_abs < math.pi / 4:
                raw_pred = 1  # Override: not enough tilt

            # Check 2: confidence must be > 0.4
            elif confidence < 0.4:
                raw_pred = 1

            # Check 3: bbox height shouldn't be too close to normal standing height
            elif track.height_ema > 0 and track.bbox_height > 2 * track.height_ema / 3:
                if angle_abs < math.pi / 4:
                    raw_pred = 1

        if raw_pred == 0:  # Still classified as fall after checks
            track.fall_count += 1
            if track.fall_count < self.fall_consec_thresh:
                return "FALL WARNING", confidence
            else:
                return "FALL", confidence
        else:
            # Update EMA of standing height
            track.fall_count = max(0, track.fall_count - 1)
            if track.height_ema_frames < self.ema_frames:
                track.height_ema_frames += 1
                track.height_ema = (track.height_ema * (track.height_ema_frames - 1) + track.bbox_height) / track.height_ema_frames
            else:
                track.height_ema = (1 - self.ema_beta) * track.bbox_height + self.ema_beta * track.height_ema
            return "normal", 1.0 - confidence

    def _rule_based_fallback(self, track, features) -> Tuple[str, float]:
        """Simple rule-based fall detection when no LSTM model is loaded.
        Uses the physics features directly.
        """
        ratio_bbox = features[0]    # w/h ratio — high means lying
        log_angle = features[1]     # body tilt — high means fallen
        re = features[2]            # rotational energy — high during fall
        ratio_deriv = features[3]   # rate of ratio change — spike during fall

        angle_abs = math.exp(log_angle) - 1
        is_fall = False

        # Rule 1: large body angle (> 50 degrees from vertical)
        if angle_abs > math.pi / 3.5:
            is_fall = True

        # Rule 2: bbox is wider than tall (lying down)
        if ratio_bbox > 1.2:
            is_fall = True

        # Rule 3: sudden high rotational energy + ratio change
        if re > 0.5 and abs(ratio_deriv) > 2.0:
            is_fall = True

        if is_fall:
            track.fall_count += 1
            track.fall_count = max(0, track.fall_count - 1)
            if track.fall_count < self.fall_consec_thresh:
                return "FALL WARNING", 0.6
            return "FALL", 0.8
        else:
            track.fall_count = max(0, track.fall_count - 1)
            # Update height EMA
            if track.height_ema_frames < self.ema_frames:
                track.height_ema_frames += 1
                track.height_ema = (track.height_ema * (track.height_ema_frames - 1) + track.bbox_height) / track.height_ema_frames
            else:
                track.height_ema = (1 - self.ema_beta) * track.bbox_height + self.ema_beta * track.height_ema
            return "normal", 0.9

    def cleanup(self, active_ids: set):
        stale = set(self._hidden_states) - active_ids
        for sid in stale: del self._hidden_states[sid]

    def release(self):
        self._hidden_states.clear()
