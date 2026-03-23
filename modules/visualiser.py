"""Visualisation: colour-coded by fall status, skeleton overlay, info panel."""
import cv2, numpy as np, time
from typing import Dict, Optional
from collections import defaultdict

SKELETON = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]
COLORS = {"FALL": (0,0,255), "FALL WARNING": (0,165,255), "normal": (0,200,0), "unknown": (128,128,128)}

class Visualiser:
    def __init__(self, bbox_thickness=2, font_scale=0.6):
        self.thickness = bbox_thickness; self.font_scale = font_scale
        self._ftimes = []; self._fps = 0.0; self.total_frames = 0

    def annotate_frame(self, frame, tracks, pose_extractor=None, model_info=None):
        out = frame.copy(); self.total_frames += 1
        now = time.time(); self._ftimes = [t for t in self._ftimes if now-t<1.0]; self._ftimes.append(now); self._fps = len(self._ftimes)

        for track in tracks:
            if not track.is_active: continue
            color = COLORS.get(track.activity, COLORS["unknown"])
            x1,y1,x2,y2 = track.bbox.astype(int)

            # Thicker box for falls
            thick = self.thickness + 2 if "FALL" in track.activity else self.thickness
            cv2.rectangle(out, (x1,y1), (x2,y2), color, thick)

            # Label
            label = f"ID:{track.track_id} | {track.activity}"
            if track.activity_confidence > 0:
                label += f" ({track.activity_confidence:.0%})"
            (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
            cv2.rectangle(out, (x1,y1-th-10), (x1+tw+4,y1), color, -1)
            cv2.putText(out, label, (x1+2,y1-5), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255,255,255), 1, cv2.LINE_AA)

            # FALL alert banner
            if track.activity == "FALL":
                h_frame = out.shape[0]
                cv2.putText(out, f"FALL DETECTED - Person {track.track_id}",
                           (x1, max(y2+25, 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)

            # Skeleton
            if pose_extractor:
                raw_kp = pose_extractor.get_raw_keypoints(track.track_id)
                if raw_kp is not None:
                    bw, bh = x2-x1, y2-y1
                    pts = [(int(x1+raw_kp[i*3]*bw), int(y1+raw_kp[i*3+1]*bh), raw_kp[i*3+2]) for i in range(33)]
                    for s,e in SKELETON:
                        if pts[s][2]>0.5 and pts[e][2]>0.5:
                            cv2.line(out, (pts[s][0],pts[s][1]), (pts[e][0],pts[e][1]), color, 1, cv2.LINE_AA)

        # Info panel
        overlay = out.copy(); cv2.rectangle(overlay,(10,10),(300,130),(0,0,0),-1)
        cv2.addWeighted(overlay,0.6,out,0.4,0,out)
        active = sum(1 for t in tracks if t.is_active)
        falls = sum(1 for t in tracks if t.is_active and "FALL" in t.activity)
        y = 30
        for line in [f"FPS: {self._fps:.1f}", f"Persons: {active}", f"Falls: {falls}", f"Frame: {self.total_frames}"]:
            cv2.putText(out, line, (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA); y+=22
        return out
