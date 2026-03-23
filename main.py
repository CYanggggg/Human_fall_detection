import argparse, os, time, cv2, numpy as np
from pathlib import Path
from utils.config_loader import load_config
from modules.detector import PersonDetector
from modules.tracker import ByteTracker
from modules.pose_extractor import PoseExtractor
from modules.fall_detector import FallDetector
from modules.visualiser import Visualiser
from modules.logger import DataLogger


class FallDetectionPipeline:
    """End-to-end multi-human fall detection."""

    def __init__(self, config_path=None, device=None):
        cfg = load_config(config_path)
        print("="*60)
        print("Multi-Human Fall Detection System")
        print("YOLOv8 + ByteTrack + MediaPipe + Physics LSTM")
        print("CM3070 Project — Chian Chin Yang")
        print("="*60)

        d, t, p, c = cfg.detection, cfg.tracking, cfg.pose, cfg.classification
        self.detector = PersonDetector(d.model_size, d.confidence_threshold, d.iou_threshold,
                                       d.person_class_id, d.input_size, device)
        self.tracker = ByteTracker(t.track_high_thresh, t.track_low_thresh, t.new_track_thresh,
                                    t.track_buffer, t.match_thresh)
        self.pose_extractor = PoseExtractor(p.model_complexity, p.min_detection_confidence,
                                             p.min_tracking_confidence, c.sequence_length)
        self.fall_detector = FallDetector(c.lstm_hidden, c.lstm_layers, c.dropout,
                                          c.fall_consec_frames, device)
        v = cfg.visualisation
        self.visualiser = Visualiser(v.bbox_thickness, v.font_scale)
        self.logger = None
        print("\n[Pipeline] All modules ready.\n")

    def load_model(self, path):
        self.fall_detector.load_model(path)

    def process_video(self, input_path, output_path=None, show=False, max_frames=-1, frame_skip=1):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened(): raise FileNotFoundError(f"Cannot open: {input_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Pipeline] {input_path} | {w}x{h} @ {fps:.1f}fps | {total} frames")

        writer = None
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        self.logger = DataLogger(os.path.dirname(output_path or "outputs/"), Path(input_path).stem)

        fc = processed = 0; t0 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret: break
            fc += 1
            if 0 < max_frames < fc: break
            if fc % frame_skip != 0: continue
            processed += 1

            # Stage 1: Detect + Track
            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections)
            active_ids = {t.track_id for t in tracks if t.is_active}

            # Stage 2: Pose + Physics Features → Fall Classification
            for track in tracks:
                if not track.is_active: continue
                features = self.pose_extractor.extract_for_track(frame, track)
                if features is None:
                    track.activity, track.activity_confidence = "normal", 0.0
                    continue

                seq = self.pose_extractor.get_feature_sequence(track.track_id)
                label, conf = self.fall_detector.predict(track, features, seq)
                track.activity = label
                track.activity_confidence = conf

            # Cleanup stale states
            self.pose_extractor.cleanup(active_ids)
            self.fall_detector.cleanup(active_ids)

            # Stage 3: Visualise + Log
            annotated = self.visualiser.annotate_frame(frame, tracks, self.pose_extractor)
            self.logger.log_frame(fc, tracks, self.visualiser._fps, fc/fps)

            if writer: writer.write(annotated)
            if show:
                cv2.imshow("Fall Detection", annotated)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"): break
                elif k == ord("p"): cv2.waitKey(0)

            if processed % 100 == 0:
                el = time.time() - t0
                print(f"[Pipeline] {fc}/{total} ({fc/max(total,1)*100:.1f}%) | {processed/max(el,.001):.1f} FPS")

        elapsed = time.time() - t0
        cap.release()
        if writer: writer.release()
        if show: cv2.destroyAllWindows()

        summary = self.logger.get_summary()
        self.logger.close()
        print(f"\n{'='*60}")
        print(f"[Pipeline] Done | {processed} frames | {elapsed:.1f}s | {processed/max(elapsed,.001):.1f} FPS")
        print(f"  Persons: {summary['unique']} unique | {summary['max_sim']} max simultaneous")
        print(f"  Fall events: {summary['falls']}")
        print("="*60)
        return summary

    def process_frame(self, frame):
        """Single frame processing for live camera."""
        dets = self.detector.detect(frame)
        tracks = self.tracker.update(dets)
        active_ids = {t.track_id for t in tracks if t.is_active}
        for track in tracks:
            if not track.is_active: continue
            features = self.pose_extractor.extract_for_track(frame, track)
            if features is None: track.activity = "normal"; continue
            seq = self.pose_extractor.get_feature_sequence(track.track_id)
            track.activity, track.activity_confidence = self.fall_detector.predict(track, features, seq)
        self.pose_extractor.cleanup(active_ids)
        self.fall_detector.cleanup(active_ids)
        return self.visualiser.annotate_frame(frame, tracks, self.pose_extractor), tracks

    def release(self):
        self.pose_extractor.release()
        self.fall_detector.release()
        if self.logger: self.logger.close()


def main():
    p = argparse.ArgumentParser(description="Multi-Human Fall Detection")
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--config", "-c", default=None)
    p.add_argument("--model_weights", default=None, help="Trained LSTM weights")
    p.add_argument("--model_size", default=None, choices=["nano","small","medium","large","xlarge"])
    p.add_argument("--show", action="store_true")
    p.add_argument("--max_frames", type=int, default=-1)
    p.add_argument("--frame_skip", type=int, default=1)
    p.add_argument("--device", default=None, choices=["cpu","cuda","mps"])
    a = p.parse_args()
    if a.output is None: a.output = f"outputs/{Path(a.input).stem}_fall_detect.mp4"

    pipe = FallDetectionPipeline(a.config, a.device)
    if a.model_weights: pipe.load_model(a.model_weights)

    try: pipe.process_video(a.input, a.output, a.show, a.max_frames, a.frame_skip)
    except KeyboardInterrupt: print("\nInterrupted.")
    finally: pipe.release()

if __name__ == "__main__": main()
