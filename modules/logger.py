"""Structured logging to JSON and CSV."""
import json, csv, os
from datetime import datetime
from collections import defaultdict

class DataLogger:
    def __init__(self, output_dir="outputs/", session_name=None):
        os.makedirs(output_dir, exist_ok=True)
        self.session = session_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = os.path.join(output_dir, f"log_{self.session}.csv")
        self._json_path = os.path.join(output_dir, f"log_{self.session}.json")
        self.frame_logs = []; self.stats = {"frames":0,"detections":0,"falls":0,"unique":set(),"max_sim":0}
        self._csv_file = open(self._csv_path,"w",newline=""); self._w = csv.writer(self._csv_file)
        self._w.writerow(["frame","timestamp","track_id","x1","y1","x2","y2","activity","confidence"])

    def log_frame(self, frame_num, tracks, fps=0, timestamp=None):
        if timestamp is None: timestamp = frame_num/30.0
        active = 0
        for t in tracks:
            if not t.is_active: continue
            active += 1; self.stats["unique"].add(t.track_id)
            if "FALL" in t.activity: self.stats["falls"] += 1
            self._w.writerow([frame_num, f"{timestamp:.3f}", t.track_id,
                *[f"{float(b):.1f}" for b in t.bbox], t.activity, f"{t.activity_confidence:.3f}"])
        self.stats["frames"]+=1; self.stats["detections"]+=active
        self.stats["max_sim"]=max(self.stats["max_sim"],active)

    def save(self):
        self._csv_file.flush()
        summary = {"session":self.session,"total_frames":self.stats["frames"],
                   "total_detections":self.stats["detections"],"fall_events":self.stats["falls"],
                   "unique_individuals":len(self.stats["unique"]),"max_simultaneous":self.stats["max_sim"]}
        with open(self._json_path,"w") as f: json.dump(summary,f,indent=2)
        print(f"[Logger] CSV: {self._csv_path} | JSON: {self._json_path}")

    def close(self): self.save(); self._csv_file.close()
    def get_summary(self):
        return {"frames":self.stats["frames"],"detections":self.stats["detections"],
                "falls":self.stats["falls"],"unique":len(self.stats["unique"]),"max_sim":self.stats["max_sim"]}
