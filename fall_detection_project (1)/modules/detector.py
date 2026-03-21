"""YOLOv8 person detection — filters to person class only."""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
from utils.config_loader import YOLO_MODEL_MAP

@dataclass
class Detection:
    bbox: np.ndarray   # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    @property
    def center(self): return ((self.bbox[0]+self.bbox[2])/2, (self.bbox[1]+self.bbox[3])/2)

class PersonDetector:
    def __init__(self, model_size="medium", confidence_threshold=0.5, iou_threshold=0.45,
                 person_class_id=0, input_size=640, device=None):
        if YOLO is None: raise ImportError("pip install ultralytics")
        weights = YOLO_MODEL_MAP.get(model_size, "yolov8m.pt")
        self.model = YOLO(weights)
        self.conf = confidence_threshold
        self.iou = iou_threshold
        self.cls = person_class_id
        self.imgsz = input_size
        self.device = device
        if device: self.model.to(device)
        print(f"[Detector] YOLOv8-{model_size} loaded | conf={confidence_threshold}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, conf=self.conf, iou=self.iou, classes=[self.cls],
                           imgsz=self.imgsz, verbose=False)
        dets = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0: continue
            for b, c, cl in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                dets.append(Detection(b.astype(np.float32), float(c), int(cl)))
        return dets

    def get_model_info(self):
        return {"model": "YOLOv8", "conf": self.conf, "device": str(self.device or "auto")}
