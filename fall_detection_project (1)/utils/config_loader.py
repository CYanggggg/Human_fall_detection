import yaml, os
from pathlib import Path

class Config:
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else
                    [Config(i) if isinstance(i, dict) else i for i in v] if isinstance(v, list) else v)
    def to_dict(self):
        r = {}
        for k, v in self.__dict__.items():
            r[k] = v.to_dict() if isinstance(v, Config) else \
                   [i.to_dict() if isinstance(i, Config) else i for i in v] if isinstance(v, list) else v
        return r

def load_config(path=None):
    if path is None:
        path = os.path.join(Path(__file__).parent.parent, "configs", "config.yaml")
    with open(path) as f:
        return Config(yaml.safe_load(f))

YOLO_MODEL_MAP = {"nano":"yolov8n.pt","small":"yolov8s.pt","medium":"yolov8m.pt","large":"yolov8l.pt","xlarge":"yolov8x.pt"}
