"""ByteTrack multi-object tracking with Kalman filtering."""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from collections import deque
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    confidence: float
    state: str = "active"
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    activity: str = "normal"
    activity_confidence: float = 0.0
    fall_count: int = 0          # consecutive fall predictions
    height_ema: float = 0.0      # EMA of bounding box height (from reference)
    height_ema_frames: int = 0
    trajectory: deque = field(default_factory=lambda: deque(maxlen=300))

    @property
    def center(self): return ((self.bbox[0]+self.bbox[2])/2, (self.bbox[1]+self.bbox[3])/2)
    @property
    def is_active(self): return self.state == "active"
    @property
    def bbox_height(self): return self.bbox[3] - self.bbox[1]
    @property
    def bbox_width(self): return self.bbox[2] - self.bbox[0]
    @property
    def bbox_ratio(self): return self.bbox_width / max(self.bbox_height, 1)

class KalmanBoxTracker:
    _count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7); self.kf.F[0,4]=self.kf.F[1,5]=self.kf.F[2,6]=1
        self.kf.H = np.zeros((4,7)); np.fill_diagonal(self.kf.H[:4,:4],1)
        self.kf.R[2:,2:]*=10.; self.kf.P[4:,4:]*=1000.; self.kf.P*=10.
        self.kf.Q[-1,-1]*=0.01; self.kf.Q[4:,4:]*=0.01
        w,h=bbox[2]-bbox[0],bbox[3]-bbox[1]
        self.kf.x[:4]=np.array([bbox[0]+w/2,bbox[1]+h/2,w*h,w/(h+1e-6)]).reshape(4,1)
        self.time_since_update=0; self.hits=0; self.hit_streak=0; self.age=0
        KalmanBoxTracker._count+=1; self.id=KalmanBoxTracker._count

    def update(self, bbox):
        self.time_since_update=0; self.hits+=1; self.hit_streak+=1
        w,h=bbox[2]-bbox[0],bbox[3]-bbox[1]
        self.kf.update(np.array([bbox[0]+w/2,bbox[1]+h/2,w*h,w/(h+1e-6)]).reshape(4,1))

    def predict(self):
        if(self.kf.x[6]+self.kf.x[2])<=0: self.kf.x[6]*=0.
        self.kf.predict(); self.age+=1
        if self.time_since_update>0: self.hit_streak=0
        self.time_since_update+=1
        z=self.kf.x[:4].flatten(); w=np.sqrt(z[2]*z[3]); h=z[2]/(w+1e-6)
        return np.array([z[0]-w/2,z[1]-h/2,z[0]+w/2,z[1]+h/2])

def compute_iou_matrix(a, b):
    if len(a)==0 or len(b)==0: return np.zeros((len(a),len(b)))
    x1=np.maximum(a[:,0:1],b[:,0:1].T); y1=np.maximum(a[:,1:2],b[:,1:2].T)
    x2=np.minimum(a[:,2:3],b[:,2:3].T); y2=np.minimum(a[:,3:4],b[:,3:4].T)
    inter=np.maximum(0,x2-x1)*np.maximum(0,y2-y1)
    aa=(a[:,2]-a[:,0])*(a[:,3]-a[:,1]); ab=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    return inter/(aa[:,None]+ab[None,:]-inter+1e-6)

class ByteTracker:
    def __init__(self, track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.6,
                 track_buffer=30, match_thresh=0.8):
        self.high=track_high_thresh; self.low=track_low_thresh; self.new=new_track_thresh
        self.buffer=track_buffer; self.match=match_thresh
        self.trackers=[]; self.tracks=[]; self.frame_count=0; self._nid=1

    def _associate(self, iou, thresh):
        if iou.size==0: return [],list(range(iou.shape[0])),list(range(iou.shape[1]))
        r,c=linear_sum_assignment(1-iou); matched=[]; ud=list(range(iou.shape[0])); ut=list(range(iou.shape[1]))
        for ri,ci in zip(r,c):
            if iou[ri,ci]>=thresh: matched.append((ri,ci)); ud.remove(ri) if ri in ud else None; ut.remove(ci) if ci in ut else None
        return matched,ud,ut

    def update(self, detections) -> List[Track]:
        self.frame_count+=1
        if not detections:
            for t in self.trackers: t.predict()
            self._cleanup(); return [t for t in self.tracks if t.is_active]
        db=np.array([d.bbox for d in detections]); dc=np.array([d.confidence for d in detections])
        hm=dc>=self.high; lm=(dc>=self.low)&~hm
        hd,hc,ld,lc=db[hm],dc[hm],db[lm],dc[lm]
        pred=np.array([t.predict() for t in self.trackers]) if self.trackers else np.empty((0,4))
        ut=list(range(len(self.trackers))); uh=list(range(len(hd))); m=[]
        if len(hd)>0 and len(pred)>0:
            m,uh,ut=self._associate(compute_iou_matrix(hd,pred),self.match)
        for di,ti in m:
            self.trackers[ti].update(hd[di]); self.tracks[ti].bbox=hd[di]
            self.tracks[ti].confidence=float(hc[di]); self.tracks[ti].time_since_update=0
            self.tracks[ti].hits+=1; self.tracks[ti].trajectory.append(self.tracks[ti].center)
        if len(ld)>0 and len(ut)>0:
            m2,_,su=self._associate(compute_iou_matrix(ld,pred[ut]),self.match)
            for di,ri in m2:
                ai=ut[ri]; self.trackers[ai].update(ld[di]); self.tracks[ai].bbox=ld[di]
                self.tracks[ai].confidence=float(lc[di]); self.tracks[ai].time_since_update=0; self.tracks[ai].hits+=1
            ut=[ut[i] for i in su]
        for ti in ut:
            self.tracks[ti].time_since_update+=1
            if self.tracks[ti].time_since_update>self.buffer: self.tracks[ti].state="removed"
        for di in uh:
            if hc[di]>=self.new:
                self.trackers.append(KalmanBoxTracker(hd[di]))
                t=Track(self._nid,hd[di],float(hc[di])); t.trajectory.append(t.center)
                self.tracks.append(t); self._nid+=1
        self._cleanup(); return [t for t in self.tracks if t.is_active]

    def _cleanup(self):
        alive=[i for i,t in enumerate(self.tracks) if t.state!="removed"]
        for i in alive: self.tracks[i].age+=1
        self.tracks=[self.tracks[i] for i in alive]; self.trackers=[self.trackers[i] for i in alive]

    def get_track_count(self): return sum(1 for t in self.tracks if t.is_active)
    def reset(self): self.trackers.clear(); self.tracks.clear(); self.frame_count=0; self._nid=1; KalmanBoxTracker._count=0
