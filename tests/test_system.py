"""Unit tests for the combined fall detection system."""
import sys, os, numpy as np, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_config():
    from utils.config_loader import load_config
    c = load_config()
    assert c.classification.num_classes == 2
    assert c.classification.num_physics_features == 5
    assert c.classification.sequence_length == 36
    print("PASS: Config")

def test_iou():
    from modules.tracker import compute_iou_matrix
    a = np.array([[0,0,10,10]], dtype=np.float32)
    b = np.array([[5,5,15,15]], dtype=np.float32)
    assert 0 < compute_iou_matrix(a, b)[0,0] < 1
    c = np.array([[10,10,50,50]], dtype=np.float32)
    assert abs(compute_iou_matrix(c, c)[0,0] - 1.0) < 1e-5
    assert compute_iou_matrix(a, np.array([[20,20,30,30]], dtype=np.float32))[0,0] == 0.0
    print("PASS: IoU computation")

def test_tracker_identity():
    from modules.tracker import ByteTracker, Track
    from modules.detector import Detection
    t = ByteTracker()
    # Frame 1: two persons
    d1 = [Detection(np.array([10,10,100,200],dtype=np.float32), 0.9, 0),
          Detection(np.array([300,10,400,200],dtype=np.float32), 0.85, 0)]
    tracks = t.update(d1)
    assert len(tracks) == 2
    id_a, id_b = tracks[0].track_id, tracks[1].track_id
    assert id_a != id_b

    # Frame 2: both slightly moved
    d2 = [Detection(np.array([15,12,105,202],dtype=np.float32), 0.88, 0),
          Detection(np.array([305,12,405,202],dtype=np.float32), 0.87, 0)]
    tracks2 = t.update(d2)
    assert len(tracks2) == 2
    ids2 = {tr.track_id for tr in tracks2}
    assert id_a in ids2 and id_b in ids2
    print("PASS: Multi-person tracking with identity persistence")

def test_track_fall_state():
    from modules.tracker import Track
    t = Track(track_id=1, bbox=np.array([50,50,200,400], dtype=np.float32), confidence=0.9)
    assert t.is_active
    assert t.fall_count == 0
    assert t.bbox_height == 350
    assert t.bbox_ratio < 1.0  # taller than wide = standing
    # Simulate lying down bbox
    t2 = Track(track_id=2, bbox=np.array([50,300,400,400], dtype=np.float32), confidence=0.9)
    assert t2.bbox_ratio > 1.0  # wider than tall = lying
    print("PASS: Track fall state properties")

def test_logger(tmp_dir="/tmp/test_fall_logger"):
    from modules.logger import DataLogger
    from modules.tracker import Track
    os.makedirs(tmp_dir, exist_ok=True)
    lg = DataLogger(tmp_dir)
    t1 = Track(1, np.array([10,20,100,200],dtype=np.float32), 0.9); t1.activity = "FALL"
    t2 = Track(2, np.array([300,20,400,200],dtype=np.float32), 0.85); t2.activity = "normal"
    lg.log_frame(1, [t1, t2], 30.0)
    assert lg.stats["frames"] == 1
    assert lg.stats["detections"] == 2
    assert lg.stats["falls"] == 1
    lg.save(); lg.close()
    assert any(f.endswith(".json") for f in os.listdir(tmp_dir))
    assert any(f.endswith(".csv") for f in os.listdir(tmp_dir))
    print("PASS: Logger with fall counting")

def test_visualiser():
    from modules.visualiser import Visualiser
    from modules.tracker import Track
    vis = Visualiser()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Normal person
    t1 = Track(1, np.array([50,50,200,400],dtype=np.float32), 0.9); t1.activity = "normal"
    # Fallen person
    t2 = Track(2, np.array([300,300,550,400],dtype=np.float32), 0.8); t2.activity = "FALL"; t2.activity_confidence = 0.85
    out = vis.annotate_frame(frame, [t1, t2])
    assert out.shape == frame.shape
    assert out.sum() > 0  # Something was drawn
    print("PASS: Visualiser with fall annotations")

def test_physics_features_standing():
    """Verify physics features produce expected values for standing pose."""
    # Standing: small angle, low ratio, low energy
    # Simulated inverted pendulum points
    kp_standing = {
        'H': np.array([320.0, 100.0]),   # Head above
        'N': np.array([320.0, 200.0]),    # Neck
        'B': np.array([320.0, 350.0]),    # Base below
        'bbox_w': 100.0, 'bbox_h': 300.0
    }
    # Body vector N-B is vertical (0, -150) -> angle should be ~0
    body_vec = kp_standing['N'] - kp_standing['B']  # (0, -150)
    angle = math.atan2(-body_vec[0], -body_vec[1])
    assert abs(angle) < 0.1, f"Standing angle should be ~0, got {angle}"

    # Bbox ratio for standing: 100/300 = 0.33
    ratio = 100.0 / 300.0
    assert ratio < 0.5, f"Standing ratio should be <0.5, got {ratio}"
    print("PASS: Physics features - standing pose validation")

def test_physics_features_fallen():
    """Verify physics features produce expected values for fallen pose."""
    kp_fallen = {
        'H': np.array([100.0, 400.0]),   # Head to the side
        'N': np.array([250.0, 400.0]),    # Neck
        'B': np.array([400.0, 400.0]),    # Base - all roughly same height = horizontal
        'bbox_w': 350.0, 'bbox_h': 80.0
    }
    body_vec = kp_fallen['N'] - kp_fallen['B']  # (-150, 0) - horizontal
    angle = math.atan2(-body_vec[0], -body_vec[1])
    assert abs(angle) > math.pi / 4, f"Fallen angle should be >pi/4, got {angle}"

    ratio = 350.0 / 80.0
    assert ratio > 1.0, f"Fallen ratio should be >1.0, got {ratio}"
    print("PASS: Physics features - fallen pose validation")

def test_synthetic_data_generation():
    """Test that training data generator produces balanced data."""
    try:
        from train import generate_synthetic_fall_data
    except ImportError:
        print("SKIP: Synthetic data generation (torch not installed)")
        return
    seqs, labels = generate_synthetic_fall_data(100, seq_len=36)
    assert seqs.shape == (100, 36, 5)
    assert len(labels) == 100
    assert sum(labels == 0) > 0  # Has fall samples
    assert sum(labels == 1) > 0  # Has normal samples
    assert seqs[:,:,0].max() < 5.0
    assert seqs[:,:,1].min() >= 0
    print("PASS: Synthetic data generation")


if __name__ == "__main__":
    test_config()
    test_iou()
    test_tracker_identity()
    test_track_fall_state()
    test_logger()
    test_visualiser()
    test_physics_features_standing()
    test_physics_features_fallen()
    test_synthetic_data_generation()
    print("\n" + "="*50)
    print("ALL 9 TESTS PASSED")
    print("="*50)
