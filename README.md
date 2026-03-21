# Multi-Human Fall Detection in Shared Spaces

**CM3070 Final Year Project** — University of London  
**Author:** Chian Chin Yang (230369401) | **Supervisor:** Gerald Deng Xiang Chua

## Pipeline

```
Video Frame
  → YOLOv8        (detect persons)
  → ByteTrack     (persistent identity tracking)
  → MediaPipe     (33 skeletal keypoints)
  → Physics Engine (5 biomechanical features per person)
  → Fall LSTM     (fall / no-fall classification)
  → Dampening     (consecutive-frame + angle + height checks)
  → Annotated Output + Logging
```

## 5 Physics Features (from inverted pendulum model)

| Feature | Description | Fall indicator |
|---------|-------------|----------------|
| `ratio_bbox` | Width/height of bounding box | >1.0 = lying down |
| `log_angle` | Log of torso angle from vertical | High = tilted/fallen |
| `re` | Rotational energy of head+torso | Spikes during fall |
| `ratio_deriv` | Rate of change of bbox ratio | Sudden change = falling |
| `gf` | Generalised force (double pendulum) | High = loss of balance |

## Structure

```
combined/
├── main.py                    # End-to-end pipeline
├── train.py                   # Train fall LSTM
├── requirements.txt
├── configs/config.yaml
├── modules/
│   ├── detector.py            # YOLOv8 person detection
│   ├── tracker.py             # ByteTrack multi-object tracking
│   ├── pose_extractor.py      # MediaPipe + physics features
│   ├── fall_detector.py       # LSTM classifier + dampening
│   ├── visualiser.py          # Annotated display
│   └── logger.py              # JSON/CSV logging
├── utils/config_loader.py
├── models/                    # Saved weights
├── tests/test_system.py
└── outputs/
```

## Quick Start

```bash
pip install -r requirements.txt

# Option 1: Run with rule-based fall detection (no training needed)
python main.py --input video.mp4 --show

# Option 2: Train LSTM first, then run
python train.py --synthetic --epochs 50
python main.py --input video.mp4 --model_weights models/fall_lstm.pth --show

# Save output video
python main.py --input video.mp4 --output outputs/result.mp4

# Use lighter model for edge devices
python main.py --input video.mp4 --model_size nano --frame_skip 2 --show
```

## How Fall Detection Works

1. **Detection**: YOLOv8 detects all persons; ByteTrack assigns persistent IDs
2. **Pose**: MediaPipe extracts 33 keypoints → inverted pendulum model (H=head, N=neck, B=base)
3. **Features**: 5 physics features computed per person per frame
4. **LSTM**: Processes temporal sequence of features → fall probability
5. **Dampening** (from reference paper):
   - Fall must be predicted for 9+ consecutive frames
   - Torso angle must exceed 45° from vertical
   - Bounding box height must be significantly reduced from standing EMA
   - Confidence must exceed 40%
   - This prevents false alarms from bending, picking up objects, etc.

## References

- Taufeeque et al., "Multi-camera, multi-person, and real-time fall detection using long short term memory," SPIE Medical Imaging 2021
- Jocher et al., "Ultralytics YOLOv8," 2023
- Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," ECCV 2022
- Bazarevsky et al., "BlazePose: On-Device Real-Time Body Pose Tracking," 2020# Human_fall_detection
