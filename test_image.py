"""Quick test: run detection on a single image."""
import cv2
import sys
from modules.detector import PersonDetector

# Load image
img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
img = cv2.imread(img_path)
if img is None:
    print(f"Cannot read: {img_path}")
    sys.exit(1)

# Detect
detector = PersonDetector(model_size="medium", confidence_threshold=0.5)
detections = detector.detect(img)

# Draw results
for det in detections:
    x1, y1, x2, y2 = det.bbox.astype(int)
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(img, f"{det.confidence:.0%}", (x1, y1-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

print(f"Detected {len(detections)} persons")

# Save as JPG (not MP4)
out_path = img_path.rsplit(".", 1)[0] + "_result.jpg"
cv2.imwrite(out_path, img)
print(f"Saved: {out_path}")