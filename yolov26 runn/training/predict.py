"""
Run predictions on the test images using the trained YOLO26 model.
Results (annotated images) are saved to model/training/predictions/

Usage:
    python model/training/predict.py
"""

from pathlib import Path
import cv2
from ultralytics import YOLO  # type: ignore

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
WEIGHTS     = ROOT / "model" / "training" / "runs" / "yolo26" / "weights" / "best.pt"
TEST_IMAGES = ROOT / "dataset" / "test" / "images"
OUT_DIR     = ROOT / "model" / "training" / "predictions"

CONF = 0.01   # effectively 0 — show all detections (ultralytics min is 0.001)

# ── Load model ─────────────────────────────────────────────────────────────────
print(f"Loading weights: {WEIGHTS}\n")
model = YOLO(str(WEIGHTS))

OUT_DIR.mkdir(parents=True, exist_ok=True)
images = sorted(TEST_IMAGES.glob("*.jpg")) + sorted(TEST_IMAGES.glob("*.png"))

if not images:
    print("No test images found.")
    raise SystemExit(1)

# ── Predict ────────────────────────────────────────────────────────────────────
print(f"Running predictions on {len(images)} image(s)  (conf>={CONF})\n")
print("Press any key in the image window to advance to the next image.\n")

for img_path in images:
    results  = model.predict(str(img_path), conf=CONF, verbose=False)
    result   = results[0]

    # Print detections for this image
    print(f"── {img_path.name}")
    if len(result.boxes) == 0:
        print("   No detections above confidence threshold.")
    else:
        for box in result.boxes:
            cls_id   = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf     = float(box.conf[0])
            xyxy     = box.xyxy[0].tolist()
            print(f"   [{cls_name}] conf={conf:.2f}  box={[round(v) for v in xyxy]}")
    print()

    # Save annotated image
    annotated = result.plot()
    out_path  = OUT_DIR / img_path.name
    cv2.imwrite(str(out_path), annotated)

    # Show annotated image in a window (press any key to continue)
    cv2.imshow(img_path.name, annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print(f"Annotated images saved to: {OUT_DIR}")
