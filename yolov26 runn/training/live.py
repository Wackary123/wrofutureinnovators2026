"""
Real-time YOLO26 inference from webcam using GPU.

Usage:
    python model/training/live.py

Controls:
    Q  — quit
"""

from pathlib import Path
import cv2
import torch
from ultralytics import YOLO  # type: ignore

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "model" / "training" / "runs" / "yolo26" / "weights" / "best.pt"

CAM_INDEX = 0       # change if your webcam isn't device 0
IMGSZ     = 640
CONF      = 0.01    # near-zero to see everything the model outputs
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load model ─────────────────────────────────────────────────────────────────
print(f"Device  : {DEVICE}")
print(f"Weights : {WEIGHTS}\n")

model = YOLO(str(WEIGHTS))
model.to(DEVICE)

# ── Open camera ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera {CAM_INDEX}")

print("Camera opened. Press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run inference on GPU
    results   = model.predict(frame, imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)
    result    = results[0]
    annotated = result.plot()

    # FPS overlay
    inf_ms = result.speed["inference"]
    fps    = 1000 / inf_ms if inf_ms > 0 else 0
    n_det  = len(result.boxes)
    cv2.putText(annotated, f"FPS: {fps:.1f}  [{DEVICE.upper()}]  detections: {n_det}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # If detections exist, print them to terminal
    for box in result.boxes:
        cls_name = model.names[int(box.cls[0])]
        conf_val = float(box.conf[0])
        print(f"  {cls_name}  {conf_val:.3f}", end="  ")
    if n_det:
        print()

    cv2.imshow("YOLO26 Live", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
