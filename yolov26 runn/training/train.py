"""
Train a YOLO model on the paintings dataset.
Classes: libre, monalisa, objects, skrik, starrynight, sunflower

Usage:
    python model/training/train.py
"""

import os
from pathlib import Path
from ultralytics import YOLO  # type: ignore

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]          # project root
DATA_YAML = ROOT / "dataset" / "data.yaml"
OUTPUT_DIR = ROOT / "model" / "training" / "runs"

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_MODEL = "yolo26n.pt"   # pretrained weights to fine-tune from
                             # swap for yolo26s.pt / yolo26m.pt for more capacity

EPOCHS      = 100
IMGSZ       = 640
BATCH       = 16            # lower to 8 if you hit OOM
LR0         = 0.01          # initial learning rate
WORKERS     = 4
PATIENCE    = 20            # early-stop patience (epochs without improvement)
RUN_NAME    = "yolo26"      # name shown in results folder

# ── Train ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Dataset : {DATA_YAML}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Output   : {OUTPUT_DIR / RUN_NAME}")
    print()

    model = YOLO(BASE_MODEL)

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        lr0=LR0,
        workers=WORKERS,
        patience=PATIENCE,
        project=str(OUTPUT_DIR),
        name=RUN_NAME,
        exist_ok=True,          # resume / overwrite same run folder
        plots=True,             # save training plots
        save=True,              # save best.pt and last.pt
    )

    # ── Evaluate on test split ─────────────────────────────────────────────────
    print("\nRunning evaluation on test split...")
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        project=str(OUTPUT_DIR),
        name=f"{RUN_NAME}_test_eval",
        exist_ok=True,
    )

    print("\n── Test results ──────────────────────────────────────────────────────")
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")

    best_weights = OUTPUT_DIR / RUN_NAME / "weights" / "best.pt"
    print(f"\nBest weights saved to: {best_weights}")


if __name__ == "__main__":
    main()
