# inference.py
# Run MambaWeed inference on images or video
# Supports both standard YOLO checkpoints and MambaTrainer checkpoints

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from models.mamba_block import FourDirectionalMamba

CLASS_NAMES = [
    "crop", "weed", "waterhemp", "morningglory", "purslane",
    "spottedspurge", "carpetweed", "ragweed", "eclipta",
    "pricklysida", "palmeramaranth", "sicklepod", "goosegrass", "cutleaf"
]


def load_model(weights_path: str, device: str = "cuda"):
    path = Path(weights_path)
    ckpt = torch.load(path, map_location="cpu")

    # MambaTrainer checkpoint (full model object)
    if isinstance(ckpt, dict) and "model" in ckpt and not isinstance(ckpt["model"], dict):
        print("Loading MambaTrainer checkpoint (full model)...")
        model_nn = ckpt["model"].float().to(device).eval()
        # Wrap in YOLO shell for .predict() API
        yolo = YOLO("yolov8m.pt")
        yolo.model = model_nn
        return yolo

    # Standard YOLO checkpoint
    print("Loading standard YOLO checkpoint...")
    return YOLO(weights_path)


def run(weights, source, conf=0.25, iou=0.45, imgsz=640, device="cuda", save=True, show=False):
    model = load_model(weights, device)

    results = model.predict(
        source  = source,
        conf    = conf,
        iou     = iou,
        imgsz   = imgsz,
        device  = device,
        save    = save,
        show    = show,
        classes = list(range(14)),
    )

    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes):
            print(f"\n{Path(r.path).name}:")
            for box in boxes:
                cls  = int(box.cls)
                conf = float(box.conf)
                xyxy = box.xyxy[0].tolist()
                print(f"  {CLASS_NAMES[cls]:<20s} conf={conf:.3f}  box={[round(v,1) for v in xyxy]}")
        else:
            print(f"\n{Path(r.path).name}: no detections")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MambaWeed Inference")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--source",  required=True, help="Image/folder/video path")
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--iou",     type=float, default=0.45)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--device",  default="cuda")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--show",    action="store_true")
    args = parser.parse_args()

    run(
        weights = args.weights,
        source  = args.source,
        conf    = args.conf,
        iou     = args.iou,
        imgsz   = args.imgsz,
        device  = args.device,
        save    = not args.no_save,
        show    = args.show,
    )
