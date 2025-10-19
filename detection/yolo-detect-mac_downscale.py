#!/usr/bin/env python3
"""
YOLOv7 object detection (MPS/CPU/CUDA) for static images.
Processes .jpg files and outputs object counts to CSV.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from datetime import datetime

# --------------------------------------------------------------------------
# Dynamic paths
# --------------------------------------------------------------------------
ROOT = Path("/Users/bhupendra/projects/yolov7/code/python/yolo/")
sys.path.append(str(ROOT))

from models.experimental import Ensemble
from models.common import Conv
from utils.general import non_max_suppression
from models.yolo import Model

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_class_names(namesfile: str):
    with open(namesfile, "r") as f:
        return [line.strip() for line in f.readlines()]

# --------------------------------------------------------------------------
# YOLOv7 Detector
# --------------------------------------------------------------------------
class YOLOv7Detector:
    """Lightweight YOLOv7 inference wrapper for MPS/CPU/CUDA."""

    def __init__(self, weight_path, class_file, conf_thr=0.3, iou_thr=0.45):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"[INFO] Using device: {self.device}")

        import torch.serialization as serialization
        serialization.add_safe_globals([Model])
        with serialization.safe_globals([Model]):
            ckpt = torch.load(weight_path, map_location=self.device, weights_only=False)

        self.model = Ensemble()
        self.model.append(ckpt["ema" if ckpt.get("ema") else "model"].float().fuse().eval())
        for m in self.model.modules():
            if isinstance(m, Conv):
                m._non_persistent_buffers_set = set()
        self.model.to(self.device)
        if self.device == "mps":
            self.model = self.model.to(torch.float16)

        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.class_names = load_class_names(class_file)
        print(f"[INFO] Loaded {len(self.class_names)} classes from {class_file}")

    def predict(self, img_path: Path):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Could not read {img_path}")
            return []

        h, w = frame.shape[:2]
        img = cv2.cvtColor(cv2.resize(frame, (640, 640)), cv2.COLOR_BGR2RGB)
        dtype = torch.float16 if self.device == "mps" else torch.float32
        tensor = torch.from_numpy((img.transpose(2, 0, 1) / 255.0).astype(np.float32)).unsqueeze(0).to(self.device, dtype=dtype)

        with torch.no_grad():
            pred = self.model(tensor)[0]
            pred = non_max_suppression(pred, self.conf_thr, self.iou_thr, agnostic=True)

        dets = []
        if len(pred) and len(pred[0]):
            for *xyxy, conf, cls in pred[0].cpu().numpy():
                x1, y1, x2, y2 = xyxy
                label = self.class_names[int(cls)]
                dets.append((label, float(conf), x1*w/640, y1*h/640, x2*w/640, y2*h/640))
        return dets


# --------------------------------------------------------------------------
# Main execution
# --------------------------------------------------------------------------

import re
from datetime import datetime

def extract_timestamp_from_filename(filename: str) -> str:
    """Extracts datetime from pattern YYYYMMDD.HHMMSS inside filename."""
    match = re.search(r"(\d{8})\.(\d{6})", filename)
    if match:
        date_str, time_str = match.groups()
        try:
            dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
            return dt.isoformat()
        except ValueError:
            pass
    return None


def main():
    base_dir = Path("/Users/bhupendra/projects/yolov7/")
    data_dir = base_dir / "data" / "238106"
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    weight_path = ROOT / "yolov7.pt"
    class_file = ROOT / "coco.names"
    detector = YOLOv7Detector(weight_path, class_file, conf_thr=0.3, iou_thr=0.45)

    image_files = sorted(data_dir.glob("*.jpg"))
    if not image_files:
        print(f"[WARN] No .jpg files found in {data_dir}")
        return

    print(f"[INFO] Found {len(image_files)} images. Processing every 10th for test run...")

    # Vehicle types of interest
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
    records = []

    for idx, img_path in enumerate(image_files):
        #if idx % 10 != 0:
        #    continue
        timestamp = extract_timestamp_from_filename(img_path.name)
        

        dets = detector.predict(img_path)
        counts = {cls: 0 for cls in vehicle_classes}

        for label, *_ in dets:
            if label in counts:
                counts[label] += 1

        record = {"datetime": timestamp, "image": img_path.name}
        record.update(counts)
        records.append(record)
        print(f"[{idx+1}/{len(image_files)}] {img_path.name} → {counts}")

    # Ensure at least one row even if no detections
    if not records:
        records = [{"datetime": datetime.now().isoformat(), "image": "none", **{cls: 0 for cls in vehicle_classes}}]

    df = pd.DataFrame(records)
    for cls in vehicle_classes:
        if cls not in df.columns:
            df[cls] = 0
    df = df.fillna(0).astype({cls: int for cls in vehicle_classes})

    csv_path = output_dir / "detections.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n✅ CSV saved at {csv_path}")
    print(df.head())


# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
