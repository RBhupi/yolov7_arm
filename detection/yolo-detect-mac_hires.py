#!/usr/bin/env python3
"""
YOLOv7 object detection (MPS/CPU/CUDA) for ARM ENA images.
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
# --- yolo_small_object_boost.py
import math
import numpy as np
import torch
from pathlib import Path
import cv2
from typing import List, Tuple

from models.experimental import Ensemble
from models.common import Conv
from models.yolo import Model
from utils.general import non_max_suppression

# ---------- image preproc ----------
def letterbox(im: np.ndarray, new_shape=640, color=(114,114,114), stride=32) -> Tuple[np.ndarray, float, Tuple[int,int]]:
    """Resize with unchanged aspect ratio, padding to multiple of stride."""
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2

    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)

def to_tensor(im_rgb_chw01: np.ndarray, device: str, use_fp16: bool):
    """Numpy HWC uint8 → CHW float32/16 in [0,1]."""
    arr = (im_rgb_chw01.transpose(2,0,1) / 255.0).astype(np.float32)
    dtype = torch.float16 if (use_fp16 and device == "mps") else torch.float32
    return torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=dtype)

# ---------- postproc ----------
def scale_coords(xyxy: np.ndarray, r: float, pad: Tuple[int,int]) -> np.ndarray:
    """Map coords from letterboxed space back to original image."""
    x1,y1,x2,y2 = xyxy
    x1 = (x1 - pad[0]) / r
    x2 = (x2 - pad[0]) / r
    y1 = (y1 - pad[1]) / r
    y2 = (y2 - pad[1]) / r
    return np.array([x1,y1,x2,y2], dtype=np.float32)

# ---------- detector ----------
class YOLOv7Detector:
    """YOLOv7 inference with small-object friendly options: bigger img, letterbox, multiscale, tiling."""
    def __init__(self, weight_path: Path, class_file: Path,
                 conf_thr=0.15, iou_thr=0.55, img_size=1280,
                 multiscales: Tuple[float,...]=(1.0,),  # e.g., (0.75, 1.0, 1.25)
                 tile: Tuple[int,int]=(1,1), tile_overlap: float=0.15,
                 use_fp16=True):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.use_fp16 = use_fp16
        
        print(f"[INFO] Model loaded on {self.device} (FP16={self.use_fp16})")
        
        import torch.serialization as serialization
        serialization.add_safe_globals([Model])
        with serialization.safe_globals([Model]):
            ckpt = torch.load(str(weight_path), map_location=self.device, weights_only=False)

        self.model = Ensemble()
        self.model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())
        for m in self.model.modules():
            if isinstance(m, Conv):
                m._non_persistent_buffers_set = set()
        self.model.to(self.device)
        if self.device == "mps" and self.use_fp16:
            self.model = self.model.to(torch.float16)

        with open(class_file, 'r') as f:
            self.names = [ln.strip() for ln in f]

        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.img_size = img_size
        self.multiscales = multiscales
        self.tile = tile
        self.tile_overlap = tile_overlap

    # --- tiling ---
    def _tiles(self, img: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int,int]]]:
        if self.tile == (1,1):
            return [(img, (0,0))]
        h, w = img.shape[:2]
        nx, ny = self.tile
        ox = int(self.tile_overlap * w)
        oy = int(self.tile_overlap * h)
        tiles = []
        xs = np.linspace(0, w, nx+1, dtype=int)
        ys = np.linspace(0, h, ny+1, dtype=int)
        for i in range(nx):
            for j in range(ny):
                x1 = max(0, xs[i] - (ox if i>0 else 0))
                y1 = max(0, ys[j] - (oy if j>0 else 0))
                x2 = min(w, xs[i+1] + (ox if i<nx-1 else 0))
                y2 = min(h, ys[j+1] + (oy if j<ny-1 else 0))
                tiles.append((img[y1:y2, x1:x2], (x1,y1)))
        return tiles

    def _infer_once(self, bgr: np.ndarray, scale: float=1.0):
        """Infer on single frame with one scale; returns raw xyxy, conf, cls in original image coords."""
        H, W = bgr.shape[:2]
        target = int(round(self.img_size * scale))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lb, r, pad = letterbox(rgb, new_shape=target, stride=32)
        t = to_tensor(lb, self.device, self.use_fp16)

        with torch.no_grad():
            pred = self.model(t)[0]
            det = non_max_suppression(pred, self.conf_thr, self.iou_thr, agnostic=True, max_det=1000)[0]
        out = []
        if det is not None and len(det):
            det = det.cpu().numpy()
            for x1,y1,x2,y2,conf,cls in det:
                xyxy0 = scale_coords(np.array([x1,y1,x2,y2], dtype=np.float32), r, pad)
                # clip to image
                xyxy0[0::2] = np.clip(xyxy0[0::2], 0, W-1)
                xyxy0[1::2] = np.clip(xyxy0[1::2], 0, H-1)
                out.append((xyxy0, float(conf), int(cls)))
        return out

    def predict(self, img_path: Path):
        """Run multiscale + tiling; merge via NMS; return [(label, conf, x1,y1,x2,y2)]."""
        frame = cv2.imread(str(img_path))
        if frame is None:
            return []

        all_boxes = []
        # tiling for far/small targets
        for tile_img, (offx, offy) in self._tiles(frame):
            for s in self.multiscales:
                dets = self._infer_once(tile_img, scale=s)
                for (x1,y1,x2,y2), conf, cls in dets:
                    all_boxes.append([x1+offx, y1+offy, x2+offx, y2+offy, conf, cls])

        if not all_boxes:
            return []
        
          # merge detections from tiles and scales without reapplying YOLO's NMS
        result = []
        for x1, y1, x2, y2, conf, cls in all_boxes:
            label = self.names[int(cls)]
            result.append((label, float(conf), float(x1), float(y1), float(x2), float(y2)))
        return result
        



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
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'aeroplane']
    records = []

    for idx, img_path in enumerate(image_files):
        if idx % 10 != 0:
            continue
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
