"""
Make Detection Video

Assembles all input images into an MP4 with YOLO detection labels drawn from
the DuckDB database.  No re-inference is needed — labels come from stored
bounding boxes.

For each image:
  - Bounding box drawn in category colour (A=orange, B=red, C=blue)
  - Label: class name + confidence, e.g.  "airplane  0.81"
  - Timestamp overlay (top-left) and frame counter (top-right)
  - Images with no detections are included as plain frames (important for
    showing the full time-lapse, including quiet periods)

Usage:
    python make_detection_video.py [--config config.yaml] [--fps 8] [--out output/detection_video.mp4]
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import cv2
import duckdb
import numpy as np
import yaml
from tqdm import tqdm

# ── Visual style ─────────────────────────────────────────────────────────────
CAT_BGR = {
    "A": (0,   140, 224),   # orange  (BGR)
    "B": (44,   57, 192),   # red
    "C": (178, 128,  41),   # blue
}
CAT_LABEL = {
    "A": "Airplane",
    "B": "Big vehicle",
    "C": "Car/Moto",
}

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.65
THICKNESS  = 2
BOX_THICK  = 2

TIMESTAMP_RE = re.compile(r"(\d{8})\.(\d{6})")


def parse_ts(filename: str) -> datetime | None:
    m = TIMESTAMP_RE.search(filename)
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") if m else None


def draw_box(img, x1, y1, x2, y2, label: str, color_bgr: tuple, confidence: float):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # Bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, BOX_THICK)

    # Label background
    text = f"{label}  {confidence:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
    pad = 4
    label_y = y1 - th - pad * 2 if y1 - th - pad * 2 > 0 else y2 + th + pad * 2
    cv2.rectangle(
        img,
        (x1, label_y - th - pad),
        (x1 + tw + pad * 2, label_y + pad),
        color_bgr, -1,
    )
    # Text (white on coloured background)
    cv2.putText(
        img, text,
        (x1 + pad, label_y),
        FONT, FONT_SCALE, (255, 255, 255), THICKNESS, cv2.LINE_AA,
    )


def draw_timestamp(img, ts: datetime, frame_idx: int, total: int):
    h, w = img.shape[:2]
    ts_str = ts.strftime("%Y-%m-%d  %H:%M:%S")
    frame_str = f"Frame {frame_idx+1}/{total}"

    # Semi-transparent dark bar at top
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    cv2.putText(img, ts_str,    (12, 24), FONT, 0.70, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, frame_str, (w - 200, 24), FONT, 0.60, (160, 160, 160), 1, cv2.LINE_AA)


def draw_legend(img):
    """Small category legend in bottom-left corner."""
    h, w = img.shape[:2]
    entries = [
        ("A – Airplane",       CAT_BGR["A"]),
        ("B – Bus / Truck",    CAT_BGR["B"]),
        ("C – Car / Moto",     CAT_BGR["C"]),
    ]
    x0, y0 = 14, h - 14 - len(entries) * 26
    for i, (label, color) in enumerate(entries):
        y = y0 + i * 26
        cv2.rectangle(img, (x0, y - 14), (x0 + 18, y + 4), color, -1)
        cv2.putText(img, label, (x0 + 26, y), FONT, 0.55, (230, 230, 230), 1, cv2.LINE_AA)


# ── Main ─────────────────────────────────────────────────────────────────────

def run(config_path: str, fps: int, out_path: str | None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    images_dir = Path(cfg["paths"]["images_dir"])
    db_path    = cfg["paths"]["db_path"]
    out_video  = Path(out_path) if out_path else \
                 Path(cfg["paths"]["db_path"]).parent / "detection_video.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)

    # ── Gather all images sorted by timestamp ────────────────────────────
    all_images = sorted(
        list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.JPG")) +
        list(images_dir.glob("*.png")) + list(images_dir.glob("*.PNG"))
    )
    all_images = [p for p in all_images if parse_ts(p.name) is not None]
    all_images.sort(key=lambda p: parse_ts(p.name))

    if not all_images:
        print(f"No images found in {images_dir}")
        return

    # ── Load all detections into a dict keyed by image filename ─────────
    print(f"Loading detections from {db_path} …")
    con = duckdb.connect(db_path, read_only=True)
    det_df = con.execute(
        "SELECT image_file, class_name, category, confidence, x1, y1, x2, y2 "
        "FROM detections ORDER BY image_file"
    ).df()
    con.close()

    det_by_file: dict[str, list] = {}
    for row in det_df.itertuples(index=False):
        det_by_file.setdefault(row.image_file, []).append(row)

    n_with_dets = len(det_by_file)
    print(f"  {len(det_df):,} detections across {n_with_dets:,} images")
    print(f"  {len(all_images):,} total images → {fps} fps → "
          f"~{len(all_images)/fps:.0f}s video")

    # ── Probe first image for frame size ────────────────────────────────
    probe = cv2.imread(str(all_images[0]))
    if probe is None:
        print(f"Cannot read {all_images[0]}")
        return
    h, w = probe.shape[:2]
    print(f"  Frame size: {w}×{h}")

    # ── Set up VideoWriter ───────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))
    if not writer.isOpened():
        print("ERROR: cv2.VideoWriter failed to open — check codec/path.")
        return

    # ── Process frames ───────────────────────────────────────────────────
    total = len(all_images)
    for idx, img_path in enumerate(tqdm(all_images, desc="Building video")):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        ts = parse_ts(img_path.name)

        # Draw detections from DB
        dets = det_by_file.get(img_path.name, [])
        for d in dets:
            color = CAT_BGR.get(d.category, (180, 180, 180))
            draw_box(frame, d.x1, d.y1, d.x2, d.y2, d.class_name, color, d.confidence)

        draw_timestamp(frame, ts, idx, total)
        draw_legend(frame)
        writer.write(frame)

    writer.release()
    print(f"\nVideo saved → {out_video}")
    print(f"  Frames: {total}   Duration: {total/fps:.1f}s   FPS: {fps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build detection overlay video from DuckDB labels")
    parser.add_argument("--config", default="config.yaml",  help="Path to config.yaml")
    parser.add_argument("--fps",    type=int, default=8,    help="Frames per second (default 8)")
    parser.add_argument("--out",    default=None,           help="Output MP4 path (default: output/detection_video.mp4)")
    args = parser.parse_args()
    run(args.config, args.fps, args.out)
