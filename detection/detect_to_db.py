"""
Phase 1: Vehicle Detection → DuckDB

Runs YOLOv10x (or any ultralytics model) on a directory of timestamped images,
detects vehicles/airplanes, and stores per-detection rows in a DuckDB database.

Resumable: images already in `images_processed` are skipped unless force_rerun=true.

Usage:
    python detect_to_db.py [--config config.yaml] [--force-rerun]
"""

import argparse
import math
import re
from datetime import datetime
from pathlib import Path

import cv2
import duckdb
import yaml
from tqdm import tqdm
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------
CATEGORY_MAP = {
    "airplane":   "A",
    "bus":        "B",
    "truck":      "B",
    "car":        "C",
    "motorcycle": "C",
}

TIMESTAMP_RE = re.compile(r"(\d{8})\.(\d{6})")


def parse_timestamp(filename: str) -> datetime | None:
    m = TIMESTAMP_RE.search(filename)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------
def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE SEQUENCE IF NOT EXISTS det_id_seq START 1;

        CREATE TABLE IF NOT EXISTS detections (
            id          INTEGER DEFAULT nextval('det_id_seq') PRIMARY KEY,
            ts          TIMESTAMP,
            image_file  VARCHAR,
            class_name  VARCHAR,
            category    VARCHAR,
            confidence  FLOAT,
            x1          FLOAT,
            y1          FLOAT,
            x2          FLOAT,
            y2          FLOAT,
            cx          FLOAT,
            cy          FLOAT,
            img_w       INTEGER,
            img_h       INTEGER
        );

        CREATE TABLE IF NOT EXISTS images_processed (
            image_file   VARCHAR PRIMARY KEY,
            ts           TIMESTAMP,
            n_detections INTEGER,
            processed_at TIMESTAMP DEFAULT now()
        );
    """)


def already_processed(con: duckdb.DuckDBPyConnection, image_file: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM images_processed WHERE image_file = ?", [image_file]
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def detect_image(model: YOLO, image_path: Path, cfg: dict) -> list[dict]:
    """Run inference on one image; return list of detection dicts."""
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    h, w = img.shape[:2]

    target_set = set(cfg["detection"]["target_classes"])
    results = model.predict(
        source=str(image_path),
        conf=cfg["model"]["conf"],
        iou=cfg["model"]["iou"],
        imgsz=cfg["model"]["imgsz"],
        verbose=False,
    )

    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            if class_name not in target_set:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            detections.append({
                "class_name": class_name,
                "category":   CATEGORY_MAP[class_name],
                "confidence": conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": cx,  "cy": cy,
                "img_w": w, "img_h": h,
            })
    return detections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(config_path: str, force_rerun: bool = False) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if force_rerun:
        cfg["analysis"]["force_rerun"] = True

    images_dir = Path(cfg["paths"]["images_dir"])
    db_path    = Path(cfg["paths"]["db_path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model : {cfg['model']['name']}")
    print(f"Images: {images_dir}")
    print(f"DB    : {db_path}")

    model = YOLO(cfg["model"]["name"])

    con = duckdb.connect(str(db_path))
    init_db(con)

    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.JPG"))
    image_files += sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.PNG"))
    image_files = sorted(set(image_files))

    skipped = processed = 0
    det_rows = []
    proc_rows = []

    for img_path in tqdm(image_files, desc="Detecting"):
        fname = img_path.name
        ts = parse_timestamp(fname)
        if ts is None:
            continue  # skip files without a parseable timestamp

        if not cfg["analysis"]["force_rerun"] and already_processed(con, fname):
            skipped += 1
            continue

        dets = detect_image(model, img_path, cfg)

        for d in dets:
            det_rows.append((
                ts, fname,
                d["class_name"], d["category"], d["confidence"],
                d["x1"], d["y1"], d["x2"], d["y2"],
                d["cx"], d["cy"],
                d["img_w"], d["img_h"],
            ))

        proc_rows.append((fname, ts, len(dets)))
        processed += 1

        # Flush every 500 images to avoid large in-memory accumulation
        if len(proc_rows) >= 500:
            _flush(con, det_rows, proc_rows)
            det_rows, proc_rows = [], []

    _flush(con, det_rows, proc_rows)
    con.close()

    print(f"\nDone. Processed: {processed}  Skipped (already done): {skipped}")


def _flush(con, det_rows, proc_rows):
    if det_rows:
        con.executemany("""
            INSERT INTO detections
              (ts, image_file, class_name, category, confidence,
               x1, y1, x2, y2, cx, cy, img_w, img_h)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, det_rows)
    if proc_rows:
        con.executemany("""
            INSERT OR REPLACE INTO images_processed
              (image_file, ts, n_detections)
            VALUES (?,?,?)
        """, proc_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Detect vehicles → DuckDB")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Re-detect images already in the database")
    args = parser.parse_args()
    run(args.config, force_rerun=args.force_rerun)
