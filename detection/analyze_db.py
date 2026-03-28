"""
Phase 2: DuckDB → Activity CSV

Reads raw detections from DuckDB, computes per-image-pair movement events
per category, aggregates to configurable time bins, and writes a CSV.

Categories:
  A = airplane
  B = bus + truck  (big vehicles)
  C = car + motorcycle

Output CSV columns (example at 15-min bins):
  datetime, A_events, B_events, C_events, A_count, B_count, C_count

Usage:
    python analyze_db.py [--config config.yaml]
"""

import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import yaml


CATEGORIES = ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Movement detection helpers
# ---------------------------------------------------------------------------

def centroid_distance(c1: tuple, c2: tuple) -> float:
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def image_diagonal(img_w: int, img_h: int) -> float:
    return math.sqrt(img_w ** 2 + img_h ** 2)


def count_movement_events(
    prev_boxes: list[tuple],  # list of (cx, cy)
    curr_boxes: list[tuple],
    threshold: float,
) -> int:
    """
    Greedy nearest-neighbour matching between consecutive frame centroids.
    Returns number of movement events:
      - count change (appear/disappear)
      - matched pairs whose centroid shifted > threshold
    """
    events = 0
    n_prev, n_curr = len(prev_boxes), len(curr_boxes)

    # Count-change events (vehicles appeared or disappeared)
    events += abs(n_prev - n_curr)

    # Match the smaller set to the larger; check centroid displacement
    if n_prev == 0 or n_curr == 0:
        return events

    used = set()
    for p in prev_boxes:
        best_dist = float("inf")
        best_j = -1
        for j, c in enumerate(curr_boxes):
            if j in used:
                continue
            d = centroid_distance(p, c)
            if d < best_dist:
                best_dist = d
                best_j = j
        if best_j >= 0:
            used.add(best_j)
            if best_dist > threshold:
                events += 1

    return events


# ---------------------------------------------------------------------------
# Frame-pair analysis
# ---------------------------------------------------------------------------

def build_frame_events(df: pd.DataFrame, movement_threshold_pct: float) -> pd.DataFrame:
    """
    For each consecutive image pair, compute per-category events and peak counts.
    Returns a DataFrame with columns: ts, category, events, count.
    """
    # Get all unique image timestamps, sorted
    images = sorted(df["ts"].unique())

    records = []
    prev_ts = None

    for ts in images:
        frame = df[df["ts"] == ts]

        # Get image dimensions for threshold (use first row)
        if len(frame) > 0:
            img_w = int(frame["img_w"].iloc[0])
            img_h = int(frame["img_h"].iloc[0])
        else:
            img_w, img_h = 3840, 2160  # fallback

        diag = image_diagonal(img_w, img_h)
        threshold = movement_threshold_pct * diag

        for cat in CATEGORIES:
            curr_boxes = list(zip(
                frame[frame["category"] == cat]["cx"],
                frame[frame["category"] == cat]["cy"],
            ))
            curr_count = len(curr_boxes)

            if prev_ts is not None:
                prev_frame = df[df["ts"] == prev_ts]
                prev_boxes = list(zip(
                    prev_frame[prev_frame["category"] == cat]["cx"],
                    prev_frame[prev_frame["category"] == cat]["cy"],
                ))
                evts = count_movement_events(prev_boxes, curr_boxes, threshold)
            else:
                evts = 0  # first frame has no prior to compare

            records.append({
                "ts":       ts,
                "category": cat,
                "events":   evts,
                "count":    curr_count,
            })

        prev_ts = ts

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Time-bin aggregation
# ---------------------------------------------------------------------------

def floor_to_bin(ts: pd.Timestamp, bin_minutes: int) -> pd.Timestamp:
    """Floor a timestamp to the nearest bin boundary, then advance by one bin."""
    total_minutes = ts.hour * 60 + ts.minute
    floored = (total_minutes // bin_minutes) * bin_minutes
    base = ts.normalize() + pd.Timedelta(minutes=floored)
    return base + pd.Timedelta(minutes=bin_minutes)  # label = end of bin


def aggregate_to_bins(
    frame_events: pd.DataFrame,
    intermediate_bin: int,
    output_bin: int,
) -> pd.DataFrame:
    """
    Two-stage aggregation:
      1. frame_events → intermediate bins (sum events, max count)
      2. intermediate bins → output bins (sum events, max count)
    """
    fe = frame_events.copy()
    fe["bin_mid"] = fe["ts"].apply(
        lambda t: floor_to_bin(pd.Timestamp(t), intermediate_bin)
    )

    mid = (
        fe.groupby(["bin_mid", "category"])
        .agg(events=("events", "sum"), count=("count", "max"))
        .reset_index()
    )

    mid["bin_out"] = mid["bin_mid"].apply(
        lambda t: floor_to_bin(t - pd.Timedelta(seconds=1), output_bin)
    )

    out = (
        mid.groupby(["bin_out", "category"])
        .agg(events=("events", "sum"), count=("count", "max"))
        .reset_index()
        .rename(columns={"bin_out": "datetime"})
    )
    return out


def pivot_to_csv_format(aggregated: pd.DataFrame) -> pd.DataFrame:
    """Pivot long → wide; fill missing bins with zeros."""
    events_wide = aggregated.pivot(
        index="datetime", columns="category", values="events"
    ).fillna(0).astype(int)
    events_wide.columns = [f"{c}_events" for c in events_wide.columns]

    counts_wide = aggregated.pivot(
        index="datetime", columns="category", values="count"
    ).fillna(0).astype(int)
    counts_wide.columns = [f"{c}_count" for c in counts_wide.columns]

    result = events_wide.join(counts_wide)

    # Ensure all expected columns exist (even if category never detected)
    for cat in CATEGORIES:
        for suffix in ("_events", "_count"):
            col = f"{cat}{suffix}"
            if col not in result.columns:
                result[col] = 0

    # Reorder columns predictably: A_events, B_events, C_events, A_count, B_count, C_count
    col_order = [f"{cat}_events" for cat in CATEGORIES] + [f"{cat}_count" for cat in CATEGORIES]
    result = result[col_order].sort_index()
    result.index.name = "datetime"
    return result.reset_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    db_path    = cfg["paths"]["db_path"]
    output_csv = Path(cfg["paths"]["output_csv"])
    mv_thresh  = cfg["detection"]["movement_threshold_pct"]
    int_bin    = cfg["analysis"]["intermediate_bin_minutes"]
    out_bin    = cfg["analysis"]["output_bin_minutes"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading detections from: {db_path}")
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(
        "SELECT ts, category, cx, cy, img_w, img_h FROM detections ORDER BY ts"
    ).df()
    con.close()

    if df.empty:
        print("No detections found in database. Run detect_to_db.py first.")
        return

    print(f"  {len(df):,} detection rows across {df['ts'].nunique():,} images")

    print("Computing frame-pair movement events …")
    frame_events = build_frame_events(df, mv_thresh)

    print(f"Aggregating: {int_bin}-min intermediate → {out_bin}-min output bins …")
    aggregated = aggregate_to_bins(frame_events, int_bin, out_bin)

    result = pivot_to_csv_format(aggregated)
    result.to_csv(output_csv, index=False)
    print(f"\nSaved activity CSV → {output_csv}")
    print(f"  Rows: {len(result)}  ({out_bin}-minute bins)")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: DuckDB → activity CSV")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    run(args.config)
