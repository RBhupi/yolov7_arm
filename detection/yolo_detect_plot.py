#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 06:40:38 2025
Vehicle presence time series with colored circles and readable datetime axis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------
csv_path = Path("/Users/bhupendra/projects/yolov7/output/detections.csv")
df = pd.read_csv(csv_path)

vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'aeroplane']

# Ensure all columns exist
for cls in vehicle_classes:
    if cls not in df.columns:
        df[cls] = 0

# Parse datetime column
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=False)
df = df.dropna(subset=['datetime']).sort_values('datetime')
df = df.fillna(0)

# -------------------------------------------------------------------------
# Prepare long-form dataframe
# -------------------------------------------------------------------------
long_df = df.melt(
    id_vars=['datetime', 'image'],
    value_vars=vehicle_classes,
    var_name='vehicle',
    value_name='count'
)
long_df = long_df[long_df['count'] > 0]

# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
colors = {
    'car': '#1f77b4',
    'truck': '#ff7f0e',
    'bus': '#2ca02c',
    'motorcycle': '#d62728',
    'bicycle': '#9467bd'
}

for cls in vehicle_classes:
    subset = long_df[long_df['vehicle'] == cls]
    if not subset.empty:
        ax.scatter(
            subset['datetime'],
            [cls]*len(subset),
            s=subset['count']*60 + 30,
            color=colors.get(cls, 'gray'),
            alpha=0.7,
            label=cls
        )

# Format the x-axis as dates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
fig.autofmt_xdate(rotation=45)

ax.set_title("Vehicle detections over time")
ax.set_xlabel("Timestamp [local]")
ax.set_ylabel("Vehicle type [-]")
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
