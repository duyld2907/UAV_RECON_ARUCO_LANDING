#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt


log_dir = Path.home() / "UAV_RECON_ARUCO_LANDING/hinh_quy_dao"
csv_path = log_dir / "wp_accuracy_log.csv"

if not csv_path.exists():
    raise SystemExit(f"Chưa có file log: {csv_path}")

with open(csv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

rows = [r for r in rows if r.get("mission_id")]

if not rows:
    raise SystemExit("File CSV chưa có dữ liệu waypoint. Hãy bay mission trước.")

# Nếu truyền mission_id thì vẽ mission đó, không thì lấy mission mới nhất
if len(sys.argv) >= 2:
    mission_id = sys.argv[1]
else:
    mission_id = rows[-1]["mission_id"]

rows = [r for r in rows if r["mission_id"] == mission_id]
rows.sort(key=lambda r: int(r["wp_index"]))

if len(rows) < 2:
    raise SystemExit("Cần ít nhất 2 waypoint để vẽ quỹ đạo.")

lat0 = float(rows[0]["target_lat"])
lon0 = float(rows[0]["target_lon"])

def gps_to_xy(lat, lon):
    x = (lon - lon0) * 111111.0 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 111111.0
    return x, y

target_xy = [
    gps_to_xy(float(r["target_lat"]), float(r["target_lon"]))
    for r in rows
]

actual_xy = [
    gps_to_xy(float(r["reached_lat"]), float(r["reached_lon"]))
    for r in rows
]

# Mặc định nối điểm cuối về điểm đầu để vẽ hình tròn/vuông/tam giác rõ hơn.
# Nếu không muốn nối kín, chạy: CLOSE_TRAJECTORY=0 python3 ve_quy_dao_wp.py
close_path = os.environ.get("CLOSE_TRAJECTORY", "1") == "1"

target_plot = target_xy + [target_xy[0]] if close_path and len(target_xy) >= 3 else target_xy
actual_plot = actual_xy + [actual_xy[0]] if close_path and len(actual_xy) >= 3 else actual_xy

tx, ty = zip(*target_plot)
ax, ay = zip(*actual_plot)

plt.figure(figsize=(8, 7))
plt.plot(tx, ty, "--o", linewidth=2, label="Quỹ đạo mong muốn")
plt.plot(ax, ay, "-o", linewidth=2, label="Quỹ đạo bay thực tế")

for i, (x, y) in enumerate(target_xy, start=1):
    plt.text(x, y, f" W{i}", fontsize=9)

for i, (x, y) in enumerate(actual_xy, start=1):
    plt.text(x, y, f" A{i}", fontsize=9)

plt.xlabel("Trục X - hướng Đông/Tây (m)")
plt.ylabel("Trục Y - hướng Bắc/Nam (m)")
plt.title(f"So sánh quỹ đạo waypoint mong muốn và thực tế\nMission ID: {mission_id}")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.tight_layout()

out_path = log_dir / f"quy_dao_xy_{mission_id}.png"
latest_path = log_dir / "quy_dao_xy_latest.png"

plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.savefig(latest_path, dpi=300, bbox_inches="tight")

print(f"Đã lưu hình: {out_path}")
print(f"Đã lưu bản mới nhất: {latest_path}")
