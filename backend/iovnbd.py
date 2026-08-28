"""Loader for synchronized IO-VNBD smartphone/vehicle route pairs.

Model inputs are smartphone IMU only; vehicle odometry is used only as supervision.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

PHONE = ("ACCELEROMETER X", "ACCELEROMETER Y", "ACCELEROMETER Z", "GYROSCOPE Yaw", "GYROSCOPE Pitch", "GYROSCOPE Roll")

def _key(row: dict, starts: tuple[str, ...]) -> str:
    return next(key for key in row if any(key.strip().startswith(start) for start in starts))

def load_route(phone_csv: Path, vehicle_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned [ax,ay,az,gx,gy,gz] and vehicle speed targets in m/s."""
    with phone_csv.open(encoding="latin1", newline="") as fh: phone = list(csv.DictReader(fh))
    with vehicle_csv.open(encoding="utf-8-sig", newline="") as fh: vehicle = list(csv.DictReader(fh))
    if len(phone) != len(vehicle): raise ValueError(f"Unsynchronised pair: {len(phone)} vs {len(vehicle)}")
    keys = [_key(phone[0], (field,)) for field in PHONE]
    speed = _key(vehicle[0], ("Velocity",))
    x = np.asarray([[float(row[key]) for key in keys] for row in phone], dtype=np.float32)
    y = np.asarray([float(row[speed]) / 3.6 for row in vehicle], dtype=np.float32)
    keep = np.isfinite(x).all(axis=1) & np.isfinite(y)
    return x[keep], y[keep]

def route_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs=[]
    for phone in root.rglob("S-*.csv"):
        candidate = phone.with_name("V-" + phone.name[2:])
        # A repository clone can contain LFS pointer placeholders for routes that
        # have not been downloaded yet; never treat those as observations.
        if candidate.exists() and phone.stat().st_size > 1_000 and candidate.stat().st_size > 1_000:
            pairs.append((phone,candidate))
    return pairs
