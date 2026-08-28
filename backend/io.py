from __future__ import annotations
import csv
from pathlib import Path

FLOAT_FIELDS = {"timestamp", "ax", "ay", "az", "gx", "gy", "gz", "latitude", "longitude", "altitude", "speed", "heading", "accuracy", "reference_latitude", "reference_longitude", "reference_velocity", "reference_heading"}

def read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            for key in FLOAT_FIELDS:
                if key in row and row[key] != "": row[key] = float(row[key])
            rows.append(row)
    return rows
