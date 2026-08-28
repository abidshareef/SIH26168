from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

LAT0, LON0 = 17.3850, 78.4867
FIELDS = ["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "latitude", "longitude", "altitude", "speed", "heading", "accuracy", "gnss_available", "reference_latitude", "reference_longitude", "reference_velocity", "reference_heading", "reference_acceleration", "reference_angular_velocity"]


def _trajectory(seconds: int, seed: int, outage: tuple[float, float] | None = None) -> list[dict]:
    rng = np.random.default_rng(seed); dt = 0.1; n = int(seconds / dt)
    speed = 0.0; heading = 0.0; north = east = 0.0; ax_bias = rng.normal(0, .06); gz_bias = rng.normal(0, .008)
    rows: list[dict] = []
    for i in range(n):
        t = round(i * dt, 2); phase = (t % 36.0)
        # stationary, accelerate, cruise, turn, brake/stop-go, all deterministic
        accel = 0.0 if phase < 4 else (1.0 if phase < 11 else (-0.75 if phase > 29 else 0.0))
        yaw = 0.0 if not 17 < phase < 27 else (0.13 if (seed + int(t // 36)) % 2 else -0.13)
        speed = float(np.clip(speed + accel * dt, 0, 18))
        heading += yaw * dt; north += speed * np.cos(heading) * dt; east += speed * np.sin(heading) * dt
        ref_lat = LAT0 + north / 111_111; ref_lon = LON0 + east / (111_111 * np.cos(np.deg2rad(LAT0)))
        unavailable = outage is not None and outage[0] <= t < outage[1]
        accuracy = 1.8 + abs(rng.normal(0, .5)) + (4.0 if i % 173 == 0 else 0.0)
        pos_noise_n, pos_noise_e = rng.normal(0, accuracy, 2)
        rows.append({"timestamp": t, "ax": accel + ax_bias + rng.normal(0,.10), "ay": speed*yaw + rng.normal(0,.12), "az": 9.81+rng.normal(0,.08), "gx": rng.normal(0,.01), "gy": rng.normal(0,.01), "gz": yaw+gz_bias+rng.normal(0,.015), "latitude": "" if unavailable else ref_lat+pos_noise_n/111_111, "longitude": "" if unavailable else ref_lon+pos_noise_e/(111_111*np.cos(np.deg2rad(LAT0))), "altitude": "" if unavailable else 540+rng.normal(0,1), "speed": "" if unavailable else max(0,speed+rng.normal(0,.35)), "heading": "" if unavailable else np.rad2deg(heading)%360, "accuracy": "" if unavailable else accuracy, "gnss_available": int(not unavailable), "reference_latitude": ref_lat, "reference_longitude": ref_lon, "reference_velocity": speed, "reference_heading": np.rad2deg(heading)%360, "reference_acceleration": accel, "reference_angular_velocity": yaw})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)


def generate_dataset(output: Path = Path("data/synthetic"), trajectories: int = 18, seconds: int = 45) -> dict[str, Path]:
    """Generate compact deterministic train/validation/test datasets; no real-world claim."""
    splits = {"train": range(trajectories), "validation": range(trajectories, trajectories + 4), "test": range(trajectories + 4, trajectories + 7)}
    result = {}
    for name, seeds in splits.items():
        rows = [row for seed in seeds for row in _trajectory(seconds, seed)]
        result[name] = output / f"{name}.csv"; write_csv(result[name], rows)
    demo = output / "demo_outage.csv"; write_csv(demo, _trajectory(90, 999, (30, 60))); result["demo"] = demo
    return result
