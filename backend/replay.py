from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .contracts import GnssSample, NavigationState
from .io import read_csv
from .models import TCNVelocityModel
from .navigation import NavigationEngine
from .train import train


def _trajectory_distance(rows: list[dict]) -> float:
    if len(rows) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(rows, rows[1:]):
        north = (b["reference_latitude"] - a["reference_latitude"]) * 111_111.0
        east = (b["reference_longitude"] - a["reference_longitude"]) * 111_111.0 * np.cos(np.deg2rad(a["reference_latitude"]))
        total += float(np.hypot(north, east))
    return total


def run_replay(
    data: str | Path,
    outage_start: float | None = None,
    outage_duration: float = 0,
    checkpoint: Path = Path("models/tcn_velocity.pt"),
    log_path: Path | None = None,
) -> dict:
    if not checkpoint.exists():
        train(checkpoint=checkpoint)

    rows = read_csv(data)
    model = TCNVelocityModel(checkpoint)
    first = rows[0]
    initial = NavigationState(
        first["timestamp"],
        first["reference_latitude"],
        first["reference_longitude"],
        540,
        0,
        first["reference_heading"],
        3.0,
        4.0,
    )
    engine = NavigationEngine(initial)
    logs: list[dict] = []
    window = 20

    for i, row in enumerate(rows):
        if i < window - 1:
            continue
        w = np.asarray(
            [[r[k] for k in ("ax", "ay", "az", "gx", "gy", "gz")] for r in rows[i - window + 1 : i + 1]],
            np.float32,
        )
        prediction = model.predict(w, row["timestamp"])
        forced = outage_start is not None and outage_start <= row["timestamp"] < outage_start + outage_duration
        gnss = None if forced or not int(row["gnss_available"]) else GnssSample(
            row["timestamp"], row["latitude"], row["longitude"], row["altitude"],
            row["speed"], row["heading"], row["accuracy"],
        )
        state = engine.update(row["timestamp"], row["ax"], row["gz"], prediction, gnss)
        north_error = (state.latitude - row["reference_latitude"]) * 111_111.0
        east_error = (state.longitude - row["reference_longitude"]) * 111_111.0 * np.cos(np.deg2rad(state.latitude))
        error = float(np.hypot(north_error, east_error))
        logs.append({
            **state.to_dict(),
            "reference_velocity": row["reference_velocity"],
            "position_error_m": error,
            "ai_uncertainty": prediction.uncertainty,
        })

    if not logs:
        raise ValueError("Replay produced no samples; provide a sequence with at least 20 sensor rows.")

    outage = [r for r in logs if r["mode"] == "DEAD_RECKONING"]
    distance_m = _trajectory_distance(logs)
    max_outage_drift_m = max((r["position_error_m"] for r in outage), default=0.0)
    drift_percent = (max_outage_drift_m / distance_m * 100.0) if distance_m > 0 else 0.0

    recovery_latency_s = None
    if outage:
        outage_end = max(r["timestamp"] for r in outage)
        recovered = [r for r in logs if r["timestamp"] >= outage_end and r["mode"] == "GNSS"]
        if recovered:
            recovery_latency_s = float(recovered[0]["timestamp"] - outage_end)

    timestamps = np.asarray([r["timestamp"] for r in logs], dtype=float)
    update_hz = float(1.0 / np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.0
    result = {
        "logs": logs,
        "position_rmse_m": float(np.sqrt(np.mean([r["position_error_m"] ** 2 for r in logs]))),
        "velocity_mae_mps": float(np.mean([abs(r["velocity"] - r["reference_velocity"]) for r in logs])),
        "max_outage_drift_m": max_outage_drift_m,
        "drift_percent_of_reference_distance": drift_percent,
        "reference_distance_m": distance_m,
        "recovery_latency_s": recovery_latency_s,
        "effective_update_hz": update_hz,
        "outage_samples": len(outage),
        "final_state": logs[-1],
    }
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NAV-AION GNSS-outage replay and evaluation")
    p.add_argument("--data", default="data/synthetic/demo_outage.csv")
    p.add_argument("--outage-start", type=float, default=30)
    p.add_argument("--outage-duration", type=float, default=30)
    args = p.parse_args()
    r = run_replay(args.data, args.outage_start, args.outage_duration, log_path=Path("outputs/replay.json"))
    f = r["final_state"]
    recovery = "n/a" if r["recovery_latency_s"] is None else f"{r['recovery_latency_s']:.2f} s"
    print(
        "NAV-AION NAVIGATION REPLAY — SYNTHETIC\n"
        f"Position RMSE: {r['position_rmse_m']:.2f} m | "
        f"max outage drift: {r['max_outage_drift_m']:.2f} m | "
        f"drift ratio: {r['drift_percent_of_reference_distance']:.2f}%\n"
        f"Velocity MAE: {r['velocity_mae_mps']:.3f} m/s | "
        f"effective update rate: {r['effective_update_hz']:.2f} Hz | "
        f"recovery latency: {recovery}\n"
        f"Final mode: {f['mode']} | uncertainty: {f['position_uncertainty']:.2f} m\n"
        "Replay log: outputs/replay.json"
    )
