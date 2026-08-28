from __future__ import annotations

import numpy as np


FEATURE_COLUMNS = ("ax", "ay", "az", "gx", "gy", "gz")


def ordered_rows(rows: list[dict]) -> list[dict]:
    """Sort replay records and retain the final record for duplicate timestamps."""
    unique = {float(row["timestamp"]): row for row in rows}
    return [unique[t] for t in sorted(unique)]


def imu_features(rows: list[dict], means: np.ndarray | None = None, stds: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray([[float(row[name]) for name in FEATURE_COLUMNS] for row in rows], dtype=np.float32)
    if means is None:
        means = values.mean(axis=0)
    if stds is None:
        stds = values.std(axis=0)
    return (values - means) / np.maximum(stds, 1e-5)


def create_windows(features: np.ndarray, labels: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if len(features) < window:
        return np.empty((0, window, features.shape[1]), np.float32), np.empty((0,), np.float32)
    x = np.stack([features[i - window + 1:i + 1] for i in range(window - 1, len(features))])
    return x.astype(np.float32), labels[window - 1:].astype(np.float32)


def motion_quality(window: np.ndarray) -> float:
    """Transparent [0,1] signal-quality proxy based on finite values and jerk."""
    if not np.isfinite(window).all():
        return 0.0
    jerk = float(np.mean(np.abs(np.diff(window[:, :3], axis=0))))
    return float(np.clip(np.exp(-0.2 * jerk), 0.05, 1.0))
