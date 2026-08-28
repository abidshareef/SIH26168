from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import torch
from torch import nn

from .contracts import VelocityPrediction
from .preprocessing import motion_quality


class VelocityModel(ABC):
    @abstractmethod
    def predict(self, window: np.ndarray, timestamp: float) -> VelocityPrediction: ...


class MockVelocityModel(VelocityModel):
    """Fallback based on recent longitudinal acceleration; only for development."""
    def predict(self, window: np.ndarray, timestamp: float) -> VelocityPrediction:
        speed = max(0.0, float(np.cumsum(window[:, 0]).mean() * .1 + 8.0))
        uncertainty = 2.0 / max(motion_quality(window), .1)
        return VelocityPrediction(timestamp, speed, uncertainty, float(np.exp(-uncertainty / 4)))


class TemporalConvNet(nn.Module):
    def __init__(self, channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(6, channels, 3, padding=2, dilation=1), nn.ReLU(), nn.Conv1d(channels, channels, 3, padding=4, dilation=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x.transpose(1, 2)).squeeze(-1)).squeeze(-1)


class TCNVelocityModel(VelocityModel):
    def __init__(self, checkpoint: Path, residual_std: float = 1.5):
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.mean = np.asarray(data["mean"], dtype=np.float32); self.std = np.asarray(data["std"], dtype=np.float32)
        self.residual_std = float(data.get("residual_std", residual_std)); self.model = TemporalConvNet(int(data.get("channels", 16)))
        self.model.load_state_dict(data["state_dict"]); self.model.eval()

    def predict(self, window: np.ndarray, timestamp: float) -> VelocityPrediction:
        normalized = (window - self.mean) / np.maximum(self.std, 1e-5)
        with torch.no_grad(): velocity = float(self.model(torch.tensor(normalized[None], dtype=torch.float32)).item())
        quality = motion_quality(normalized); anomaly = float(np.mean(np.abs(normalized)) / 3)
        uncertainty = self.residual_std * (1.0 + anomaly) / max(quality, .1)
        confidence = float(np.clip(np.exp(-uncertainty / 3.0), 0.01, .99))
        return VelocityPrediction(timestamp, max(0.0, velocity), uncertainty, confidence)
