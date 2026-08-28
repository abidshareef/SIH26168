from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


class NavigationMode(str, Enum):
    GNSS = "GNSS"
    GNSS_DEGRADED = "GNSS_DEGRADED"
    DEAD_RECKONING = "DEAD_RECKONING"
    RECOVERY = "RECOVERY"


@dataclass
class SensorSample:
    timestamp: float
    ax: float; ay: float; az: float
    gx: float; gy: float; gz: float
    mx: Optional[float] = None; my: Optional[float] = None; mz: Optional[float] = None
    barometer: Optional[float] = None


@dataclass
class GnssSample:
    timestamp: float
    latitude: float; longitude: float; altitude: float
    speed: float; heading: float; accuracy: float
    satellite_count: Optional[int] = None; hdop: Optional[float] = None; vdop: Optional[float] = None


@dataclass
class ReferenceState:
    timestamp: float
    latitude: float; longitude: float; altitude: float
    velocity: float; heading: float


@dataclass
class VelocityPrediction:
    timestamp: float
    forward_velocity: float
    uncertainty: float
    confidence: float


@dataclass
class TrustState:
    gnss_confidence: float = 0.0
    ai_confidence: float = 0.0
    imu_confidence: float = 0.8
    nhc_confidence: float = 0.95
    map_confidence: float = 0.65
    mount_confidence: float = 0.95


@dataclass
class NavigationState:
    timestamp: float
    latitude: float; longitude: float; altitude: float
    velocity: float; heading: float
    position_uncertainty: float
    velocity_variance: float
    trust: TrustState = field(default_factory=TrustState)
    mode: NavigationMode = NavigationMode.GNSS

    def to_dict(self) -> dict:
        value = asdict(self)
        value["mode"] = self.mode.value
        return value


@dataclass
class AlignmentState:
    heading_offset_deg: float = 0.0
    mount_confidence: float = 0.95
