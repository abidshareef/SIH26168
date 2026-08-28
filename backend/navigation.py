from __future__ import annotations

from dataclasses import replace
import math
import numpy as np

from .contracts import GnssSample, NavigationMode, NavigationState, TrustState, VelocityPrediction


METERS_PER_DEGREE = 111_111.0


class TrustEngine:
    """Maps confidence to update covariance; high confidence means high influence."""
    def covariance(self, source: str, confidence: float, base_variance: float) -> float:
        return base_variance / max(float(np.clip(confidence, 0.0, 1.0)) ** 2, 0.01)


class GnssQualityEstimator:
    def confidence(self, gnss: GnssSample | None, predicted_speed: float) -> float:
        if gnss is None: return 0.0
        accuracy_score = math.exp(-max(gnss.accuracy, 0) / 8)
        consistency = math.exp(-abs(gnss.speed - predicted_speed) / 12)
        return float(np.clip(accuracy_score * (.65 + .35 * consistency), 0.0, .99))


class SimpleMapMatcher:
    """Prototype contract: provides a cautious road constraint, not real map matching."""
    def confidence(self, _state: NavigationState) -> float: return 0.65


class NavigationEngine:
    def __init__(self, initial: NavigationState):
        self.state = initial; self.trust_engine = TrustEngine(); self.gnss_quality = GnssQualityEstimator(); self.map_matcher = SimpleMapMatcher(); self._recovery_steps = 0

    def _propagate(self, dt: float, accel_forward: float, yaw_rate: float) -> None:
        s = self.state; velocity = max(0.0, s.velocity + accel_forward * dt); heading = (s.heading + math.degrees(yaw_rate * dt)) % 360
        radians = math.radians(heading); north, east = velocity * math.cos(radians) * dt, velocity * math.sin(radians) * dt
        self.state = replace(s, latitude=s.latitude + north/METERS_PER_DEGREE, longitude=s.longitude + east/(METERS_PER_DEGREE*math.cos(math.radians(s.latitude))), velocity=velocity, heading=heading, position_uncertainty=math.sqrt(s.position_uncertainty**2 + (0.4 + .1*velocity)*dt), velocity_variance=s.velocity_variance + .08*dt)

    def _scalar_update(self, value: float, variance: float, observation: float, obs_variance: float) -> tuple[float, float]:
        gain = variance / (variance + obs_variance)
        return value + gain * (observation - value), (1 - gain) * variance

    def update(self, timestamp: float, accel_forward: float, yaw_rate: float, ai: VelocityPrediction, gnss: GnssSample | None, dt: float = .1) -> NavigationState:
        self._propagate(dt, accel_forward, yaw_rate); s = self.state
        gnss_c = self.gnss_quality.confidence(gnss, s.velocity)
        trust = TrustState(gnss_confidence=gnss_c, ai_confidence=ai.confidence, imu_confidence=.9, nhc_confidence=.95, map_confidence=self.map_matcher.confidence(s), mount_confidence=.95)
        # AI velocity update, with trust-derived covariance.
        ai_var = self.trust_engine.covariance("ai", ai.confidence * trust.mount_confidence, max(ai.uncertainty**2, .1))
        velocity, variance = self._scalar_update(s.velocity, s.velocity_variance, ai.forward_velocity, ai_var)
        s = replace(s, velocity=velocity, velocity_variance=variance, trust=trust)
        if gnss is None:
            mode = NavigationMode.DEAD_RECKONING; self._recovery_steps = 0
        else:
            # Innovation gate prevents a blind GNSS snap; recovery increases trust over 1 s.
            distance = math.hypot((gnss.latitude-s.latitude)*METERS_PER_DEGREE, (gnss.longitude-s.longitude)*METERS_PER_DEGREE*math.cos(math.radians(s.latitude)))
            # A generous but finite first-return gate permits a controlled recovery after
            # a planned outage while still rejecting gross jumps (for example, bad fixes).
            gate = max(80.0, 3*s.position_uncertainty + gnss.accuracy*3)
            if distance <= gate:
                self._recovery_steps += 1; ramp = min(1.0, self._recovery_steps / 10)
                c = gnss_c * ramp; pos_var = self.trust_engine.covariance("gnss", c, max(gnss.accuracy**2, 1))
                gain = s.position_uncertainty**2 / (s.position_uncertainty**2 + pos_var)
                s = replace(s, latitude=s.latitude+gain*(gnss.latitude-s.latitude), longitude=s.longitude+gain*(gnss.longitude-s.longitude), position_uncertainty=math.sqrt((1-gain)*s.position_uncertainty**2), trust=replace(trust, gnss_confidence=c))
                mode = NavigationMode.GNSS if ramp >= 1 else NavigationMode.RECOVERY
            else: mode = NavigationMode.GNSS_DEGRADED
        self.state = replace(s, timestamp=timestamp, mode=mode)
        return self.state
