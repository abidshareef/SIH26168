"""Vehicle and road constraint interfaces used by the NAV-AION prototype.

The prototype state is currently scalar in velocity, so the NHC implementation
can only enforce the forward-motion assumption. Full 3-D body-frame lateral
velocity constraints and production map matching remain production work.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import NavigationState


@dataclass
class ConstraintResult:
    velocity: float
    confidence: float


class NonHolonomicConstraint:
    """Prototype NHC: enforce forward-only vehicle speed in the scalar state."""

    def apply(self, velocity: float) -> ConstraintResult:
        return ConstraintResult(velocity=max(0.0, float(velocity)), confidence=0.95)


class MapConstraint:
    """Map constraint interface; production road projection is intentionally replaceable."""

    def confidence(self, _state: NavigationState) -> float:
        return 0.65

    def project(self, state: NavigationState) -> NavigationState:
        return state


class NullMapConstraint(MapConstraint):
    """No-op map constraint used until a real road graph/map matcher is connected."""

    pass
