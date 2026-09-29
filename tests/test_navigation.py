import math

from backend.constraints import NonHolonomicConstraint
from backend.contracts import NavigationState, VelocityPrediction
from backend.navigation import NavigationEngine


def test_nhc_rejects_negative_forward_velocity():
    result = NonHolonomicConstraint().apply(-3.0)
    assert result.velocity == 0.0
    assert result.confidence == 0.95


def test_gnss_outage_switches_to_dead_reckoning():
    initial = NavigationState(0.0, 17.4, 78.4, 540.0, 10.0, 90.0, 3.0, 4.0)
    engine = NavigationEngine(initial)
    ai = VelocityPrediction(0.1, 10.0, 0.5, 0.9)
    state = engine.update(0.1, 0.0, 0.0, ai, None)
    assert state.mode.value == "DEAD_RECKONING"
    assert state.velocity >= 0.0
    assert math.isfinite(state.position_uncertainty)


def test_gnss_recovery_is_ramped():
    initial = NavigationState(0.0, 17.4, 78.4, 540.0, 10.0, 90.0, 3.0, 4.0)
    engine = NavigationEngine(initial)
    ai = VelocityPrediction(0.1, 10.0, 0.5, 0.9)
    engine.update(0.1, 0.0, 0.0, ai, None)
    from backend.contracts import GnssSample
    gnss = GnssSample(0.2, 17.4, 78.4, 540.0, 10.0, 90.0, 3.0)
    state = engine.update(0.2, 0.0, 0.0, ai, gnss)
    assert state.mode.value in {"RECOVERY", "GNSS"}
    assert 0.0 <= state.trust.gnss_confidence <= 1.0
