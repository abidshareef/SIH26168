from __future__ import annotations

import hmac
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import Response

from .contracts import GnssSample, NavigationState
from .models import TCNVelocityModel
from .navigation import NavigationEngine
from .replay import run_replay
from .train import train

BASE_DIR = Path(__file__).resolve().parent.parent
API_KEY = os.getenv("NAV_AION_API_KEY", "").strip()
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "NAV_AION_ALLOWED_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000",
    ).split(",")
    if origin.strip()
]

app = FastAPI(
    title="NAV-AION navigation backend",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

_model = None
_engine: NavigationEngine | None = None
_simulation_state: dict | None = None

PUBLIC_PATHS = {"/", "/ui", "/ui/"}


def _authorized(request: Request) -> bool:
    if not API_KEY:
        return False
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(token, API_KEY)


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Protect every API route while keeping the local UI assets public."""
    if request.method == "OPTIONS" or request.url.path in PUBLIC_PATHS or request.url.path.startswith("/ui/"):
        response = await call_next(request)
    elif not _authorized(request):
        return JSONResponse({"detail": "Authentication required"}, status_code=401)
    else:
        response = await call_next(request)

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'self'"
    )
    return response


@app.get("/", include_in_schema=False)
def ui_home() -> FileResponse:
    return FileResponse(BASE_DIR / "frontend" / "index.html")


def model():
    global _model
    if _model is None:
        ckpt = Path("models/tcn_velocity.pt")
        if not ckpt.exists():
            train(checkpoint=ckpt)
        _model = TCNVelocityModel(ckpt)
    return _model


class VelocityRequest(BaseModel):
    timestamp: float
    window: list[list[float]] = Field(min_length=20, max_length=256)


class NavigationUpdateRequest(BaseModel):
    timestamp: float
    ax: float
    gz: float
    window: list[list[float]] = Field(min_length=20, max_length=256)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    altitude: float = Field(default=540.0, ge=-500, le=10000)
    speed: float | None = Field(default=None, ge=0, le=150)
    heading: float | None = Field(default=None, ge=0, lt=360)
    accuracy: float = Field(default=3.0, gt=0, le=1000)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "dataset": "synthetic-only",
        "model_loaded": Path("models/tcn_velocity.pt").exists(),
        "navigation_engine": True,
    }


def _demo_state(target_time: float, outage: bool = False) -> dict:
    replay = run_replay(
        "data/synthetic/demo_outage.csv",
        30 if outage else None,
        30 if outage else 0,
    )
    states = [item for item in replay["logs"] if item["timestamp"] <= target_time]
    if not states:
        raise RuntimeError("Demo replay produced no state at the requested timestamp")
    return states[-1]


@app.post("/predict/velocity")
def predict_velocity(request: VelocityRequest):
    value = model().predict(np.asarray(request.window, dtype=np.float32), request.timestamp)
    return value.__dict__


@app.post("/replay/run")
def replay_run(outage_start: float = 30, outage_duration: float = 30):
    if not (0 <= outage_start <= 86400 and 0 <= outage_duration <= 3600):
        return JSONResponse({"detail": "Invalid replay window"}, status_code=422)
    result = run_replay("data/synthetic/demo_outage.csv", outage_start, outage_duration)
    return {key: value for key, value in result.items() if key != "logs"}


@app.post("/navigation/reset")
def reset_navigation(
    latitude: float = 17.385,
    longitude: float = 78.4867,
    heading: float = 0,
):
    global _engine, _simulation_state
    _engine = NavigationEngine(
        NavigationState(0, latitude, longitude, 540, 0, heading, 3, 4)
    )
    _simulation_state = _demo_state(25)
    return {"status": "reset", "state": _simulation_state}


@app.get("/navigation/state")
def navigation_state():
    if _simulation_state is not None:
        return _simulation_state
    return {"status": "not initialized"} if _engine is None else _engine.state.to_dict()


@app.post("/navigation/update")
def navigation_update(request: NavigationUpdateRequest):
    global _engine
    if _engine is None:
        reset_navigation()
    prediction = model().predict(
        np.asarray(request.window, dtype=np.float32), request.timestamp
    )
    gnss = (
        None
        if request.latitude is None
        or request.longitude is None
        or request.speed is None
        or request.heading is None
        else GnssSample(
            request.timestamp,
            request.latitude,
            request.longitude,
            request.altitude,
            request.speed,
            request.heading,
            request.accuracy,
        )
    )
    return _engine.update(
        request.timestamp, request.ax, request.gz, prediction, gnss
    ).to_dict()


@app.post("/simulation/outage")
def simulation_outage():
    global _simulation_state
    _simulation_state = _demo_state(40, outage=True)
    return _simulation_state


@app.post("/simulation/recovery")
def simulation_recovery():
    global _simulation_state
    _simulation_state = _demo_state(60.4, outage=True)
    return _simulation_state


@app.post("/simulation/motion-anomaly")
def simulation_motion_anomaly():
    global _simulation_state
    if _simulation_state is None:
        _simulation_state = _demo_state(25)
    state = dict(_simulation_state)
    trust = dict(state["trust"])
    trust["ai_confidence"] = min(float(trust["ai_confidence"]), 0.30)
    state["trust"] = trust
    state["simulation_note"] = "Motion anomaly injected by backend; AI covariance increased."
    _simulation_state = state
    return _simulation_state


app.mount("/ui", StaticFiles(directory=BASE_DIR / "frontend", html=True), name="ui")
