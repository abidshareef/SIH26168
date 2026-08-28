from __future__ import annotations
from pathlib import Path
from dataclasses import replace
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
from .contracts import GnssSample, NavigationState
from .models import TCNVelocityModel
from .navigation import NavigationEngine
from .replay import run_replay
from .train import train

app=FastAPI(title="SIH26168 synthetic navigation backend")
app.add_middleware(CORSMiddleware, allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?", allow_methods=["*"], allow_headers=["*"])
_model=None
_engine: NavigationEngine | None = None
_simulation_state: dict | None = None

@app.get("/", include_in_schema=False)
def ui_home() -> FileResponse:
    """Explicit demo entrypoint; avoids relying on a directory redirect."""
    return FileResponse(Path(__file__).resolve().parent.parent / "frontend" / "index.html")
def model():
    global _model
    if _model is None:
        ckpt=Path("models/tcn_velocity.pt")
        if not ckpt.exists(): train(checkpoint=ckpt)
        _model=TCNVelocityModel(ckpt)
    return _model

class VelocityRequest(BaseModel):
    timestamp: float
    window: list[list[float]]

class NavigationUpdateRequest(BaseModel):
    timestamp: float
    ax: float
    gz: float
    window: list[list[float]]
    latitude: float | None = None
    longitude: float | None = None
    altitude: float = 540.0
    speed: float | None = None
    heading: float | None = None
    accuracy: float = 3.0

@app.get("/health")
def health(): return {"status":"ok","dataset":"synthetic-only","model_loaded":Path("models/tcn_velocity.pt").exists(),"navigation_engine":True}

def _demo_state(target_time: float, outage: bool = False) -> dict:
    """Run the real replay and retain its state at a deterministic demo moment."""
    replay = run_replay("data/synthetic/demo_outage.csv", 30 if outage else None, 30 if outage else 0)
    states = [item for item in replay["logs"] if item["timestamp"] <= target_time]
    return states[-1]

@app.post("/predict/velocity")
def predict_velocity(request: VelocityRequest):
    value=model().predict(np.asarray(request.window,dtype=np.float32),request.timestamp)
    return value.__dict__

@app.post("/replay/run")
def replay_run(outage_start: float=30, outage_duration: float=30):
    r=run_replay("data/synthetic/demo_outage.csv",outage_start,outage_duration)
    return {k:v for k,v in r.items() if k != "logs"}

@app.post("/navigation/reset")
def reset_navigation(latitude: float=17.385, longitude: float=78.4867, heading: float=0):
    global _engine, _simulation_state
    _engine=NavigationEngine(NavigationState(0,latitude,longitude,540,0,heading,3,4))
    _simulation_state=_demo_state(25)
    return {"status":"reset","state":_simulation_state}

@app.get("/navigation/state")
def navigation_state():
    if _simulation_state is not None: return _simulation_state
    return {"status":"not initialized"} if _engine is None else _engine.state.to_dict()

@app.post("/navigation/update")
def navigation_update(request: NavigationUpdateRequest):
    global _engine
    if _engine is None: reset_navigation()
    prediction=model().predict(np.asarray(request.window,dtype=np.float32),request.timestamp)
    gnss = None if request.latitude is None or request.longitude is None or request.speed is None or request.heading is None else GnssSample(request.timestamp,request.latitude,request.longitude,request.altitude,request.speed,request.heading,request.accuracy)
    return _engine.update(request.timestamp,request.ax,request.gz,prediction,gnss).to_dict()

@app.post("/simulation/outage")
def simulation_outage():
    global _simulation_state
    _simulation_state=_demo_state(40, outage=True)
    return _simulation_state

@app.post("/simulation/recovery")
def simulation_recovery():
    global _simulation_state
    _simulation_state=_demo_state(60.4, outage=True)
    return _simulation_state

@app.post("/simulation/motion-anomaly")
def simulation_motion_anomaly():
    global _simulation_state
    if _simulation_state is None: _simulation_state=_demo_state(25)
    state=dict(_simulation_state); trust=dict(state["trust"]); trust["ai_confidence"]=min(float(trust["ai_confidence"]), .30); state["trust"]=trust; state["simulation_note"]="Motion anomaly injected by backend; AI covariance increased."; _simulation_state=state
    return _simulation_state

# Same-origin hosting makes the local hackathon demo work with one server command.
app.mount("/ui", StaticFiles(directory=Path(__file__).resolve().parent.parent / "frontend", html=True), name="ui")
