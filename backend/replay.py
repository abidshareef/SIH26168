from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .contracts import GnssSample, NavigationState
from .io import read_csv
from .models import TCNVelocityModel
from .navigation import NavigationEngine
from .train import train


def run_replay(data: str | Path, outage_start: float | None = None, outage_duration: float = 0, checkpoint: Path = Path("models/tcn_velocity.pt"), log_path: Path | None = None) -> dict:
    if not checkpoint.exists(): train(checkpoint=checkpoint)
    rows=read_csv(data); model=TCNVelocityModel(checkpoint); first=rows[0]
    initial=NavigationState(first["timestamp"],first["reference_latitude"],first["reference_longitude"],540,0,first["reference_heading"],3.0,4.0)
    engine=NavigationEngine(initial); logs=[]; window=20
    for i,row in enumerate(rows):
        if i < window-1: continue
        w=np.asarray([[r[k] for k in ("ax","ay","az","gx","gy","gz")] for r in rows[i-window+1:i+1]],np.float32); prediction=model.predict(w,row["timestamp"])
        forced=outage_start is not None and outage_start <= row["timestamp"] < outage_start+outage_duration
        gnss=None if forced or not int(row["gnss_available"]) else GnssSample(row["timestamp"],row["latitude"],row["longitude"],row["altitude"],row["speed"],row["heading"],row["accuracy"])
        state=engine.update(row["timestamp"],row["ax"],row["gz"],prediction,gnss)
        error=np.hypot((state.latitude-row["reference_latitude"])*111111,(state.longitude-row["reference_longitude"])*111111*np.cos(np.deg2rad(state.latitude)))
        logs.append({**state.to_dict(),"reference_velocity":row["reference_velocity"],"position_error_m":float(error),"ai_uncertainty":prediction.uncertainty})
    if log_path: log_path.parent.mkdir(parents=True,exist_ok=True); log_path.write_text(json.dumps(logs,indent=2))
    outage=[r for r in logs if r["mode"]=="DEAD_RECKONING"]
    return {"logs":logs,"position_rmse_m":float(np.sqrt(np.mean([r["position_error_m"]**2 for r in logs]))),"velocity_mae_mps":float(np.mean([abs(r["velocity"]-r["reference_velocity"]) for r in logs])),"max_outage_drift_m":max((r["position_error_m"] for r in outage),default=0.0),"final_state":logs[-1]}

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data",default="data/synthetic/demo_outage.csv"); p.add_argument("--outage-start",type=float,default=30); p.add_argument("--outage-duration",type=float,default=30); args=p.parse_args()
    r=run_replay(args.data,args.outage_start,args.outage_duration,log_path=Path("outputs/replay.json")); f=r["final_state"]
    print("SIH26168 NAVIGATION REPLAY — SYNTHETIC\n"+f"Position RMSE: {r['position_rmse_m']:.2f} m | max outage drift: {r['max_outage_drift_m']:.2f} m\nFinal mode: {f['mode']} | uncertainty: {f['position_uncertainty']:.2f} m\nReplay log: outputs/replay.json")
