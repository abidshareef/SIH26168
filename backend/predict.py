from pathlib import Path
import numpy as np
from .io import read_csv
from .models import TCNVelocityModel
from .train import train

if __name__ == "__main__":
    ckpt=Path("models/tcn_velocity.pt")
    if not ckpt.exists(): train(checkpoint=ckpt)
    rows=read_csv("data/synthetic/test.csv"); row=rows[19]; win=np.asarray([[r[k] for k in ("ax","ay","az","gx","gy","gz")] for r in rows[:20]],np.float32); p=TCNVelocityModel(ckpt).predict(win,row["timestamp"])
    print(f"SYNTHETIC prediction\nPredicted velocity: {p.forward_velocity:.2f} m/s\nGround truth: {row['reference_velocity']:.2f} m/s\nUncertainty: {p.uncertainty:.2f} m/s\nConfidence: {p.confidence:.2f}")
