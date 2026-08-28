from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .io import read_csv
from .models import TemporalConvNet
from .preprocessing import FEATURE_COLUMNS, create_windows
from .synthetic import generate_dataset


def arrays(path: Path, window: int, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    rows = read_csv(path); raw = np.asarray([[r[k] for k in FEATURE_COLUMNS] for r in rows], np.float32); labels = np.asarray([r["reference_velocity"] for r in rows], np.float32)
    mean = raw.mean(0) if mean is None else mean; std = raw.std(0) if std is None else std
    return *create_windows((raw-mean)/np.maximum(std,1e-5), labels, window), mean, std


def train(data_dir: Path = Path("data/synthetic"), checkpoint: Path = Path("models/tcn_velocity.pt"), epochs: int = 8, window: int = 20) -> dict:
    if not (data_dir / "train.csv").exists(): generate_dataset(data_dir)
    x, y, mean, std = arrays(data_dir / "train.csv", window); xv, yv, _, _ = arrays(data_dir / "validation.csv", window, mean, std)
    torch.manual_seed(7); model = TemporalConvNet(); opt = torch.optim.Adam(model.parameters(), lr=.003); loss_fn = nn.SmoothL1Loss()
    loader = DataLoader(TensorDataset(torch.tensor(x), torch.tensor(y)), batch_size=128, shuffle=True)
    for _ in range(epochs):
        model.train()
        for bx, by in loader: opt.zero_grad(); loss_fn(model(bx),by).backward(); opt.step()
    model.eval()
    with torch.no_grad(): residual = model(torch.tensor(xv)).numpy()-yv; val_loss=float(np.mean(np.abs(residual)))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # Primitive lists keep checkpoints compatible with PyTorch's safe weights-only loader.
    torch.save({"state_dict":model.state_dict(),"mean":mean.tolist(),"std":std.tolist(),"residual_std":float(np.std(residual)+.15),"channels":16},checkpoint)
    return {"validation_mae":val_loss,"checkpoint":str(checkpoint),"windows":len(x)}

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--epochs",type=int,default=8); args=p.parse_args(); result=train(epochs=args.epochs)
    print(f"SYNTHETIC TCN trained: {result['windows']} windows | validation MAE: {result['validation_mae']:.3f} m/s\ncheckpoint: {result['checkpoint']}")
