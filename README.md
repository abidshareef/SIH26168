# SIH26168 navigation prototype

Offline, deterministic proof-of-architecture for confidence-aware vehicle dead reckoning. All generated data and metrics are **synthetic**; they do not demonstrate real-vehicle performance.

## Quick start

```powershell
python -m backend.demo
```

## IO-VNBD real-data baseline

The repository does not contain benchmark payloads. Clone the official upstream with Git LFS and pull selected synchronized route pairs, then train with vehicle odometry used only as a label:

```powershell
git clone https://github.com/onyekpeu/IO-VNBD.git IO-VNBD-upstream
git -C IO-VNBD-upstream lfs pull --include="Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/*.csv,Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S2/*.csv"
python -m backend.train --iovnbd-root "IO-VNBD-upstream/Synchronised V abd S datasets/Categorised IOVNB Dataset" --epochs 20
```

The IO-VNBD adapter uses only phone accelerometer/gyroscope fields as inputs and vehicle speed as supervision. It holds out a complete route for validation. Initial short-run results are a baseline only; do not claim real-world readiness until multi-route, cross-driver evaluation and outage metrics are completed.

## Run the integrated UI

```powershell
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/). The UI is served by the backend itself, so no second server, CORS setting, or internet connection is required.

This command creates synthetic data and, if needed, a small TCN checkpoint, then runs a 30–60 second GNSS outage/recovery replay.

```powershell
python -m backend.generate_data
python -m backend.train
python -m backend.predict
python -m backend.replay --data data/synthetic/demo_outage.csv --outage-start 30 --outage-duration 30
python -m uvicorn backend.api:app --reload
python -m pytest -q
```

The core is deliberately independent from FastAPI, so its model and fusion interfaces can be replaced by Android/on-device implementations later.

## Architecture

`synthetic CSV → IMU windows → lightweight TCN (velocity + residual-based uncertainty) → trust engine → navigation fusion → replay/API state`

The fusion engine predicts with IMU, then applies AI velocity and GNSS measurements with covariances derived from confidence. GNSS return is innovation-gated and its trust ramps over ten updates. NHC and map matching are represented as replaceable prototype interfaces.

## API examples

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod -Method Post 'http://127.0.0.1:8000/replay/run?outage_start=30&outage_duration=30'
Invoke-RestMethod -Method Post 'http://127.0.0.1:8000/navigation/reset?latitude=17.385&longitude=78.4867'
```

`POST /predict/velocity` and `POST /navigation/update` accept JSON windows shaped `[20][6]` in the order `ax, ay, az, gx, gy, gz`. Interactive OpenAPI documentation is at `/docs`.

## Scope and handoff

Real: deterministic sensor imperfections, CSV training/inference, compact executable PyTorch TCN, uncertainty/confidence, dynamic covariance, outage/recovery replay, metrics, and local development API.

Placeholders: synthetic-only data, rule-based GNSS quality, simple NHC/map contracts, fixed phone alignment, and a simplified Kalman-style navigation state. Before any real deployment, replace synthetic data with IO-VNBD/collected data, calibrate uncertainty, implement production ESKF/InEKF and alignment/map matching, validate across phones/vehicles, then export/deploy the model on Android.
