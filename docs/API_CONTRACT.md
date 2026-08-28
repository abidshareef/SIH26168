# API contract — SIH26168 synthetic prototype

Base URL is configurable by adding `?api=http://HOST:8000` to the frontend URL (default `http://127.0.0.1:8000`). All navigation quantities come from the backend; the UI only validates and renders them.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Backend/model status |
| GET | `/navigation/state` | Current authoritative `NavigationState` |
| POST | `/navigation/reset` | Restore deterministic GNSS demo state |
| POST | `/simulation/outage` | Run actual replay to an outage state |
| POST | `/simulation/recovery` | Run actual replay to gated recovery state |
| POST | `/simulation/motion-anomaly` | Backend injects AI-confidence anomaly |
| POST | `/replay/run` | Execute complete synthetic replay |

`NavigationState` is a flat JSON object: `timestamp` seconds, `latitude`/`longitude` degrees, `velocity` m/s, `heading` degrees `[0,360)`, `position_uncertainty` metres, `mode` (`GNSS`, `GNSS_DEGRADED`, `DEAD_RECKONING`, `RECOVERY`), and `trust` containing six `[0,1]` confidences named `gnss_confidence`, `ai_confidence`, `imu_confidence`, `nhc_confidence`, `map_confidence`, and `mount_confidence`.

Errors use normal HTTP status codes. The UI retains its last valid state and displays an explicit disconnected/error status on malformed values or request failures. This local API and all demo values are synthetic-only.
