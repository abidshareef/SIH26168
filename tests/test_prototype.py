from pathlib import Path
import numpy as np
from backend.preprocessing import create_windows, ordered_rows
from backend.synthetic import generate_dataset
from backend.train import train
from backend.models import TCNVelocityModel
from backend.replay import run_replay
from backend.navigation import TrustEngine
import backend.api as api

def test_order_and_windows():
    assert [r['timestamp'] for r in ordered_rows([{'timestamp':2},{'timestamp':1},{'timestamp':2,'x':1}])] == [1,2]
    x,y=create_windows(np.zeros((5,6)),np.arange(5),3); assert x.shape == (3,3,6) and y.tolist()==[2,3,4]

def test_trust_covariance_increases_when_confidence_falls():
    engine=TrustEngine(); assert engine.covariance('ai',.2,1) > engine.covariance('ai',.9,1)

def test_synthetic_end_to_end(tmp_path: Path):
    files=generate_dataset(tmp_path/'data',trajectories=3,seconds=12); ckpt=tmp_path/'model.pt'
    result=train(tmp_path/'data',ckpt,epochs=2,window=10); assert result['validation_mae'] >= 0
    model=TCNVelocityModel(ckpt); p=model.predict(np.zeros((10,6),np.float32),1.0); assert 0 <= p.confidence <= 1 and p.uncertainty > 0
    replay=run_replay(files['demo'],3,4,ckpt); modes={x['mode'] for x in replay['logs']}; assert 'DEAD_RECKONING' in modes and replay['position_rmse_m'] >= 0

def test_demo_api_state_transitions():
    assert api.health()['status'] == 'ok'
    assert api.reset_navigation()['state']['mode'] == 'GNSS'
    assert api.simulation_outage()['mode'] == 'DEAD_RECKONING'
    assert api.simulation_recovery()['mode'] == 'RECOVERY'
    assert api.simulation_motion_anomaly()['trust']['ai_confidence'] <= .30
