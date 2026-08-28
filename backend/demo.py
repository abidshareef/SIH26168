from pathlib import Path
from .synthetic import generate_dataset
from .train import train
from .replay import run_replay

if __name__ == "__main__":
    files=generate_dataset(); ckpt=Path("models/tcn_velocity.pt")
    if not ckpt.exists(): train(checkpoint=ckpt)
    result=run_replay(files["demo"],30,30,ckpt,Path("outputs/demo_replay.json")); final=result["final_state"]
    print("="*50+"\nSIH26168 INTELLIGENT DEAD RECKONING\nSYNTHETIC PROTOTYPE\n"+"="*50)
    print(f"Position RMSE: {result['position_rmse_m']:.2f} m\nMax outage drift: {result['max_outage_drift_m']:.2f} m\nFinal mode: {final['mode']}\nFinal uncertainty: {final['position_uncertainty']:.2f} m\nGNSS trust: {final['trust']['gnss_confidence']:.2f}\nAI trust: {final['trust']['ai_confidence']:.2f}")
