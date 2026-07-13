import os
import pickle
import numpy as np
import torch
import joblib
import sys

sys.path.insert(0, "/home/dyros/mcy_ws")
from tactile_anomaly.model import GRUAE
from dual_channel_detector import FramewiseDeltaDetector

def main():
    data_dir = "/home/dyros/mcy_ws/transport_new"
    filename = "transport_20260527_211214_914393_failure_raw.npy"
    path = f"{data_dir}/{filename}"
    
    model_path = os.path.join(data_dir, "transport_gru_ae_raw.pt")
    scaler_path = os.path.join(data_dir, "transport_scaler_raw.pkl")
    thresh_a_path = os.path.join(data_dir, "transport_threshold_raw.npy")
    cfg_path = os.path.join(data_dir, "transport_infer_cfg_raw.pkl")
    
    with open(cfg_path, "rb") as f:
        cfg = pickle.load(f)
    scaler = joblib.load(scaler_path)
    threshold_a = float(np.load(thresh_a_path)[0])
    
    seq = np.load(path).astype(np.float32)
    T = len(seq)
    
    model = GRUAE(
        input_dim=scaler.mean_.shape[0],
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        seq_len=50,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    
    mrls = [FramewiseDeltaDetector._compute_mrl(frame) for frame in seq]
    norms = [np.linalg.norm(frame) for frame in seq]
    
    print(f"File: {filename}")
    print(f"Length: {T} frames")
    print(f"Threshold A: {threshold_a:.6f}")
    
    print(f"\nWindow-by-Window Reconstruction Errors (stride=10):")
    with torch.no_grad():
        for start in range(0, T - 50 + 1, 10):
            if start < 30: # warmup
                continue
            window = seq[start:start + 50]
            x = scaler.transform(window).astype(np.float32)
            x_t = torch.tensor(x).unsqueeze(0).to(device)
            score = model.reconstruction_error(x_t, 0.0).item()
            
            # Print window range, score, and whether it exceeds threshold
            status = "EXCEEDS!" if score > threshold_a else "normal"
            print(f"  Window {start}:{start+50} | Score: {score:.6f} | {status}")
            
    # Print MRL/Norm of the last 40 frames
    print("\nLast 40 MRLs:")
    print([round(float(m), 3) for m in mrls[-40:]])
    print("Last 40 Norms:")
    print([round(float(n), 3) for n in norms[-40:]])

if __name__ == "__main__":
    main()
