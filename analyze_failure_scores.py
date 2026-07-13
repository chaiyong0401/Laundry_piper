import os
import sys
import glob
import pickle
import joblib
import numpy as np
import torch

WS = "/home/dyros/mcy_ws"
sys.path.insert(0, WS)
from train_gru_ae import GRUAE, apply_relative_transform, build_hybrid_window

def main():
    data_dir = "/home/dyros/mcy_ws/grasp_new"
    
    # ── Load Model and Scaler
    model_path = os.path.join(data_dir, "grasp_gru_ae.pt")
    scaler_path = os.path.join(data_dir, "grasp_scaler.pkl")
    cfg_path = os.path.join(data_dir, "grasp_infer_cfg.pkl")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    scaler = joblib.load(scaler_path)
    with open(cfg_path, "rb") as f:
        cfg = pickle.load(f)
        
    seq_len = cfg.get("seq_len", 50)
    relative_mode = cfg.get("relative", "none")
    weight_alpha = cfg.get("weight_alpha", 0.0)
    n_rel_dims = cfg.get("n_rel_dims", None)
    hybrid = cfg.get("hybrid", False)
    
    # Reconstruct architecture
    state = torch.load(model_path, map_location="cpu")
    input_dim = state["encoder.gru.weight_ih_l0"].shape[1]
    hidden_dim = state["encoder.gru.weight_ih_l0"].shape[0] // 3
    num_layers = sum(1 for k in state if k.startswith("encoder.gru.weight_ih_l"))
    
    model = GRUAE(input_dim, hidden_dim, num_layers, dropout=0.2, seq_len=seq_len)
    model.load_state_dict(state)
    model.eval()
    
    # ── Load Failure Files
    pattern = os.path.join(data_dir, "grasp_*_failure.npy")
    failure_files = sorted([f for f in glob.glob(pattern) if not f.endswith("_raw.npy")])
    
    print("="*80)
    print(f" [Failure Analysis] Total Files: {len(failure_files)}")
    print("="*80)
    
    # Threshold A for 95% and 99%
    # The user has these thresholds:
    thr_95 = 0.127375
    thr_99 = 0.403853
    
    for f in failure_files:
        name = os.path.basename(f)
        arr = np.load(f).astype(np.float32)
        T = len(arr)
        
        # Calculate features info
        # Columns: 0: MRL, 1: Norm, 2: Max Norm
        max_mrl = arr[:, 0].max()
        max_norm = arr[:, 1].max()
        max_max_norm = arr[:, 2].max()
        
        # Calculate sliding window reconstruction errors
        win_scores = []
        if T >= seq_len:
            for start in range(0, T - seq_len + 1, 10):
                window = arr[start:start + seq_len].copy()
                if hybrid:
                    window = build_hybrid_window(window)
                x = scaler.transform(window).astype(np.float32)
                x = apply_relative_transform(x, relative_mode, n_rel_dims)
                x_t = torch.tensor(x).unsqueeze(0)
                with torch.no_grad():
                    score = model.reconstruction_error(x_t, weight_alpha).item()
                win_scores.append(score)
        
        # Evaluate consec_A=2
        triggered_95 = False
        triggered_99 = False
        consec_95 = 0
        consec_99 = 0
        
        for s in win_scores:
            if s > thr_95:
                consec_95 += 1
                if consec_95 >= 2:
                    triggered_95 = True
            else:
                consec_95 = 0
                
            if s > thr_99:
                consec_99 += 1
                if consec_99 >= 2:
                    triggered_99 = True
            else:
                consec_99 = 0
                
        # Status
        status = "Detected(95 & 99)" if triggered_99 else ("Detected(95 Only)" if triggered_95 else "Missed(Both)")
        max_err = max(win_scores) if win_scores else 0.0
        
        print(f"File: {name} (T={T})")
        print(f"  - Max MRL: {max_mrl:.4f}, Max Norm: {max_norm:.4f}, Max MaxNorm: {max_max_norm:.4f}")
        print(f"  - Sliding Win Scores (n={len(win_scores)}): {['{:.4f}'.format(x) for x in win_scores]}")
        print(f"  - Max Error: {max_err:.4f} | Status: {status}")
        print("-"*80)

if __name__ == "__main__":
    main()
