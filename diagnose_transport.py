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
    data_dir = "/home/dyros/mcy_ws/transport_new"
    
    # 1. Load Model, Scaler, and Cfg
    model_path = os.path.join(data_dir, "transport_gru_ae.pt")
    scaler_path = os.path.join(data_dir, "transport_scaler.pkl")
    cfg_path = os.path.join(data_dir, "transport_infer_cfg.pkl")
    
    scaler = joblib.load(scaler_path)
    with open(cfg_path, "rb") as f:
        cfg = pickle.load(f)
        
    seq_len = cfg.get("seq_len", 50)
    relative_mode = cfg.get("relative", "none")
    weight_alpha = cfg.get("weight_alpha", 0.0)
    n_rel_dims = cfg.get("n_rel_dims", None)
    hybrid = cfg.get("hybrid", False)
    
    print(f"Loaded config: relative={relative_mode}, hybrid={hybrid}, seq_len={seq_len}")
    
    # Reconstruct architecture
    state = torch.load(model_path, map_location="cpu")
    input_dim = state["encoder.gru.weight_ih_l0"].shape[1]
    hidden_dim = state["encoder.gru.weight_ih_l0"].shape[0] // 3
    num_layers = sum(1 for k in state if k.startswith("encoder.gru.weight_ih_l"))
    
    model = GRUAE(input_dim, hidden_dim, num_layers, dropout=0.2, seq_len=seq_len)
    model.load_state_dict(state)
    model.eval()
    
    # Load all files
    succ_files = sorted([f for f in glob.glob(os.path.join(data_dir, "transport_*_success.npy")) if not f.endswith("_raw.npy")])
    fail_files = sorted([f for f in glob.glob(os.path.join(data_dir, "transport_*_failure.npy")) if not f.endswith("_raw.npy")])
    
    # Load success window errors to compute exact thresholds
    from train_gru_ae import load_sequences_windowed
    success_windows = load_sequences_windowed(data_dir, 'transport', 'success', False, seq_len, 10)
    succ_win_errors = []
    for seq in success_windows:
        # If relative is window_mean, we must transform
        x = scaler.transform(seq).astype(np.float32)
        x = apply_relative_transform(x, relative_mode, n_rel_dims)
        x_t = torch.tensor(x).unsqueeze(0)
        with torch.no_grad():
            err = model.reconstruction_error(x_t, weight_alpha).item()
        succ_win_errors.append(err)
    succ_win_errors = np.array(succ_win_errors)
    
    th_95 = np.percentile(succ_win_errors, 95)
    th_98 = np.percentile(succ_win_errors, 98)
    th_99 = np.percentile(succ_win_errors, 99)
    
    def evaluate_file(filepath):
        arr = np.load(filepath).astype(np.float32)
        T = len(arr)
        
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
        
        max_score = max(win_scores) if win_scores else 0.0
        return {
            "name": os.path.basename(filepath),
            "max_score": max_score,
            "win_scores": win_scores
        }
        
    succ_results = [evaluate_file(f) for f in succ_files]
    fail_results = [evaluate_file(f) for f in fail_files]
    
    print("\n" + "="*80)
    print(" [DIAGNOSTIC REPORT FOR TRANSPORT]")
    print(f" Threshold 95%: {th_95:.4f}")
    print(f" Threshold 98%: {th_98:.4f}")
    print(f" Threshold 99%: {th_99:.4f}")
    print("="*80)
    
    print("\n--- [FAILURES MAX SCORES] ---")
    fail_scores = [r["max_score"] for r in fail_results]
    print(f"Min Fail Score: {min(fail_scores):.4f}, Mean: {np.mean(fail_scores):.4f}, Max: {max(fail_scores):.4f}")
    for r in sorted(fail_results, key=lambda x: x["max_score"]):
        print(f"File: {r['name']} -> Max Score: {r['max_score']:.4f}")
        
    print("\n--- [SUCCESS MAX SCORES (TOP 10 HIGHEST)] ---")
    succ_results_sorted = sorted(succ_results, key=lambda x: x["max_score"], reverse=True)
    for r in succ_results_sorted[:10]:
        print(f"File: {r['name']} -> Max Score: {r['max_score']:.4f}")

if __name__ == "__main__":
    main()
