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
    
    # Load all files
    succ_files = sorted([f for f in glob.glob(os.path.join(data_dir, "grasp_*_success.npy")) if not f.endswith("_raw.npy")])
    fail_files = sorted([f for f in glob.glob(os.path.join(data_dir, "grasp_*_failure.npy")) if not f.endswith("_raw.npy")])
    
    # Load success window errors to compute exact thresholds
    from train_gru_ae import load_sequences_windowed
    success_windows = load_sequences_windowed(data_dir, 'grasp', 'success', False, seq_len, 10)
    succ_win_errors = []
    for seq in success_windows:
        x = scaler.transform(seq).astype(np.float32)
        x = apply_relative_transform(x, relative_mode, n_rel_dims)
        x_t = torch.tensor(x).unsqueeze(0)
        with torch.no_grad():
            err = model.reconstruction_error(x_t, weight_alpha).item()
        succ_win_errors.append(err)
    succ_win_errors = np.array(succ_win_errors)
    
    th_95 = np.percentile(succ_win_errors, 95)
    th_98 = np.percentile(succ_win_errors, 98)
    
    def evaluate_file(filepath):
        arr = np.load(filepath).astype(np.float32)
        T = len(arr)
        
        # Guard rail check
        max_mrl = arr[:, 0].max()
        max_norm = arr[:, 1].max()
        max_max_norm = arr[:, 2].max()
        
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
                
        # consec_A = 2 check
        def is_triggered(thr):
            consec = 0
            for s in win_scores:
                if s > thr:
                    consec += 1
                    if consec >= 2:
                        return True
                else:
                    consec = 0
            return False
            
        triggered_95 = (max_mrl < 0.3) or is_triggered(th_95)
        triggered_98 = (max_mrl < 0.3) or is_triggered(th_98)
        
        return {
            "name": os.path.basename(filepath),
            "T": T,
            "max_mrl": max_mrl,
            "max_norm": max_norm,
            "max_max_norm": max_max_norm,
            "win_scores": win_scores,
            "triggered_95": triggered_95,
            "triggered_98": triggered_98
        }

    print("="*80)
    print(" [DIAGNOSTIC REPORT]")
    print(f" Threshold 95%: {th_95:.4f}")
    print(f" Threshold 98%: {th_98:.4f}")
    print("="*80)
    
    # Analyze successes
    succ_results = [evaluate_file(f) for f in succ_files]
    print("\n--- [FALSE ALARMS ON SUCCESS (95%)] ---")
    fa_95 = [r for r in succ_results if r["triggered_95"]]
    print(f"Total False Alarms at 95%: {len(fa_95)}")
    for r in fa_95:
        tag_98 = "ALSO FA at 98%" if r["triggered_98"] else "Normal at 98%"
        print(f"File: {r['name']} (T={r['T']}) -> {tag_98}")
        print(f"  Max MRL: {r['max_mrl']:.4f}, Max Norm: {r['max_norm']:.4f}, Max MaxNorm: {r['max_max_norm']:.4f}")
        print(f"  Window Scores: {['{:.4f}'.format(x) for x in r['win_scores']]}")
        
    print("\n--- [FALSE ALARMS ON SUCCESS (98%)] ---")
    fa_98 = [r for r in succ_results if r["triggered_98"]]
    print(f"Total False Alarms at 98%: {len(fa_98)}")
    for r in fa_98:
        print(f"File: {r['name']} (T={r['T']})")
        print(f"  Max MRL: {r['max_mrl']:.4f}, Max Norm: {r['max_norm']:.4f}, Max MaxNorm: {r['max_max_norm']:.4f}")
        print(f"  Window Scores: {['{:.4f}'.format(x) for x in r['win_scores']]}")

    # Analyze failures
    fail_results = [evaluate_file(f) for f in fail_files]
    print("\n--- [MISSED FAILURES (98%)] ---")
    missed_98 = [r for r in fail_results if not r["triggered_98"]]
    print(f"Total Missed Failures at 98%: {len(missed_98)}")
    for r in missed_98:
        print(f"File: {r['name']} (T={r['T']})")
        print(f"  Max MRL: {r['max_mrl']:.4f}, Max Norm: {r['max_norm']:.4f}, Max MaxNorm: {r['max_max_norm']:.4f}")
        print(f"  Window Scores: {['{:.4f}'.format(x) for x in r['win_scores']]}")

if __name__ == "__main__":
    main()
