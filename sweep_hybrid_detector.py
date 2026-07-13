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
    
    # Let's test a range of low thresholds (from 97% to 99%) and high thresholds (from 0.6 to 1.0)
    # We want to see if we can find a setting with low FPR and high TPR
    
    def run_hybrid_eval(thr_pct, thr_high, consec_A, mrl_thresh):
        threshold_a = np.percentile(succ_win_errors, thr_pct)
        
        def is_anomaly(filepath):
            arr = np.load(filepath).astype(np.float32)
            T = len(arr)
            
            # Rule 1: Miss Guard Rail
            max_mrl = arr[:, 0].max()
            if max_mrl < mrl_thresh:
                return True, "Miss Guard Rail"
                
            # Rule 2: GRU-AE sliding window
            if T >= seq_len:
                consec = 0
                for start in range(0, T - seq_len + 1, 10):
                    window = arr[start:start + seq_len].copy()
                    if hybrid:
                        window = build_hybrid_window(window)
                    x = scaler.transform(window).astype(np.float32)
                    x = apply_relative_transform(x, relative_mode, n_rel_dims)
                    x_t = torch.tensor(x).unsqueeze(0)
                    with torch.no_grad():
                        score = model.reconstruction_error(x_t, weight_alpha).item()
                    
                    # Rule 2.1: Super high error cutoff (immediate trigger)
                    if score > thr_high:
                        return True, f"High Error Cutoff ({score:.4f} > {thr_high:.4f})"
                    
                    # Rule 2.2: Normal threshold with consecutive count
                    if score > threshold_a:
                        consec += 1
                        if consec >= consec_A:
                            return True, f"GRU-AE ({score:.4f} > {threshold_a:.4f})"
                    else:
                        consec = 0
            return False, "Normal"

        succ_anoms = 0
        succ_reasons = {}
        for f in succ_files:
            anom, reason = is_anomaly(f)
            if anom:
                succ_anoms += 1
                succ_reasons[os.path.basename(f)] = reason
                
        fail_anoms = 0
        fail_reasons = {}
        for f in fail_files:
            anom, reason = is_anomaly(f)
            if anom:
                fail_anoms += 1
                fail_reasons[os.path.basename(f)] = reason
            else:
                fail_reasons[os.path.basename(f)] = "Missed"
                
        return succ_anoms, len(succ_files), fail_anoms, len(fail_files), threshold_a, succ_reasons, fail_reasons

    print("="*100)
    print(" [HYBRID DETECTOR OPTIMIZATION SWEEP]")
    print("="*100)
    
    # Sweep configurations
    best_f1 = 0
    best_cfg = None
    
    for pct in [95, 96, 97, 98, 98.5, 99]:
        for thr_h in [0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
            for consec in [2]:
                for mrl_t in [0.3]:
                    sa, st, fa, ft, th_a, succ_reasons, fail_reasons = run_hybrid_eval(pct, thr_h, consec, mrl_t)
                    fpr = (sa / st) * 100
                    tpr = (fa / ft) * 100
                    
                    # Calculate F1 Score
                    precision = fa / (fa + sa) if (fa + sa) > 0 else 0
                    recall = fa / ft
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    print(f"pct={pct}% (th_low={th_a:.4f}), th_high={thr_h:.1f} | Success FPR: {sa}/{st} ({fpr:.1f}%) | Failure TPR: {fa}/{ft} ({tpr:.1f}%) | F1: {f1:.4f}")
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_cfg = (pct, th_a, thr_h, sa, st, fa, ft, succ_reasons, fail_reasons)
                        
    print("\n" + "="*100)
    print(" [BEST CONFIGURATION FOUND]")
    print("="*100)
    pct, th_a, thr_h, sa, st, fa, ft, succ_reasons, fail_reasons = best_cfg
    print(f"Config: threshold_pct={pct}% (th_low={th_a:.4f}), th_high={thr_h:.1f}, consec_A=2, MRL_thresh=0.3")
    print(f"  - Success False Positives: {sa}/{st} ({(sa/st*100):.1f}%)")
    print(f"  - Failure Detection (TPR): {fa}/{ft} ({(fa/ft*100):.1f}%)")
    print("  - Detailed Success False Alarms:")
    for fname, status in succ_reasons.items():
        print(f"    * {fname}: {status}")
    print("  - Detailed Failure Detections:")
    for fname, status in fail_reasons.items():
        print(f"    * {fname}: {status}")

if __name__ == "__main__":
    main()
