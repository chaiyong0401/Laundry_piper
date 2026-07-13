import os
import glob
import pickle
import numpy as np
import torch
import joblib
import sys

# Add directory to import path
sys.path.insert(0, "/home/dyros/mcy_ws")
from tactile_anomaly.model import GRUAE
from dual_channel_detector import FramewiseDeltaDetector

def build_hybrid_window(w):
    norms = np.linalg.norm(w, axis=1, keepdims=True)
    return np.hstack([w, norms])

def apply_relative_transform(x, mode, n_rel_dims=None):
    if mode == "none":
        return x
    elif mode == "window_mean":
        if n_rel_dims is not None:
            x_rel = x[:, :n_rel_dims] - x[:, :n_rel_dims].mean(axis=0, keepdims=True)
            return np.hstack([x_rel, x[:, n_rel_dims:]])
        return x - x.mean(axis=0, keepdims=True)
    elif mode == "first_frame":
        if n_rel_dims is not None:
            x_rel = x[:, :n_rel_dims] - x[0, :n_rel_dims]
            return np.hstack([x_rel, x[:, n_rel_dims:]])
        return x - x[0]
    return x

def main():
    data_dir = "/home/dyros/mcy_ws/transport_new"
    prefix = "transport"
    
    # Load thresholds, model, scaler
    model_path = os.path.join(data_dir, f"transport_gru_ae_raw.pt")
    scaler_path = os.path.join(data_dir, f"transport_scaler_raw.pkl")
    thresh_a_path = os.path.join(data_dir, f"transport_threshold_raw.npy")
    thresh_b_path = os.path.join(data_dir, f"transport_threshold_b_raw.npy")
    cfg_path = os.path.join(data_dir, f"transport_infer_cfg_raw.pkl")
    
    with open(cfg_path, "rb") as f:
        cfg = pickle.load(f)
        
    scaler = joblib.load(scaler_path)
    threshold_a = float(np.load(thresh_a_path)[0])
    threshold_b = float(np.load(thresh_b_path)[0])
    
    input_dim = scaler.mean_.shape[0]
    seq_len = int(cfg.get("seq_len", 50))
    relative_mode = cfg.get("relative", "none")
    weight_alpha = float(cfg.get("weight_alpha", 0.0))
    hybrid = bool(cfg.get("hybrid", False))
    n_rel_dims = cfg.get("n_rel_dims", None)
    mrl_thresh = float(cfg.get("mrl_thresh", 0.2))
    cutoff_multiplier = float(cfg.get("cutoff_multiplier", 2.2))
    warmup = 30
    
    model = GRUAE(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        seq_len=seq_len,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    
    consec_a = 2
    consec_b = 5
    infer_stride = 10
    
    # Load all raw failure files
    pattern = os.path.join(data_dir, f"{prefix}_*_failure_raw.npy")
    files = sorted(glob.glob(pattern))
    
    print(f"Total failure files: {len(files)}")
    print(f"Using Threshold A: {threshold_a:.6f}, Threshold B: {threshold_b:.6f}")
    
    undetected_count = 0
    with torch.no_grad():
        for f in files:
            seq = np.load(f).astype(np.float32)
            T = len(seq)
            
            # Channel B
            det_b_obj = FramewiseDeltaDetector(threshold_b, smooth_window=3)
            b_consec = 0
            flag_b = False
            for t, frame in enumerate(seq):
                _, trig = det_b_obj.update(frame)
                if t < warmup:
                    b_consec = 0
                    continue
                if trig:
                    b_consec += 1
                    if b_consec >= consec_b:
                        flag_b = True
                        break
                else:
                    b_consec = 0
            
            # Channel A
            flag_a = False
            # Rule 1: Miss Guard Rail
            mrls = [FramewiseDeltaDetector._compute_mrl(frame) for frame in seq]
            max_mrl = max(mrls) if mrls else 0.0
            if max_mrl < mrl_thresh:
                flag_a = True
                
            if not flag_a and T >= seq_len:
                a_consec = 0
                max_score_seen = 0.0
                for start in range(0, T - seq_len + 1, infer_stride):
                    if start < warmup:
                        continue
                    window = seq[start:start + seq_len]
                    if hybrid:
                        window = build_hybrid_window(window)
                    x = apply_relative_transform(
                        scaler.transform(window).astype(np.float32),
                        relative_mode, n_rel_dims)
                    x_t = torch.tensor(x).unsqueeze(0).to(device)
                    score = model.reconstruction_error(x_t, weight_alpha).item()
                    max_score_seen = max(max_score_seen, score)
                    
                    if score > threshold_a * cutoff_multiplier:
                        flag_a = True
                        break
                    if score > threshold_a:
                        a_consec += 1
                        if a_consec >= consec_a:
                            flag_a = True
                            break
                    else:
                        a_consec = 0
            
            detected = flag_a or flag_b
            if not detected:
                undetected_count += 1
                basename = os.path.basename(f)
                print(f"\n[UNDETECTED] {basename}")
                print(f"  Length: {T} frames")
                print(f"  Max MRL: {max_mrl:.4f} (mrl_thresh={mrl_thresh})")
                if T >= seq_len:
                    print(f"  Max reconstruction error seen: {max_score_seen:.6f} (threshold_A={threshold_a:.6f})")
                else:
                    print(f"  Sequence too short for Channel A (length {T} < seq_len {seq_len})")
                    
    print(f"\nTotal undetected failures: {undetected_count}/{len(files)}")

if __name__ == "__main__":
    main()
