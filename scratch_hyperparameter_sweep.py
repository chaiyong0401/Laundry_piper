import os
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import sys

sys.path.insert(0, "/home/dyros/mcy_ws")
from tactile_anomaly.model import GRUAE
from train_gru_ae import load_sequences_windowed, fit_scaler, apply_scaler, apply_relative_transform, load_sequences_raw, evaluate_dual_channel_combined

def train_model(seqs_norm, input_dim, hidden_dim, num_layers, seq_len, device):
    # Convert to PyTorch Dataset
    x_train = np.array(seqs_norm, dtype=np.float32)
    dataset = TensorDataset(torch.tensor(x_train))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = GRUAE(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.2, seq_len=seq_len)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train for 150 epochs (enough for quick convergence)
    model.train()
    for epoch in range(150):
        for batch in loader:
            x_b = batch[0].to(device)
            loss = model.reconstruction_error(x_b).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def main():
    data_dir = "/home/dyros/mcy_ws/transport_new"
    prefix = "transport"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We sweep over:
    # 1. seq_len: [30, 50]
    # 2. hidden_dim: [64, 128]
    # 3. num_layers: [1, 2]
    
    configs = [
        {"seq_len": 30, "hidden_dim": 64, "num_layers": 1},
        {"seq_len": 30, "hidden_dim": 64, "num_layers": 2},
        {"seq_len": 30, "hidden_dim": 128, "num_layers": 2},
        {"seq_len": 50, "hidden_dim": 64, "num_layers": 1},
        {"seq_len": 50, "hidden_dim": 64, "num_layers": 2},
        {"seq_len": 50, "hidden_dim": 128, "num_layers": 2}, # Current
    ]
    
    print(f"Starting Hyperparameter Sweep on {device}...")
    print(f"{'SeqLen':<6} | {'Hidden':<6} | {'Layers':<6} | {'Thresh A':<10} | {'Success FA (2,5)':<16} | {'Failure Det (2,5)':<18}")
    print("-" * 80)
    
    for c in configs:
        seq_len = c["seq_len"]
        hidden_dim = c["hidden_dim"]
        num_layers = c["num_layers"]
        
        # Load success sequences for this window length
        success_seqs = load_sequences_windowed(data_dir, prefix, "success", use_raw=True, seq_len=seq_len, stride=10)
        input_dim = success_seqs[0].shape[1]
        
        scaler = fit_scaler(success_seqs)
        success_seqs_norm = apply_scaler(success_seqs, scaler)
        
        # Train model
        model = train_model(success_seqs_norm, input_dim, hidden_dim, num_layers, seq_len, device)
        model.eval()
        
        # Get threshold A (99.5th percentile)
        with torch.no_grad():
            x_val = torch.tensor(np.array(success_seqs_norm, dtype=np.float32)).to(device)
            val_errors = []
            # compute error in batches to avoid GPU memory overflow
            for i in range(0, len(x_val), 128):
                batch_errs = [model.reconstruction_error(x_val[j:j+1]).item() for j in range(i, min(i+128, len(x_val)))]
                val_errors.extend(batch_errs)
        threshold_a = float(np.percentile(val_errors, 99.5))
        
        # Load raw evaluation data
        raw_seqs_by_label = {}
        raw_success_all = load_sequences_raw(data_dir, prefix, "success", use_raw=True)
        raw_seqs_by_label["success"] = raw_success_all
        raw_seqs_by_label["failure"] = load_sequences_raw(data_dir, prefix, "failure", use_raw=True)
        
        # Evaluate
        combined = evaluate_dual_channel_combined(
            model=model, raw_seqs_by_label=raw_seqs_by_label, scaler=scaler,
            threshold_a=threshold_a, threshold_b=0.212874,
            seq_len=seq_len, relative_mode="none", weight_alpha=0.0,
            n_rel_dims=None, device=device, smooth_window=3,
            consec_a=2, consec_b=5, infer_stride=10, hybrid=False,
            mrl_thresh=0.2, cutoff_multiplier=2.2, warmup=30
        )
        
        det_succ, tot_succ = combined["success"]["ab"]
        det_fail, tot_fail = combined["failure"]["ab"]
        
        fa_pct = (det_succ / tot_succ) * 100
        det_pct = (det_fail / tot_fail) * 100
        
        print(f"{seq_len:<6} | {hidden_dim:<6} | {num_layers:<6} | {threshold_a:.6f} | {det_succ}/{tot_succ} ({fa_pct:4.1f}%) | {det_fail}/{tot_fail} ({det_pct:4.1f}%)")

if __name__ == "__main__":
    main()
