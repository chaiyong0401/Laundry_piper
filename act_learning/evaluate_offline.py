import torch
from torch.utils.data import DataLoader
from act_learning.policy import ACTPolicy
from act_learning.utils import PiperDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    ckpt_path = args.ckpt_path
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    stats = checkpoint['stats']
    
    policy = ACTPolicy(config).to(device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    print("Model loaded.")

    # Load Data (Use same path as training for simplicity, or specific file)
    dataset = PiperDataset(args.data_path, config['chunk_size'], stats=stats)
    # Just take first N samples (e.g. one episode length approx)
    eval_len = min(len(dataset), 400) 
    print(f"Evaluating on first {eval_len} steps...")

    gt_trajectory = []
    pred_trajectory = []

    with torch.no_grad():
        for i in range(eval_len):
            img, qpos, action_chunk, is_pad = dataset[i]
            
            img = img.unsqueeze(0).to(device) # [1, C, H, W]
            qpos = qpos.unsqueeze(0).to(device) # [1, Dim]
            
            # Inference
            # For evaluation, we want deterministic output usually, but CVAE is stochastic.
            # We can set mu=0 to check mean behavior, or sample. ACT usually samples mu=0 during inference.
            action_preds, _, _ = policy(qpos, img, actions=None)
            
            # action_preds: [1, Chunk, Dim]
            # We take the first step of the chunk (Receding Horizon Control)
            # Or we could plot the whole chunk, but for trajectory plotting, next step is standard.
            pred_action = action_preds[0, 0].cpu().numpy()
            
            # Unnormalize
            pred_action = dataset.unnormalize(pred_action, stats['qpos_mean'], stats['qpos_std'])
            gt_action = dataset.unnormalize(dataset.qpos[i+1] if i+1 < len(dataset.qpos) else dataset.qpos[i], 
                                          stats['qpos_mean'], stats['qpos_std'])
            
            pred_trajectory.append(pred_action)
            gt_trajectory.append(gt_action)
            
            if i % 50 == 0:
                print(f"Step {i}/{eval_len}")

    pred_trajectory = np.array(pred_trajectory)
    gt_trajectory = np.array(gt_trajectory)

    # Plotting
    print("Plotting results...")
    dim = pred_trajectory.shape[1]
    # Assuming 6 joints + gripper, or whatever dataset has.
    # Create subplots
    fig, axes = plt.subplots(dim, 1, figsize=(10, 2*dim))
    if dim == 1: axes = [axes]
    
    for j in range(dim):
        ax = axes[j]
        ax.plot(gt_trajectory[:, j], label='Ground Truth', color='black', linestyle='--')
        ax.plot(pred_trajectory[:, j], label='Prediction', color='blue')
        ax.set_title(f'Joint {j+1}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    save_path = 'eval_result.png'
    plt.savefig(save_path)
    print(f"Evaluation plot saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/dyros/mcy_ws/data')
    parser.add_argument('--ckpt_path', type=str, default='/home/dyros/mcy_ws/act_learning/ckpts/policy_full_last.ckpt')
    args = parser.parse_args()
    
    evaluate(args)
