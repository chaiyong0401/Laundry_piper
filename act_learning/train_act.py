import torch
from torch.utils.data import DataLoader
from act_learning.policy import ACTPolicy
from act_learning.utils import PiperDataset
import os
import argparse
import numpy as np

def train(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    data_path = args.data_path
    dataset = PiperDataset(data_path, args.chunk_size)
    print(f"Dataset Size: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Model Args
    policy_config = {
        'state_dim': dataset.action_dim, # qpos dim
        'action_dim': dataset.action_dim, # action dim (same as qpos for joint ctrl)
        'chunk_size': args.chunk_size,
        'camera_names': ['front'],
        'latent_dim': 32,
        'hidden_dim': 512,
        'n_heads': 8,
        'n_enc_layers': 4,
        'n_dec_layers': 4
    }
    
    policy = ACTPolicy(policy_config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    
    # Training Loop
    for epoch in range(args.epochs):
        policy.train()
        epoch_loss = []
        
        for batch_idx, (img, qpos, actions, is_pad) in enumerate(dataloader):
            img = img.to(device)
            qpos = qpos.to(device)
            actions = actions.to(device)
            is_pad = is_pad.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            action_preds, mu, log_var = policy(qpos, img, actions)
            
            # Loss Calculation (L1 + KL)
            # 1. Reconstruction Loss (L1)
            # Masking padding? For now assume valid.
            l1_loss = torch.nn.functional.l1_loss(action_preds, actions, reduction='mean')
            
            # 2. KL Divergence
            # KL(N(mu, sigma) || N(0, 1))
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            kl_loss /= (actions.shape[0] * actions.shape[1]) # Mean over batch/chunk
            
            # Total Loss
            # beta for KL? Usually small like 10 or dynamic
            kl_weight = 10
            loss = l1_loss + kl_weight * kl_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss.append(loss.item())
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f} (L1: {l1_loss.item():.4f}, KL: {kl_loss.item():.4f})")
        
        avg_loss = np.mean(epoch_loss)
        print(f"Epoch {epoch} Done. Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"policy_epoch_{epoch+1}.ckpt")
            torch.save({
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'stats': dataset.stats, # Important to save stats for inference normalization!
                'config': policy_config
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # Save last
    torch.save({
        'model_state_dict': policy.state_dict(),
        'stats': dataset.stats,
        'config': policy_config
    }, os.path.join(args.ckpt_dir, "policy_last.ckpt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/dyros/mcy_ws/data')
    parser.add_argument('--ckpt_dir', type=str, default='/home/dyros/mcy_ws/act_learning/ckpts')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50) # Reduced for demo
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--chunk_size', type=int, default=20) # 20 steps prediction
    
    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    train(args)
