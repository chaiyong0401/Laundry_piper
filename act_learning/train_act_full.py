import torch
from torch.utils.data import DataLoader
from act_learning.policy_full import ACTPolicyBaseline as ACTPolicyFull
from act_learning.utils import PiperDataset
import os
import argparse
import numpy as np

def train(args):
    # Enable CUDA optimization
    torch.backends.cudnn.benchmark = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("WARNING: You are trying to train the FULL model on CPU. This will be very slow.")

    # Dataset
    dataset = PiperDataset(args.data_path, args.chunk_size)
    print(f"Dataset Size: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    # Model Args (Heavier Config)
    policy_config = {
        'state_dim': dataset.action_dim, 
        'action_dim': dataset.action_dim,
        'chunk_size': args.chunk_size,
        'camera_names': ['front'],
        'latent_dim': 32,
        'hidden_dim': 512,
        'n_heads': 8,
        'n_enc_layers': 6, # Deeper for Full Model
        'n_dec_layers': 6  # Deeper for Full Model
    }
    
    policy = ACTPolicyFull(policy_config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    
    print(f"Model Parameters: {sum(p.numel() for p in policy.parameters())}")
    
    # Training Loop
    for epoch in range(args.epochs):
        policy.train()
        epoch_loss = []
        
        for batch_idx, (img, qpos, actions, is_pad) in enumerate(dataloader):
            img = img.to(device, non_blocking=True)
            qpos = qpos.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward
            action_preds, mu, log_var = policy(qpos, img, actions)
            
            # Loss
            l1_loss = torch.nn.functional.l1_loss(action_preds, actions, reduction='mean')
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            kl_loss /= (actions.shape[0] * actions.shape[1]) 
            
            kl_weight = 10
            loss = l1_loss + kl_weight * kl_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss.append(loss.item())
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        avg_loss = np.mean(epoch_loss)
        print(f"Epoch {epoch} Done. Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % 20 == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"policy_full_epoch_{epoch+1}.ckpt")
            torch.save({
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'stats': dataset.stats, 
                'config': policy_config
            }, ckpt_path)

    # Save last
    torch.save({
        'model_state_dict': policy.state_dict(),
        'stats': dataset.stats,
        'config': policy_config
    }, os.path.join(args.ckpt_dir, "policy_full_last.ckpt"))
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data') # Expects data in current dir on training PC
    parser.add_argument('--ckpt_dir', type=str, default='./ckpts')
    parser.add_argument('--batch_size', type=int, default=32) # Bigger batch for GPU
    parser.add_argument('--epochs', type=int, default=500) # Full training needs more epochs
    parser.add_argument('--lr', type=float, default=1e-5) # Lower LR for fine-tuning/transformer
    parser.add_argument('--chunk_size', type=int, default=50) # Larger chunk
    
    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    train(args)
