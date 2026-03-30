import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import cv2

class PiperDataset(Dataset):
    def __init__(self, data_path, chunk_size, stats=None):
        super().__init__()
        self.data_path = data_path
        self.chunk_size = chunk_size
        
        # Load multiple files
        import os
        import glob
        
        if os.path.isdir(data_path):
            files = sorted(glob.glob(os.path.join(data_path, "*.h5")))
        else:
            files = [data_path]
            
        if len(files) == 0:
            raise ValueError(f"No .h5 files found in {data_path}")
            
        print(f"Found {len(files)} episodes: {files}")
        
        all_images = []
        all_qpos = []
        
        for f_path in files:
            with h5py.File(f_path, 'r') as f:
                # Assuming simple concatenation for now
                # Ideally we should respect episode boundaries for chunks
                # But for simple implementation, we just concat
                all_images.append(f['observations/images'][:])
                all_qpos.append(f['observations/qpos'][:])
        
        self.images = np.concatenate(all_images, axis=0)
        self.qpos = np.concatenate(all_qpos, axis=0)
            
        self.num_samples = len(self.images)
        self.action_dim = self.qpos.shape[1]
        
        # Calculate stats if not provided
        if stats is None:
            self.stats = self.get_stats()
        else:
            self.stats = stats
            
        # Normalize Data
        self.qpos = self.normalize(self.qpos, self.stats['qpos_mean'], self.stats['qpos_std'])
        # Images are normalized to [0,1] and then standard ImageNet norm usually. 
        # Here we keep simple 0-1 for now or handled in getitem.

    def get_stats(self):
        qpos_mean = np.mean(self.qpos, axis=0)
        qpos_std = np.std(self.qpos, axis=0)
        # Avoid div by zero
        qpos_std = np.clip(qpos_std, 1e-2, None) 
        return {
            'qpos_mean': qpos_mean,
            'qpos_std': qpos_std
        }
        
    def normalize(self, data, mean, std):
        return (data - mean) / std
        
    def unnormalize(self, data, mean, std):
        return data * std + mean

    def __len__(self):
        # We need future chunks, so length is num_samples - chunk_size
        return max(0, self.num_samples - self.chunk_size)

    def __getitem__(self, idx):
        # Image
        # ResNet expects (C, H, W)
        img = self.images[idx] # (H, W, C)
        img = cv2.resize(img, (320, 240)) # Resize to save memory/compute
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        
        # State (Joints)
        qpos = torch.from_numpy(self.qpos[idx]).float()
        
        # Action Chunk
        # Actions are usually the qpos of future steps in Joint Control
        # If we control using Joint Position, Action = Target Joint Position
        action_chunk = self.qpos[idx : idx + self.chunk_size]
        action_chunk = torch.from_numpy(action_chunk).float()
        
        # Masking (is_pad)
        # If we are at the very end, we might need padding. 
        # But here we limited __len__ so we are safe.
        is_pad = torch.zeros(self.chunk_size).bool()
        
        return img, qpos, action_chunk, is_pad
