#!/usr/bin/env python3
import h5py
import numpy as np
import cv2
import sys
import os

def main():
    filename = "data/episode_20260107_130657.h5"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        return

    print(f"Inspecting {filename}...")
    
    with h5py.File(filename, 'r') as f:
        print("Keys:", list(f.keys()))
        
        if 'observations' in f:
            obs = f['observations']
            print("Observations Keys:", list(obs.keys()))
            
            if 'images' in obs:
                imgs = obs['images']
                print(f"Images Shape: {imgs.shape}, Type: {imgs.dtype}")
                
                # Save first image to check
                if imgs.shape[0] > 0:
                    cv2.imwrite("sample_image.jpg", imgs[0])
                    print("Saved 'sample_image.jpg' for inspection.")
            
            if 'qpos' in obs:
                qpos = obs['qpos']
                print(f"Joints (qpos) Shape: {qpos.shape}, Type: {qpos.dtype}")
                if qpos.shape[0] > 0:
                    print(f"Sample Joint Data [0]: {qpos[0]}")

            if 'ee_pose' in obs:
                ee = obs['ee_pose']
                print(f"EE Pose Shape: {ee.shape}, Type: {ee.dtype}")
                if ee.shape[0] > 0:
                    print(f"Sample EE Pose [0]: {ee[0]}")
                    
        num_samples = f.attrs.get('num_samples', 'Unknown')
        print(f"Total Samples (Metadata): {num_samples}")

if __name__ == "__main__":
    main()
