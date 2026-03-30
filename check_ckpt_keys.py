
import torch
import sys

ckpt_path = "/home/dyros/mcy_ws/Laundry_decision/run_laundry/models/rgb_only_best.pth"

try:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"Type of checkpoint: {type(ckpt)}")
    if isinstance(ckpt, dict):
        print(f"Keys: {ckpt.keys()}")
        if "model" in ckpt:
            print("Found 'model' key.")
        else:
            print("No 'model' key found. Checking first few keys...")
            print(list(ckpt.keys())[:5])
    else:
        print("Checkpoint is not a dict.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
