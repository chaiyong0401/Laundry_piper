import numpy as np
import sys
sys.path.insert(0, "/home/dyros/mcy_ws")
from dual_channel_detector import FramewiseDeltaDetector

files = [
    "transport_20260522_152441_634491_failure_raw.npy",
    "transport_20260527_203840_506373_failure_raw.npy"
]

data_dir = "/home/dyros/mcy_ws/transport_new"

for filename in files:
    path = f"{data_dir}/{filename}"
    seq = np.load(path).astype(np.float32)
    T = len(seq)
    
    mrls = [FramewiseDeltaDetector._compute_mrl(frame) for frame in seq]
    norms = [np.linalg.norm(frame) for frame in seq]
    
    print(f"\n======================================")
    print(f"File: {filename} (Length: {T} frames)")
    print(f"======================================")
    
    # Print MRL and Norm at key milestones (every 10% of the sequence length)
    indices = np.linspace(0, T-1, 10, dtype=int)
    print(f"{'Frame':<8} | {'MRL':<8} | {'Norm':<8}")
    print("-" * 30)
    for idx in indices:
        print(f"{idx:<8} | {mrls[idx]:.4f} | {norms[idx]:.4f}")
        
    print(f"\nStats:")
    print(f"  MRL:  mean={np.mean(mrls):.4f}, std={np.std(mrls):.4f}, min={np.min(mrls):.4f}, max={np.max(mrls):.4f}")
    print(f"  Norm: mean={np.mean(norms):.4f}, std={np.std(norms):.4f}, min={np.min(norms):.4f}, max={np.max(norms):.4f}")
