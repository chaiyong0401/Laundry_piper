import numpy as np
import sys
sys.path.insert(0, "/home/dyros/mcy_ws")
from dual_channel_detector import FramewiseDeltaDetector

files = [
    "transport_20260522_152441_634491_failure_raw.npy",
    "transport_20260527_203840_506373_failure_raw.npy",
    "transport_20260527_210604_679439_failure_raw.npy"
]

data_dir = "/home/dyros/mcy_ws/transport_new"

for filename in files:
    path = f"{data_dir}/{filename}"
    seq = np.load(path).astype(np.float32)
    T = len(seq)
    
    mrls = [FramewiseDeltaDetector._compute_mrl(frame) for frame in seq]
    norms = [np.linalg.norm(frame) for frame in seq]
    
    print(f"\nFile: {filename}")
    print(f"  Length: {T} frames")
    print(f"  Initial MRL (first 10 frames avg): {np.mean(mrls[:10]):.4f}, Norm: {np.mean(norms[:10]):.4f}")
    print(f"  Final MRL (last 10 frames avg): {np.mean(mrls[-10:]):.4f}, Norm: {np.mean(norms[-10:]):.4f}")
    print(f"  Min MRL: {np.min(mrls):.4f}, Max MRL: {np.max(mrls):.4f}")
    print(f"  Min Norm: {np.min(norms):.4f}, Max Norm: {np.max(norms):.4f}")
    
    # Check if there is any drop in MRL at the end
    # Let's print the last 30 frames of MRL
    print("  Last 20 MRLs: ", [round(float(m), 3) for m in mrls[-20:]])
    print("  Last 20 Norms:", [round(float(n), 3) for n in norms[-20:]])
