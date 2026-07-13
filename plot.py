import numpy as np, glob, matplotlib
matplotlib.use("Agg")                                                                              
import matplotlib.pyplot as plt
                                                                                                    
files = sorted(glob.glob("grasp_05/grasp_*_success.npy"))                                          
fig, axes = plt.subplots(len(files), 1, figsize=(12, 2*len(files)))
for ax, f in zip(axes, files):                                                                     
    seq = np.load(f)          # (T, 6)                                                             
    mrl = seq[:, 0]           # MRL                                                                
    ax.plot(mrl)                                                                                   
    ax.set_title(f.split("/")[-1], fontsize=8)
    ax.set_ylim(0, 1)                                                                              
    ax.axhline(0.5, color='r', linestyle='--', alpha=0.3)
plt.tight_layout()                                                                                 
plt.savefig("grasp_success_mrl.png", dpi=100)
print("저장: grasp_success_mrl.png") 