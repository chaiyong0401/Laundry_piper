
from scipy.spatial.transform import Rotation as R
import numpy as np

q = [0.4635, -0.5322, -0.5512, -0.4451]
r = R.from_quat(q)
rot_mat = r.as_matrix()

print("Rotation Matrix:")
print(rot_mat)

z_axis = rot_mat[:, 2] # 3rd column is local Z in world frame
print(f"\nLocal Z axis in World Frame: {z_axis}")

# Check Retreat Vector (-Z)
retreat = r.apply([0, 0, -0.05])
print(f"Retreat Vector (World): {retreat}")
