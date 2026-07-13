import numpy as np
from pathlib import Path
import pinocchio as pin

URDF_OR_XACRO = "/home/dyros/mcy_ws/piper-mou/src/piper_ros/src/piper_description/urdf/piper_description.xacro"
URDF_JOINT_NAMES = [f"joint{i+1}" for i in range(8)]
CTRL_JOINT_NAMES = ['joint1','joint2','joint3','joint4','joint5','joint6']

DISPLAY_IDX = [3, 4]   # joint4, joint5
GRAVITY = np.array([0.0, 0.0, -9.81])
PRED_RATE_HZ = 100.0

SIGMA = 0.180  # Increased from 0.0856 to prevent false-positive torque violations during Mode 9 Cartesian descent
FOUR_SIGMA = 4.0 * SIGMA

A = np.array([
    [0.336031, -0.046468,  0.258372,  0.180103, -0.022022, -1.702219],
    [0.11016,  -0.00126,  -0.015487,  1.4827,    0.00906,   0.558599],
    [-0.009071, -0.001866, -0.026797, -0.026905,  1.37266,   0.943661]
])

B = np.array([0.24018, 0.031745, 0.161793])

def build_pin_model(urdf_or_xacro: str):
    p = Path(urdf_or_xacro).expanduser().resolve()
    if p.suffix == ".xacro":
        import xacro
        xml = xacro.process_file(str(p)).toxml()
        mdl = pin.buildModelFromXML(xml)
    else:
        mdl = pin.buildModelFromUrdf(str(p))
    mdl.gravity.linear = np.array(GRAVITY, dtype=float)
    return mdl
