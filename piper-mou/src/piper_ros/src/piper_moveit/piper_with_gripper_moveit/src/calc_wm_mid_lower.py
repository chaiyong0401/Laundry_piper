
import numpy as np
import pinocchio as pin
from pathlib import Path
import xacro

# --- Constants & Paths from basket.py ---
URDF_OR_XACRO = "/home/dyros/mcy_ws/piper-mou/src/piper_ros/src/piper_description/urdf/piper_description.xacro"
# URDF_JOINT_NAMES = [f"joint{i+1}" for i in range(8)] # Not strictly needed for core Algo but good for reference
GRAVITY = np.array([0.0, 0.0, -9.81])

# Current wm_init from user request
wm_target_current = np.array([1.5640, 0.4538, -1.5319, -0.0488, 1.2797, 0.3169, 0.0, 0.0])

def build_pin_model(urdf_or_xacro: str):
    p = Path(urdf_or_xacro).expanduser().resolve()
    if p.suffix == ".xacro":
        xml = xacro.process_file(str(p)).toxml()
        mdl = pin.buildModelFromXML(xml)
    else:
        mdl = pin.buildModelFromUrdf(str(p))
    mdl.gravity.linear = np.array(GRAVITY, dtype=float)
    return mdl

def solve_ik_pos_only(model, data, target_pos, q_init, eps=1e-4, max_iter=1000, dt=1e-1):
    """
    Simple damped pseudo-inverse IK for Position Only (ignoring orientation for now, or we can try to keep it).
    Since user just said 'lower 10cm', usually they want same orientation.
    """
    idx_ee = model.getFrameId("link8") # Assuming link8 is EE or close to it. In basket.py it doesn't specify frame, but usually last link.
    # Let's check frame ID log or assume last joint's child frame.
    # Actually, let's look at basket.py usage. It uses pin.rnea but not IK there.
    # We will use CLIK (Closed-Loop Inverse Kinematics) or simple Jacobian transpose/pseudo-inverse
    
    q = q_init.copy()
    
    # Selection matrix for Position (3) + Orientation (3) = 6 DOF
    # If 6DOF arm, we can try to match full pose.
    
    for i in range(max_iter):
        pin.framesForwardKinematics(model, data, q)
        # pin.computeJointJacobians(model, data, q) # Done in framesFK in some versions? No.
        
        curr_pose = data.oMf[idx_ee]
        curr_pos = curr_pose.translation
        curr_rot = curr_pose.rotation
        
        # Error
        err_pos = target_pos - curr_pos
        # Orientation error (Lie algebra log) - try to keep original orientation
        # We need the target orientation from the initial forward kinematics
        # But this function only takes target_pos? Let's make it take target_pose (SE3)
        pass 
        
    return q

def main():
    print("Loading Robot Model...")
    model = build_pin_model(URDF_OR_XACRO)
    data = model.createData()
    
    # 1. Forward Kinematics for Current wm_mid
    # wm_mid is 8 DOF (6 arm + 2 gripper). Pinocchio model from URDF might be 6 or 8 depending on definition.
    # URDF_JOINT_NAMES in basket.py len is 8.
    # Let's assume model.nq is 8.
    if model.nq != 8:
        print(f"Warning: Model nq ({model.nq}) != 8. Truncating or padding input.")
        # If model is 6 dof arm, we take first 6.
    
    q_current = wm_target_current[:model.nq]
    
    pin.forwardKinematics(model, data, q_current)
    pin.updateFramePlacements(model, data)
    
    # Find EE Frame. usually 'link8' or 'joint6' child.
    # Let's just use the last frame in the model.
    ee_frame_id = model.nframes - 1 
    # Or search for 'link8' or 'end_effector'
    for i, f in enumerate(model.frames):
        if "link8" in f.name:
            ee_frame_id = i
            break
            
    current_pose = data.oMf[ee_frame_id]
    print(f"Current EE Position: {current_pose.translation}")
    print(f"Current EE Rotation:\n{current_pose.rotation}")
    
    # 2. Target Pose (Z - 10cm)
    target_pose = current_pose.copy()
    target_pose.translation[2] -= 0.10
    print(f"Target EE Position:  {target_pose.translation}")
    
    # 3. Solve IK
    print("Solving IK...")
    
    q = q_current.copy()
    damp = 1e-6
    success = False
    
    for i in range(5000):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        
        curr_pose = data.oMf[ee_frame_id]
        dMf = target_pose.actInv(curr_pose) # relative transform
        err = pin.log(dMf).vector # 6D error vector (linear + angular)
        
        if np.linalg.norm(err) < 1e-4:
            success = True
            break
            
        J = pin.computeFrameJacobian(model, data, q, ee_frame_id)
        
        # Damped Pseudo Inverse
        v = - J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), err)
        q = pin.integrate(model, q, v * 0.1) # small step
        
    if success:
        print("\nSUCCESS!")
        print("-" * 30)
        print("New wm_mid (10cm lower):")
        # Format as list for easy copy-paste
        out_list = list(q)
        if len(out_list) < 8:
            out_list = out_list + [0.0, 0.0] # Restore gripper
            
        print(f"[{', '.join(f'{x:.9f}' for x in out_list)}]")
        print("-" * 30)
    else:
        print("\nIK Failed to converge.")

if __name__ == "__main__":
    main()
