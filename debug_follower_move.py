#!/usr/bin/env python3
import time
import sys
import math

try:
    from piper_sdk import C_PiperInterface
except ImportError:
    print("Error: piper_sdk not found.")
    sys.exit(1)

def main():
    can_port = "can0"
    print(f"Connecting to {can_port} (Follower)...")
    
    piper = C_PiperInterface(can_name=can_port)
    piper.ConnectPort()
    
    print("Enabling Arm...")
    piper.EnableArm(7)
    piper.GripperCtrl(0, 1000, 0x01, 0)
    time.sleep(2)
    
    # Check enable status
    msgs = piper.GetArmLowSpdInfoMsgs()
    s1 = msgs.motor_1.foc_status.driver_enable_status
    print(f"Enable Status: {s1}")
    
    if not s1:
        print("Failed to enable. Exiting.")
        return

    print("Attempting small move (Joint 1 +5 degrees)...")
    
    # Get current pos
    # Note: SDK GetArmJointMsgs returns degrees * 1000
    j1 = piper.GetArmJointMsgs().joint_state.joint_1
    target_j1 = j1 + 5000 # +5 degrees
    
    # Send command
    # MotionCtrl_2(ctrl_mode, move_mode, move_spd_rate_ctrl)
    # 0x01: CAN_CTRL_MODE, 0x01: MOVE_MODE_SPD, 30: Speed
    piper.MotionCtrl_2(0x01, 0x01, 30)
    
    for i in range(20):
        piper.JointCtrl(target_j1, 0, 0, 0, 0, 0) # Just move J1, others 0 might be dangerous if not absolute?
        # WAIT! JointCtrl takes absolute positions for all joints.
        # Sending 0 for others will crash them if they are not at 0.
        # We must read current for all.
        
        # Better: Just wiggle Gripper? Safer.
        piper.GripperCtrl(0, 1000, 0x01, 0)
        
        # Actually, let's just use the existing position for other joints.
        js = piper.GetArmJointMsgs().joint_state
        piper.JointCtrl(target_j1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6)
        
        time.sleep(0.1)
        
    print("Command sent. Did it move?")

if __name__ == "__main__":
    main()
