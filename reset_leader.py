#!/usr/bin/env python3
import time
import sys

try:
    from piper_sdk import C_PiperInterface
except ImportError:
    print("Error: piper_sdk not found. Make sure to source the workspace.")
    sys.exit(1)

def main():
    can_port = "can1"
    print(f"Attempting to connect to {can_port}...")
    
    try:
        piper = C_PiperInterface(can_name=can_port)
        piper.ConnectPort()
        print("ConnectPort called.")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print("----------------------------------------------------------------")
    print("FORCE RESET MODE: Sending RESET commands every 0.1s...")
    print("INSTRUCTION: PLEASE MOVE/SHAKE THE LEADER ROBOT ARM MANUALLY")
    print("WHILE THIS SCRIPT IS RUNNING TO WAKE UP THE CONTROLLER.")
    print("----------------------------------------------------------------")

    idx = 0
    success_count = 0
    
    try:
        while True:
            # 1. Force Exit Teaching / Master Mode
            # 0x00: Restore default linkage (Slave)
            piper.MasterSlaveConfig(0x00, 0, 0, 0)
            
            # 2. Stop Drag Teaching
            piper.MotionCtrl_1(emergency_stop=0x02, grag_teach_ctrl=0x02)
            
            # 3. Set Control Mode to CAN Control
            piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=30)
            
            # 4. Enable Arm
            piper.EnableArm(7)
            piper.GripperCtrl(0, 1000, 0x01, 0)
            
            # Check Status
            msgs = piper.GetArmLowSpdInfoMsgs()
            # If we are in Slave mode, we should receive continuous feedback (motor status)
            # If msgs are all False/Zero, we are still not getting standard feedback.
            
            s1 = msgs.motor_1.foc_status.driver_enable_status
            s2 = msgs.motor_2.foc_status.driver_enable_status
            
            if s1 or s2: # If we see ANY status, it means we are getting 0x2A messages!
                success_count += 1
                status_str = f"ALIVE (Enable: {s1})"
            else:
                success_count = 0
                status_str = "No Feedback (Stuck in Master?)"

            if idx % 5 == 0: # Print every 0.5s
                print(f"Step {idx}: {status_str} - Sending Reset Commands...")
            
            idx += 1
            time.sleep(0.1)
            
            if success_count > 10:
                print("\nSUCCESS! Robot seems to be back in Slave Mode and responding.")
                print("You can now Ctrl+C and run the ROS driver.")
                break

    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()
