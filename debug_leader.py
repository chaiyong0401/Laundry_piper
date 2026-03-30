#!/usr/bin/env python3
import time
import sys
import os

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

    print("Sending EnableArm(7)...")
    piper.EnableArm(7)
    piper.GripperCtrl(0, 1000, 0x01, 0)
    
    print("Monitoring status for 5 seconds...")
    for i in range(10):
        try:
            msgs = piper.GetArmLowSpdInfoMsgs()
            # Check if msgs is valid (SDK might return object with default zeros if no data)
            
            s1 = msgs.motor_1.foc_status.driver_enable_status
            s2 = msgs.motor_2.foc_status.driver_enable_status
            s3 = msgs.motor_3.foc_status.driver_enable_status
            s4 = msgs.motor_4.foc_status.driver_enable_status
            s5 = msgs.motor_5.foc_status.driver_enable_status
            s6 = msgs.motor_6.foc_status.driver_enable_status
            print(f"Step {i}: Enable Status: {[s1, s2, s3, s4, s5, s6]}")
            
            # Keep sending enable command
            piper.EnableArm(7)
            piper.GripperCtrl(0, 1000, 0x01, 0)
            
        except Exception as e:
            print(f"Step {i}: Error reading status: {e}")
            
        time.sleep(0.5)

if __name__ == "__main__":
    main()
