#!/usr/bin/env python3
import time
import sys
import can

try:
    from piper_sdk import C_PiperInterface
except ImportError:
    print("Error: piper_sdk not found. Make sure to source the workspace.")
    sys.exit(1)

def main():
    can_port = "can0"
    print(f"Connecting to {can_port}...")
    
    # Initialize interface
    piper = C_PiperInterface(can_name=can_port)
    piper.ConnectPort()
    
    print("----------------------------------------------------------------")
    print("FORCE SLAVE V2: Trying 0xFC (Motion Output) and 0x00 (Default)")
    print("INSTRUCTION: KEEP MOVING THE ROBOT ARM WHILE THIS RUNS!")
    print("----------------------------------------------------------------")

    idx = 0
    while True:
        try:
            # Method 1: Set to Motion Output (Slave-like)
            # 0xFC: Motion Output Arm
            print(f"[{idx}] Sending MasterSlaveConfig(0xFC)...")
            piper.MasterSlaveConfig(0xFC, 0, 0, 0)
            time.sleep(0.1)
            
            # Method 2: Restore Default
            # 0x00: Default linkage
            print(f"[{idx}] Sending MasterSlaveConfig(0x00)...")
            piper.MasterSlaveConfig(0x00, 0, 0, 0)
            time.sleep(0.1)
            
            # Method 3: Stop Drag Teach explicit
            piper.MotionCtrl_1(emergency_stop=0x02, grag_teach_ctrl=0x02) # STOP drag teach
            
            # Method 4: Enable Arm (Standard Handshake)
            piper.EnableArm(7)
            piper.GripperCtrl(0, 1000, 0x01, 0)
            
            # Check for Slave Feedback (0x2Ax)
            msgs = piper.GetArmLowSpdInfoMsgs()
            # If we are in Master mode, these are usually not updated or stale.
            # But let's check one specific motor status
            s1 = msgs.motor_1.foc_status.driver_enable_status
            
            # Also check raw messages if possible, but SDK hides them.
            # We rely on 's1' being updated or at least readable.
            # If the robot switches to slave, it should enable.
            
            print(f"    -> Current Enable Status reported by SDK: {s1}")
            
            if idx % 10 == 0:
                print("    (Keep moving the arm!)")
                
            idx += 1
            time.sleep(0.2)
            
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
