#!/usr/bin/env python3
import socket
import struct
import sys
import os

def open_can_socket(interface):
    try:
        s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        s.bind((interface,))
        return s
    except OSError as e:
        print(f"Error opening {interface}: {e}")
        return None

def main():
    interface = "can0"
    print(f"--------------------------------------------------")
    print(f"LISTENING ON {interface}...")
    print(f"PLEASE MOVE THE ROBOT ARM NOW TO GENERATE TRAFFIC!")
    print(f"--------------------------------------------------")
    
    sock = open_can_socket(interface)
    if not sock:
        sys.exit(1)

    count = 0
    ids_seen = set()
    
    try:
        while True:
            # Set timeout so we don't block forever
            sock.settimeout(1.0)
            try:
                frame = sock.recv(16)
                can_id, can_dlc, data = struct.unpack("=IB3x8s", frame)
                can_id = can_id & 0x7FFFFFFF
                
                hex_id = hex(can_id)
                ids_seen.add(hex_id)
                
                print(f"CAPTURED: ID={hex_id} DLC={can_dlc} Data={data.hex()}")
                
                if can_id >= 0x150 and can_id <= 0x160:
                    print(f"  -> DETECTED MASTER/TEACH MODE ID: {hex_id}")
                elif can_id >= 0x2A0 and can_id <= 0x2B0:
                    print(f"  -> DETECTED SLAVE/FEEDBACK MODE ID: {hex_id}")
                
                count += 1
                if count >= 30:
                    print("\nCaptured 30 frames. Stopping.")
                    print("Summary of IDs seen:", sorted(list(ids_seen)))
                    break
            except socket.timeout:
                print("Waiting for traffic... (Move the robot!)")
                continue
                
    except KeyboardInterrupt:
        print("\nStopped manually.")
    except Exception as e:
        print(f"Error reading: {e}")

if __name__ == "__main__":
    main()
