
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import argparse
import time

def main(args):
    # Setup directories
    empty_dir = os.path.join(args.save_root, 'empty')
    cloth_dir = os.path.join(args.save_root, 'cloth')
    os.makedirs(empty_dir, exist_ok=True)
    os.makedirs(cloth_dir, exist_ok=True)
    
    print(f"Saving 'empty' to: {empty_dir}")
    print(f"Saving 'cloth' to: {cloth_dir}")
    print("Commands:")
    print("  'e': Save as EMPTY")
    print("  'c': Save as CLOTH")
    print("  'q': Quit")

    # Realsense Setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Try to start pipeline
    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Error starting camera: {e}")
        return

    empty_count = len(os.listdir(empty_dir))
    cloth_count = len(os.listdir(cloth_dir))
    print(f"Current counts -> Empty: {empty_count}, Cloth: {cloth_count}")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # Display
            display_img = color_image.copy()
            cv2.putText(display_img, f"E:{empty_count} C:{cloth_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_img, "[e]mpty / [c]loth / [q]uit", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
            cv2.imshow('Data Collection', display_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('e'):
                timestamp = int(time.time() * 1000)
                filename = f"empty_{timestamp}.jpg"
                filepath = os.path.join(empty_dir, filename)
                cv2.imwrite(filepath, color_image)
                print(f"Saved EMPTY: {filename}")
                empty_count += 1
            elif key == ord('c'):
                timestamp = int(time.time() * 1000)
                filename = f"cloth_{timestamp}.jpg"
                filepath = os.path.join(cloth_dir, filename)
                cv2.imwrite(filepath, color_image)
                print(f"Saved CLOTH: {filename}")
                cloth_count += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', type=str, default='/home/dyros/mcy_ws/Laundry_decision/dataset_for_rgb_gate',
                        help='Root directory for dataset')
    args = parser.parse_args()
    main(args)
