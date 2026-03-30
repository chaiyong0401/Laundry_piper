#!/usr/bin/env python3
"""
Hand-Eye TF Collector for Piper (ROS 2) + RealSense & ArUco
    s : sample TF pair
    c : calibrate + print/save results
    q : quit
"""

import argparse
import sys
import numpy as np
import cv2
import pyrealsense2 as rs
import yaml
from scipy.linalg import sqrtm
from numpy.linalg import inv
import threading
import time
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import Pose


# ------------------------------------------------
# Math Helpers
# ------------------------------------------------
# Custom solver removed in favor of cv2.calibrateHandEye


# ------------------------------------------------
# ROS 2 Node for Piper EE Pose (Using TF2 to bypass SDK bug)
# ------------------------------------------------
class PiperTFSubscriber(Node):
    def __init__(self, base_frame='base_link', ee_frame='link6'):
        super().__init__('calib_pose_sub')
        self.base_frame = base_frame
        self.ee_frame = ee_frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.get_logger().info(f"Subscribed to TF: {self.base_frame} -> {self.ee_frame}")

    def get_ee_pose(self):
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(self.base_frame, self.ee_frame, now)
            
            pos = trans.transform.translation
            ori = trans.transform.rotation
            r = R.from_quat([ori.x, ori.y, ori.z, ori.w])
            T = np.eye(4)
            T[:3, :3] = r.as_matrix()
            T[:3, 3] = [pos.x, pos.y, pos.z]
            return T
        except Exception as e:
            self.get_logger().error(f"TF Lookup Error: {e}", throttle_duration_sec=2.0)
            return None


# ------------------------------------------------
# Main Collector Class
# ------------------------------------------------
class PiperTFCollector:
    def __init__(self, cfg):
        self.cfg = cfg

        # ROS2 Setup
        self.ros_node = PiperTFSubscriber()
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.ros_node)
        self.ros_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.ros_thread.start()

        # RealSense Setup
        self.pipeline = rs.pipeline()
        rs_cfg = rs.config()
        rs_cfg.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.rgb8, cfg.fps)
        self.pipeline.start(rs_cfg)
        self.K = None
        self.dist = None

        # ArUco Setup
        if hasattr(cv2, 'aruco'):
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, 'DICT_' + cfg.dictionary))
            self.aruco_param = cv2.aruco.DetectorParameters()
            # OpenCV 4.7+ uses ArucoDetector
            if hasattr(cv2.aruco, 'ArucoDetector'):
                self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_param)
            else:
                self.aruco_detector = None
        else:
            print("Error: cv2.aruco not found! Check opencv-contrib-python installation.")
            sys.exit(1)

        # Buffers
        self.list_tf_base2ee = []
        self.list_tf_cam2qr = []

        cv2.namedWindow('Piper Hand-Eye Cailbration', cv2.WINDOW_NORMAL)

    def get_intrinsics(self):
        if self.K is None:
            prof = self.pipeline.get_active_profile()
            intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.K = np.array([[intr.fx, 0, intr.ppx],
                               [0, intr.fy, intr.ppy],
                               [0, 0, 1]], dtype=np.float32)
            self.dist = np.array(intr.coeffs[:5], dtype=np.float32)
            print("Captured RealSense Intrinsics.")

    def run(self):
        print("\n--- Piper ArUco Calibration Started ---")
        print("1. Stand clear, ensure the ArUco marker is visible on screen.")
        print("2. Press 's' to capture a sample (Move arm to different angles/positions).")
        print("3. Press 'c' to calculate and save when you have 10-20 samples.")
        print("4. Press 'q' to quit.\n")

        try:
            while rclpy.ok():
                frame = self.pipeline.wait_for_frames().get_color_frame()
                if not frame:
                    continue

                img = np.asanyarray(frame.get_data())
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                self.get_intrinsics()

                # Get EE Pose from TF Tree
                base2ee = self.ros_node.get_ee_pose()

                # Get ArUco Pose
                cam2qr = None
                if self.aruco_detector is not None:
                    corners, ids, _ = self.aruco_detector.detectMarkers(img)
                else:
                    corners, ids, _ = cv2.aruco.detectMarkers(img, self.aruco_dict, parameters=self.aruco_param)
                
                if ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedMarkers(img, corners)
                    
                    # Pose estimation
                    if hasattr(cv2.aruco, 'estimatePoseSingleMarkers'):
                        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.cfg.marker_length, self.K, self.dist)
                        rvec = rvecs[0][0]
                        tvec = tvecs[0][0]
                    else:
                        # OpenCV 4.7+ fallback using solvePnP
                        marker_length = self.cfg.marker_length
                        marker_points = np.array([
                            [-marker_length / 2,  marker_length / 2, 0],
                            [ marker_length / 2,  marker_length / 2, 0],
                            [ marker_length / 2, -marker_length / 2, 0],
                            [-marker_length / 2, -marker_length / 2, 0]
                        ], dtype=np.float32)
                        _, rvec, tvec = cv2.solvePnP(marker_points, corners[0][0], self.K, self.dist, False, cv2.SOLVEPNP_IPPE_SQUARE)
                    
                    cv2.drawFrameAxes(img, self.K, self.dist, rvec, tvec, self.cfg.marker_length * 0.5)
                    
                    R_mat, _ = cv2.Rodrigues(rvec)
                    cam2qr = np.eye(4)
                    cam2qr[:3, :3] = R_mat
                    cam2qr[:3, 3] = tvec.flatten()

                # Overlay Text
                cv2.putText(img, f"Samples: {len(self.list_tf_base2ee)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if base2ee is None:
                    cv2.putText(img, "Waiting for /end_pose...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif cam2qr is None:
                    cv2.putText(img, "ArUco Not Detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(img, "Ready to Sample (Press 's')", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                cv2.imshow('Piper Hand-Eye Cailbration', img)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):
                    if base2ee is not None and cam2qr is not None:
                        self.list_tf_base2ee.append(base2ee.copy())
                        self.list_tf_cam2qr.append(cam2qr.copy())
                        
                        euler_ee = R.from_matrix(base2ee[:3,:3]).as_euler('xyz', degrees=True)
                        euler_cam = R.from_matrix(cam2qr[:3,:3]).as_euler('xyz', degrees=True)
                        
                        print(f"Sample #{len(self.list_tf_base2ee)} saved!")
                        print(f"  -> 로봇 팔(EE) 현재 각도(초단위): {euler_ee.round(2)}")
                        print(f"  -> 카메라(Cam) 현재 각도(초단위): {euler_cam.round(2)}")
                    else:
                        print("Cannot sample! Make sure ArUco is visible and Robot Pose is received.")

                elif key == ord('c'):
                    if len(self.list_tf_base2ee) < 3:
                        print("Need at least 3 samples to calculate! (10-20 recommended)")
                        continue
                    
                    print("\nCalculating Calibration using OpenCV calibrateHandEye...")
                    
                    R_gripper2base = []
                    t_gripper2base = []
                    R_target2cam = []
                    t_target2cam = []
                    
                    for T_ee in self.list_tf_base2ee:
                        R_gripper2base.append(T_ee[:3, :3])
                        t_gripper2base.append(T_ee[:3, 3].reshape(3, 1))
                        
                    for T_cam in self.list_tf_cam2qr:
                        R_target2cam.append(T_cam[:3, :3])
                        t_target2cam.append(T_cam[:3, 3].reshape(3, 1))
                        
                    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                        R_gripper2base, t_gripper2base,
                        R_target2cam, t_target2cam,
                        method=cv2.CALIB_HAND_EYE_TSAI
                    )
                    
                    R_final, t_final = R_cam2gripper, t_cam2gripper
                    
                    if R_final is not None:
                        print("\n================ CALIBRATION SUCCESS ================")
                        print("Copy these values into Logic_inte_wm.py and basket2.py:\n")
                        
                        r_str = ",\n        ".join([f"[{r[0]:.6f}, {r[1]:.6f}, {r[2]:.6f}]" for r in R_final.real])
                        t_str = ",\n                       ".join([f"[{float(t):.6f}]" for t in t_final.flatten()])

                        
                        code_snippet = f"""    self.R_cal = np.array([
        {r_str}
    ])
    self.t_cal = np.array([{t_str}])
"""
                        print(code_snippet)
                        print("=====================================================\n")

                        # Save to file
                        with open("piper_ee2cam.txt", "w") as f:
                            f.write(code_snippet)
                        print("Result also saved to 'piper_ee2cam.txt'.")
                    else:
                        print("Calibration failed. Please collect more diverse samples.")

                elif key == ord('q'):
                    print("Exiting...")
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            self.ros_node.destroy_node()

def parse_args():
    p = argparse.ArgumentParser(description="Piper ArUco Hand-Eye Collector (ROS 2)")
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--marker-length', type=float, default=0.04)
    p.add_argument('--dictionary', choices=['4X4_50', '5X5_100', '6X6_250', '7X7_1000'], default='6X6_250')
    return p.parse_args()

def main():
    rclpy.init()
    cfg = parse_args()
    collector = PiperTFCollector(cfg)
    collector.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
