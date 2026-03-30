#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from geometry_msgs.msg import Pose

#target_pose = Pose.target_pose


import os, sys, subprocess, json
from std_msgs.msg import String

# 외부 패키지 경로 수동 추가
# piper_pkg_path = "/home/cheon/Documents/dyros/piper-mou/piper_ros/src/piper_moveit"
# if piper_pkg_path not in sys.path:
#     sys.path.append(piper_pkg_path)

from piper_with_gripper_moveit.action import TargetPose

class TargetPoseServer(Node):
    def __init__(self):
        super().__init__('target_pose_server')
        self.get_logger().info("=== OLD TARGET POSE SERVER STARTED ===")

        self._action_server = ActionServer(
            self,
            TargetPose,
            'target_pose',
            self.execute_callback
        )

    async def execute_callback(self, goal_handle):
        self.get_logger().info('target_pose 요청 받음')

        if not goal_handle.request.start:
            self.get_logger().warn('start=False로 요청됨')
            goal_handle.abort()
            return TargetPose.Result(success=False)

        mode = goal_handle.request.mode  # "wm" 또는 "basket"
        # mode = "basket"
        
        self.get_logger().info(f"세탁물 감지 실행: mode={mode}")

        try:
            if mode == "wm": 
                result_json = subprocess.check_output(
                            ['python3', 'run_laundry.py', '--mode', mode, '--json', '--rgb_ckpt', '/home/dyros/mcy_ws/Laundry_decision/run_laundry/models/best.pt'], 
                            cwd='/home/dyros/mcy_ws/Laundry_decision/run_laundry',  # Correct path
                            stderr=subprocess.STDOUT,
                            timeout=660.0
                        ).decode('utf-8').strip().split('\n')[-1]

                result_data = json.loads(result_json)
                centers = result_data.get("centers", [])
                print(result_data)
                if not result_data.get("success", False) or len(centers) == 0:
                    self.get_logger().warn("의류 미검출")
                    goal_handle.abort()
                    return TargetPose.Result(success=False)
            elif mode == "basket":
                result_json = subprocess.check_output(
                            ['python3',  '/home/dyros/mcy_ws/Laundry_decision/run_bf/run_basket.py', # Use run_basket.py
                             '--mode', 'basket',
                             '--ckpt', '/home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/unet_best_multi.pt',
                             '--intr_json', '/home/dyros/mcy_ws/Laundry_decision/run_bf/intrinsics_color_640x480.json',
                             '--diff_mode', 'bilateral',
                             '--vote', '--vote_k', '10', '--vote_min_area', '150', '--vote_min_dt', '3', '--vote_require_curr',
                             '--bg_path', '/home/dyros/mcy_ws/Laundry_decision/run_bg/laundry_bg/bg_basket.npy',
                             '--json', '--show'], # Enable 2D visualization
                            cwd='/home/dyros/mcy_ws/Laundry_decision/run_bf',
                            stderr=subprocess.STDOUT,
                            timeout=660.0
                        ).decode('utf-8').strip().split('\n')[-1]

                result_data = json.loads(result_json)
                centers = result_data.get("centers", [])
                print(result_data)
                if not result_data.get("success", False) or len(centers) == 0:
                    self.get_logger().warn("의류 미검출")
                    goal_handle.abort()
                    return TargetPose.Result(success=False)
            elif mode == "basket_eh":
                result_json = subprocess.check_output(
                            ['python3',  '/home/dyros/mcy_ws/Laundry_decision/run_bf/run_basket_eh.py', # Use run_basket.py
                             '--mode', 'basket',
                             '--ckpt', '/home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/unet_best_multi.pt',
                             '--intr_json', '/home/dyros/mcy_ws/Laundry_decision/run_bf/intrinsics_color_640x480.json',
                             '--diff_mode', 'bilateral',
                             '--vote', '--vote_k', '10', '--vote_min_area', '150', '--vote_min_dt', '3', '--vote_require_curr',
                             '--bg_path', '/home/dyros/mcy_ws/Laundry_decision/run_bg/laundry_bg/bg_basket.npy',
                             '--json', '--show'], # Enable 2D visualization
                            cwd='/home/dyros/mcy_ws/Laundry_decision/run_bf',
                            stderr=subprocess.STDOUT,
                            timeout=660.0
                        ).decode('utf-8').strip().split('\n')[-1]

                result_data = json.loads(result_json)
                centers = result_data.get("centers", [])
                print(result_data)
                if not result_data.get("success", False) or len(centers) == 0:
                    self.get_logger().warn("의류 미검출")
                    goal_handle.abort()
                    return TargetPose.Result(success=False)
            elif mode == "failure_bg":
                result_json = subprocess.check_output(
                            ['python3', 'run_failure.py', '--mode', mode, '--json',
                             '--bg_path', '/home/dyros/mcy_ws/Laundry_decision/run_bf/laundry_bg/bg_failure.npy',
                             '--ckpt', '/home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/unet_best.pt',
                             '--intr_json', '/home/dyros/mcy_ws/Laundry_decision/run_bf/intrinsics_depth_640x480.json',
                             '--show'], # Added --show
                            cwd='/home/dyros/mcy_ws/Laundry_decision/run_bf', # Correct path
                            stderr=subprocess.STDOUT,
                            timeout=660.0
                        ).decode('utf-8').strip().split('\n')[-1]

                result_data = json.loads(result_json)
                centers = result_data.get("centers", []) # failure_bg는 centers를 반환하지 않음
                print(result_data)
                if not result_data.get("success", False) or len(centers) == 0:
                    self.get_logger().warn("의류 미검출")
                
                # if not result_data.get("success", False):
                #     self.get_logger().warn("배경 촬영 실패")
                    goal_handle.abort()
                    return TargetPose.Result(success=False)
                # else:
                #     self.get_logger().info(f"배경 촬영 성공: {result_data.get('bg_path')}")
                #     # 성공했으므로 빈 Pose라도 반환해서 Goal을 성공 처리함
                #     res = TargetPose.Result()
                #     res.success = True
                #     res.target_pose = Pose() # Dummy
                #     goal_handle.succeed()
                #     return res
            elif mode == "failure":
                result_json = subprocess.check_output(
                            ['python3', 'run_failure.py', '--mode', mode, '--json',
                             '--ckpt', '/home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/unet_best.pt',
                             '--rgb_ckpt', '/home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/rgb_only.pth',
                             '--intr_json', '/home/dyros/mcy_ws/Laundry_decision/run_bf/intrinsics_depth_640x480.json',
                             '--show'], # Added --show
                            cwd='/home/dyros/mcy_ws/Laundry_decision/run_bf', # Correct path
                            stderr=subprocess.STDOUT,
                            timeout=660.0
                        ).decode('utf-8').strip().split('\n')[-1] 
                
                result_data = json.loads(result_json)
                centers = result_data.get("centers", [])
                print(result_data)

                if not result_data.get("success", False) or len(centers) == 0:
                    # [Two-Stage Check] Handle Explicit Empty
                    reason = result_data.get("reason", "")
                    if reason == "rgb_gate_empty":
                        self.get_logger().info("RGB Gate 확인: Empty Confirmed (Sentinel -99.0)")
                        sentinel_pose = Pose()
                        sentinel_pose.position.z = -99.0
                        res = TargetPose.Result(success=True, target_pose=sentinel_pose)
                        goal_handle.succeed()
                        return res
                    
                    self.get_logger().info("Empty (No clothes or Timeout)") 
                    goal_handle.abort()
                    return TargetPose.Result(success=False)
                else:
                     self.get_logger().info("Found")
            elif mode == "failure_bg_eh":
                result_json = subprocess.check_output(
                            ['python3', 'run_failure_eh.py', '--mode', 'failure_bg_eh', '--json',
                            '--bg_frames', '20',
                            '--bg_path', '/home/dyros/mcy_ws/Laundry_decision/run_bf/laundry_bg/bg_failure.npy',
                            '--ckpt', '/home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/unet_best.pt',
                            '--intr_json', '/home/dyros/mcy_ws/Laundry_decision/run_bf/intrinsics_depth_640x480.json',
                            '--size', '384', '--zmax', '0.5',
                            '--save_dir', '/home/dyros/mcy_ws/Laundry_decision/run_bf/eh_dbg',
                            '--prefix', 'failure_bg',
                            '--show'],
                            cwd='/home/dyros/mcy_ws/Laundry_decision/run_bf', # Correct path
                            stderr=subprocess.STDOUT,
                            timeout=660.0
                        ).decode('utf-8').strip().split('\n')[-1]

                result_data = json.loads(result_json)
                if not result_data.get("success", False):
                    self.get_logger().warn("배경 촬영 실패")
                    goal_handle.abort()
                    return TargetPose.Result(success=False)

                # 성공이면 succeed 처리 (dummy pose)
                res = TargetPose.Result()
                res.success = True
                res.target_pose = Pose()
                goal_handle.succeed()
                return res

            elif mode == "failure_eh":
                cmd = [
                'python3', 'run_failure_eh.py', '--mode', 'failure_eh', '--json',
                '--ckpt', '/home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/unet_best.pt',
                '--rgb_ckpt', '/home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/rgb_only.pth',
                '--intr_json', '/home/dyros/mcy_ws/Laundry_decision/run_bf/intrinsics_depth_640x480.json',
                '--size', '384', '--zmax', '0.5', '--thr', '0.45',
                '--diff_mode', 'blend',
                '--vote', '--vote_k', '10', '--vote_min_area', '150', '--vote_min_dt', '3', '--vote_require_curr',
                '--stale_sec', '1.0',
                '--save_dir', '/home/dyros/mcy_ws/Laundry_decision/run_bf/eh_dbg',
                '--save_best_only', '--prefix', 'failure',
                '--show',
                '--dbg_maps', '--dbg_every', '10', '--thr_sweep', '0.35,0.45,0.55'
                ]
                result_json = subprocess.check_output(
                    cmd,
                    cwd='/home/dyros/mcy_ws/Laundry_decision/run_bf',
                    stderr=subprocess.STDOUT,
                    timeout=660.0
                ).decode('utf-8').strip().split('\n')[-1]

               
                result_data = json.loads(result_json)
                centers = result_data.get("centers", [])
                print(result_data)

                if not result_data.get("success", False) or len(centers) == 0:
                    # [Two-Stage Check] Handle Explicit Empty
                    reason = result_data.get("reason", "")
                    if reason == "rgb_gate_empty":
                        self.get_logger().info("RGB Gate 확인: Empty Confirmed (Sentinel -99.0)")
                        sentinel_pose = Pose()
                        sentinel_pose.position.z = -99.0
                        res = TargetPose.Result(success=True, target_pose=sentinel_pose)
                        goal_handle.succeed()
                        return res
                   
                    self.get_logger().info("Empty (No clothes or Timeout)")
                    goal_handle.abort()
                    return TargetPose.Result(success=False)
                else:
                     self.get_logger().info("Found")
            # 첫 번째 중심 좌표 사용
            center = centers[0]
            target_pose = Pose()
            target_pose.position.x = center[0]
            target_pose.position.y = center[1]
            target_pose.position.z = center[2]
            target_pose.orientation.w = 1.0  # 기본 회전 없음

            result = TargetPose.Result()
            result.success = True
            result.target_pose = target_pose

            self.get_logger().info(f"좌표 반환: x={center[0]:.3f}, y={center[1]:.3f}, z={center[2]:.3f}")
            goal_handle.succeed()
            return result

        except subprocess.TimeoutExpired:
            self.get_logger().error("감지 프로세스 타임아웃")
            goal_handle.abort()
            return TargetPose.Result(success=False)

        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"감지 중 예외 발생 (stderr):\n{e.output.decode()}")
            goal_handle.abort()
            return TargetPose.Result(success=False)

def main(args=None):
    rclpy.init(args=args)
    node = TargetPoseServer()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
