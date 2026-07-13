"""
ROS2 Action 노드: 외부 감지 파이프라인 라우팅
- mode == "wm"       → 기존 run_laundry.py 를 그대로 호출 ( --mode wm --json )
- mode == "basket"   → run_basket.py --mode basket --ckpt ... --intr_json ... --json
- mode == "failure"  → run_failure.py --mode failure --ckpt ... --intr_json ... --json
- mode == "failure_bg" → run_failure.py --mode failure_bg --ckpt ... --intr_json ... --json

{"success": bool, "centers": [[x,y,z], ...]}


"""

import os
import sys
import json
import subprocess
import time as _time
import argparse
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from geometry_msgs.msg import Pose
from piper_with_gripper_moveit.action import TargetPose

# ====== 경로/옵션 기본값 ======
PYTHON_BIN = sys.executable or "python3"
TIMEOUT_SEC = 660.0

RUN_LAUNDRY_DIR = "/home/dyros/mcy_ws/Laundry_decision/run_laundry"
RUN_LAUNDRY_PY  = os.path.join(RUN_LAUNDRY_DIR, "run_laundry.py")

RUN_FAILURE_DIR = "/home/dyros/mcy_ws/Laundry_decision/run_bf"
RUN_FAILURE_PY  = os.path.join(RUN_FAILURE_DIR, "run_failure_backup.py")
RUN_FAILURE_EH_PY = os.path.join(RUN_FAILURE_DIR, "run_failure_eh.py")
RUN_FAILURE_NEW_PY = os.path.join(RUN_FAILURE_DIR, "run_failure_new_backup.py")

RUN_BASKET_DIR  = "/home/dyros/mcy_ws/Laundry_decision/run_bf"
RUN_BASKET_PY   = os.path.join(RUN_BASKET_DIR, "run_basket_backup.py")
RUN_BASKET_EH_PY = os.path.join(RUN_BASKET_DIR, "run_basket_eh_backup.py")

# 공용 ckpt / 내재 파라미터(필요 시 도메인별로 나눠도 됨)
CKPT_PATH_F = "/home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/unet_best_failure_new.pt"
CKPT_PATH_B = "/home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/unet_best_basket_new.pt"
# [FIX] New RGB Gate Model Path
CKPT_PATH_RGB = "/home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/rgb_only.pth"

INTR_JSON  = "/home/dyros/mcy_ws/Laundry_decision/run_bf/intrinsics_depth_640x480.json" # Corrected to depth JSON as used in target_pose_server

# 사용가능
FAILURE_ARGS = [
    # "--size", "384",
    # "--zmax", "0.5",
    # "--thr", "0.45",
    # "--diff_mode", "blend",
    # "--vote", "--vote_k", "5", "--vote_min_area", "150", # Reduced vote_k to 5, removed require_curr
    # "--vote_min_dt", "3", "--vote_require_curr",
    # "--stale_sec", "1.0",
    "--save_dir", "/home/dyros/mcy_ws/Laundry_decision/run_bf/_dbg",
    "--save_best_only", "--prefix"
]


def _last_json_line(text: str) -> dict:
    """stdout에서 마지막 JSON 라인을 파싱.
    - 뒤에서부터 '{' 로 시작하고 '}' 로 끝나는 라인을 찾아 파싱
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") and ln.endswith("}"):
            try:
                return json.loads(ln)
            except Exception:
                pass
    # 마지막 라인 한 번 더 시도
    if lines:
        try:
            return json.loads(lines[-1])
        except Exception:
            pass
    raise json.JSONDecodeError("no json line found", doc=(text[-500:] if text else ""), pos=0)


_cb_start_time: float = 0.0  # execute_callback 시작 시각 (GAP_TIMER 용)


def run_json(cmd: List[str], cwd: str, timeout: float = TIMEOUT_SEC) -> dict:
    """외부 프로세스 실행 후 JSON 결과를 파싱해서 반환."""
    global _cb_start_time
    script_name = os.path.basename(cmd[1]) if len(cmd) > 1 else "?"
    if _cb_start_time > 0:
        print(f"[GAP_TIMER] cb_to_subprocess: {_time.time()-_cb_start_time:.2f}s")
    print("[node] exec:", " ".join(cmd))
    t0 = _time.time()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=None,  # stderr 캡처 해제: loguru 로그가 터미널에 실시간으로 출력되게 함
        timeout=timeout,
        text=True,
    )
    dt = _time.time() - t0
    out = proc.stdout or ""
    print(f"[INFER_TIMER] {script_name} : {dt:.2f}s")
    print("[node] output tail:\n" + "\n".join(out.splitlines()[-20:]))
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=out)
    return _last_json_line(out)


class TargetPoseServer(Node):
    def __init__(self, device: str = 'cpu'):
        super().__init__('target_pose_server')
        self.device = device
        self.get_logger().info(f"=== NODE2 TARGET POSE SERVER STARTED (device={self.device}) ===")
        self._action_server = ActionServer(
            self, TargetPose, 'target_pose', self.execute_callback
        )

    async def execute_callback(self, goal_handle):
        global _cb_start_time
        _tcb0 = _time.time()
        _cb_start_time = _tcb0
        self.get_logger().info('target_pose 요청 받음')

        if not goal_handle.request.start:
            self.get_logger().warn('start=False로 요청됨')
            goal_handle.abort()
            return TargetPose.Result(success=False)

        mode = goal_handle.request.mode  # "wm" | "basket" | "failure" | "failure_bg"
        self.get_logger().info(f"세탁물 감지 실행: mode={mode}")

        try:
            # ---------- 세탁기(기존 파이프라인) ----------
            if mode == "wm":
                if not os.path.isfile(RUN_LAUNDRY_PY):
                    self.get_logger().error(f"run_laundry.py 없음: {RUN_LAUNDRY_PY}")
                    goal_handle.abort()
                    return TargetPose.Result(success=False)
                # cmd = [PYTHON_BIN, RUN_LAUNDRY_PY, "--mode", "wm", "--json", "--show"]
                cmd = [
                        PYTHON_BIN, RUN_LAUNDRY_PY, 
                        "--mode", "wm", 
                        "--json", 
                        "--show",
                        "--rgb_ckpt", "/home/dyros/mcy_ws/Laundry_decision/run_laundry/models/best.pt",
                        "--rgb_thr", "0.60",
                        "--rgb_low_thr", "0.35",
                        "--device", self.device
                    ]
                result = run_json(cmd, cwd=RUN_LAUNDRY_DIR)
                
                #self.get_logger().info(f"[wm] result={result}")
            # ---------- 바구니 파이프라인 ----------
            elif mode == "basket":
                if not os.path.isfile(RUN_BASKET_PY):
                    self.get_logger().error(f"run_basket.py 없음: {RUN_BASKET_PY}")
                    goal_handle.abort()
                    return TargetPose.Result(success=False)
                cmd = [
                    PYTHON_BIN, RUN_BASKET_PY,
                    "--mode", "basket",
                    "--ckpt", CKPT_PATH_B,
                    "--intr_json", "/home/dyros/mcy_ws/Laundry_decision/run_bf/intrinsics_color_640x480.json",
                    "--json",
                    # "--show",
                    "--diff_mode", "blend",
                    "--bg_path", "/home/dyros/mcy_ws/Laundry_decision/run_bf/laundry_bg/bg_basket.npy",
                    "--device", self.device,
                    "--zmax_basket", "0.35",
                    "--no_basket_mask",
                    "--max_wait_sec", "5",
                    *FAILURE_ARGS,
                    mode
                ]
                result = run_json(cmd, cwd=RUN_BASKET_DIR)

                self.get_logger().info(f"[basket] result={result}")
            elif mode == "basket_eh":
                if not os.path.isfile(RUN_BASKET_EH_PY):
                    self.get_logger().error(f'run_basket_eh.py 없음: {RUN_BASKET_EH_PY}')
                    goal_handle.abort()
                    return TargetPose.Result(success=False)
                cmd = [
                    PYTHON_BIN, RUN_BASKET_EH_PY,
                    "--mode", "basket_eh",
                    "--ckpt", CKPT_PATH_B,
                    "--intr_json", "/home/dyros/mcy_ws/Laundry_decision/run_bf/intrinsics_color_640x480.json",
                    "--json",
                    # "--show", # Ensure show is permitted if in args
                    "--diff_mode", "bilateral",
                    "--bg_path", "/home/dyros/mcy_ws/Laundry_decision/run_bf/laundry_bg/bg_basket.npy", # Explicit Path
                    "--device", self.device, # Force CPU
                    *FAILURE_ARGS, 
                    mode
                ]
                result = run_json(cmd, cwd=RUN_BASKET_DIR)
                
                self.get_logger().info(f"[basket] result={result}")
            elif mode == "basket_bg_eh":
                if not os.path.isfile(RUN_BASKET_EH_PY):
                    self.get_logger().error(f'run_basket_eh.py 없음: {RUN_BASKET_EH_PY}')
                    goal_handle.abort()
                    return TargetPose.Result(success=False)
                cmd = [
                    PYTHON_BIN, RUN_BASKET_EH_PY,
                    "--mode", "basket_bg_eh",
                    "--ckpt", CKPT_PATH_B,
                    "--intr_json", "/home/dyros/mcy_ws/Laundry_decision/run_bf/intrinsics_color_640x480.json",
                    "--json",
                    # "--show", # Ensure show is permitted if in args
                    "--diff_mode", "bilateral",
                    "--bg_path", "/home/dyros/mcy_ws/Laundry_decision/run_bf/laundry_bg/bg_basket.npy", # Explicit Path
                    "--basket_path", "/home/dyros/mcy_ws/Laundry_decision/run_bf/eh_basket_dbg/mask_41_boundaryfill.png"
                    "--device", self.device, # Force CPU
                    *FAILURE_ARGS, 
                    mode
                ]
                result = run_json(cmd, cwd=RUN_BASKET_DIR)
                
                self.get_logger().info(f"[basket] result={result}")

            # ---------- 실패(배경 기반) 파이프라인 ----------
            elif mode == "failure":
                if not os.path.isfile(RUN_FAILURE_PY):
                    self.get_logger().error(f"run_failure.py 없음: {RUN_FAILURE_PY}")
                    goal_handle.abort()
                    return TargetPose.Result(success=False)
                cmd = [
                    PYTHON_BIN, RUN_FAILURE_PY,
                    "--mode", "failure",
                    "--ckpt", CKPT_PATH_F,
                    "--intr_json", INTR_JSON,
                    "--json",
                    # "--show",
                    "--rgb_ckpt", CKPT_PATH_RGB, # ADDED: New model
                    "--device", self.device, # ADDED: CPU Force
                    *FAILURE_ARGS,
                    mode
                ]
                result = run_json(cmd, cwd=RUN_FAILURE_DIR)
                self.get_logger().info(f"[failure] result={result}")
            # ---------- 실패: 배경 저장 ----------
            elif mode == "failure_bg":
                if not os.path.isfile(RUN_FAILURE_PY):
                    self.get_logger().error(f"run_failure.py 없음: {RUN_FAILURE_PY}")
                    goal_handle.abort()
                    return TargetPose.Result(success=False)
                cmd = [
                    PYTHON_BIN, RUN_FAILURE_PY,
                    "--mode", "failure_bg",
                    "--ckpt", CKPT_PATH_F,
                    "--intr_json", INTR_JSON,
                    "--json",
                    # "--show",
                    "--device", self.device, # ADDED: CPU Force
                ]
                result = run_json(cmd, cwd=RUN_FAILURE_DIR)
                self.get_logger().info(f"[failure_bg] result={result}")

            elif mode == "failure_eh":
                if not os.path.isfile(RUN_FAILURE_EH_PY):
                    self.get_logger().error(f"run_failure_eh.py 없음: {RUN_FAILURE_EH_PY}")
                    goal_handle.abort()
                    return TargetPose.Result(success=False)

                cmd = [
                    PYTHON_BIN, RUN_FAILURE_EH_PY,
                    "--mode", "failure_eh",
                    "--ckpt", CKPT_PATH_F,
                    "--rgb_ckpt", CKPT_PATH_RGB,
                    "--intr_json", INTR_JSON,
                    "--bg_path", "/home/dyros/mcy_ws/Laundry_decision/run_bf/laundry_bg/bg_overcome2.npy",
                    "--size", "384",
                    "--zmax", "0.5",
                    "--thr", "0.45",
                    "--diff_mode", "blend",
                    "--vote", "--vote_k", "10", "--vote_min_area", "150", "--vote_min_dt", "3", "--vote_require_curr",
                    "--stale_sec", "1.0",
                    "--save_dir", "/home/dyros/mcy_ws/Laundry_decision/run_bf/eh_dbg",
                    "--save_best_only",
                    "--prefix", "failure_eh",  # <-- 고정
                    "--json",
                    # "--show",
                    "--dbg_maps", "--dbg_every", "10",
                    "--thr_sweep", "0.35,0.45,0.55",
                    "--device", self.device
                ]

                # EH 모드에서는 FAILURE_ARGS를 붙이지 않는다.
                # if FAILURE_ARGS:
                #     cmd += FAILURE_ARGS

                result = run_json(cmd, cwd=RUN_FAILURE_DIR)
                self.get_logger().info(f"[failure_eh] result={result}")

            elif mode == "failure_eh_strict":
                if not os.path.isfile(RUN_FAILURE_EH_PY):
                    self.get_logger().error(f"run_failure_eh.py 없음: {RUN_FAILURE_EH_PY}")
                    goal_handle.abort()
                    return TargetPose.Result(success=False)

                cmd = [
                    PYTHON_BIN, RUN_FAILURE_EH_PY,
                    "--mode", "failure_eh",
                    "--ckpt", CKPT_PATH_F,
                    "--rgb_ckpt", CKPT_PATH_RGB,
                    "--intr_json", INTR_JSON,
                    "--bg_path", "/home/dyros/mcy_ws/Laundry_decision/run_bf/laundry_bg/bg_overcome2.npy",
                    "--size", "384",
                    "--zmax", "0.5",
                    "--thr", "0.45",
                    "--diff_mode", "blend",
                    "--vote", "--vote_k", "10", "--vote_min_area", "300", "--vote_min_dt", "5", "--vote_require_curr",
                    "--stale_sec", "1.0",
                    "--save_dir", "/home/dyros/mcy_ws/Laundry_decision/run_bf/eh_dbg",
                    "--save_best_only",
                    "--prefix", "failure_eh_strict",
                    "--json",
                    # "--show",
                    "--device", self.device
                ]

                result = run_json(cmd, cwd=RUN_FAILURE_DIR)
                self.get_logger().info(f"[failure_eh_strict] result={result}")

            elif mode == "failure_rgb_diff":
                if not os.path.isfile(RUN_FAILURE_NEW_PY):
                    self.get_logger().error(f"run_failure_new.py 없음: {RUN_FAILURE_NEW_PY}")
                    goal_handle.abort()
                    return TargetPose.Result(success=False)

                cmd = [
                    PYTHON_BIN, RUN_FAILURE_NEW_PY,
                    "--mode", "failure_rgb_diff",
                    "--bg_path", "/home/dyros/mcy_ws/Laundry_decision/run_bf/laundry_bg/bg_overcome2.npy",
                    "--ckpt", CKPT_PATH_F,
                    "--intr_json", INTR_JSON,
                    "--size", "384",
                    "--zmax", "0.5",
                    "--thr", "0.45",
                    "--diff_mode", "blend",
                    "--vote", "--vote_k", "10", "--vote_min_area", "150", "--vote_min_dt", "3", "--vote_require_curr",
                    "--stale_sec", "1.0",
                    "--save_dir", "/home/dyros/mcy_ws/Laundry_decision/run_bf/eh_dbg",
                    "--save_best_only",
                    "--prefix", "failure_rgb_diff",
                    "--json",
                    # "--show",
                    "--device", self.device
                ]

                result = run_json(cmd, cwd=RUN_FAILURE_DIR)
                self.get_logger().info(f"[failure_rgb_diff] result={result}")

            elif mode == "failure_bg_eh":
                if not os.path.isfile(RUN_FAILURE_NEW_PY):
                    self.get_logger().error(f"run_failure_new.py 없음: {RUN_FAILURE_NEW_PY}")
                    goal_handle.abort()
                    return TargetPose.Result(success=False)

                cmd = [
                    PYTHON_BIN, RUN_FAILURE_NEW_PY,
                    "--mode", "failure_bg_eh",
                    "--bg_frames", "20",
                    "--bg_path", "/home/dyros/mcy_ws/Laundry_decision/run_bf/laundry_bg/bg_overcome2.npy",
                    "--ckpt", CKPT_PATH_F,
                    "--intr_json", INTR_JSON,
                    "--size", "384",
                    "--zmax", "0.5",
                    "--save_dir", "/home/dyros/mcy_ws/Laundry_decision/run_bf/_dbg",
                    "--prefix", "failure_bg_eh",  # <-- 고정
                    "--json",
                    # "--show",
                    "--device", self.device
                ]

                # EH 모드는 FAILURE_ARGS를 합치지 않는다.
                # if FAILURE_ARGS:
                #     cmd += FAILURE_ARGS

                result = run_json(cmd, cwd=RUN_FAILURE_DIR)
                self.get_logger().info(f"[failure_bg_eh] result={result}")
                


            else:
                self.get_logger().error(f"알 수 없는 mode='{mode}'")
                goal_handle.abort()
                return TargetPose.Result(success=False)

            # ---------- 결과 처리 공통 ----------
            # debg dyd
            # self.get_logger().info(f"[{mode}] result={result}")
            centers = result.get("centers") or []
            
            # [Two-Stage Check] Handle Explicit Empty for FAILURE/EH/WM modes
            if mode in ["failure", "failure_eh", "wm","basket"]:
                 reason = result.get("reason", "")
                 if reason == "rgb_gate_empty":
                    self.get_logger().info(f"[{mode}] RGB Gate 확인: Empty Confirmed (Sentinel -99.0)")
                    sentinel_pose = Pose()
                    sentinel_pose.position.z = -99.0
                    res = TargetPose.Result(success=True, target_pose=sentinel_pose)
                    goal_handle.succeed()
                    return res


            print("********************************************************************")
            print(len(centers))
            print("********************************************************************")
            

            ok = bool(result.get("success", False)) and len(centers) > 0
            if not ok:
                self.get_logger().warn("의류 미검출: Sentinel -99.0 반환")
                sentinel_pose = Pose()
                sentinel_pose.position.z = -99.0
                res = TargetPose.Result(success=True, target_pose=sentinel_pose)
                goal_handle.succeed()
                return res
            
            cx, cy, cz = map(float, centers[0])

            target_pose = Pose()
            target_pose.position.x = cx
            target_pose.position.y = cy
            target_pose.position.z = cz
            target_pose.orientation.w = 1.0
            
            section_col = None
            
            # Original Logic Preserved
            if mode == "failure":
                colu = result.get("occupied_cells")
                if colu:
                    col1 = colu[0]
                    try:
                        col2 = colu[1]
                    except:
                        col2 = None
                        
            out = TargetPose.Result()
            out.success = True
            out.target_pose = target_pose
            # if mode == "failure":
            #     out.message = result.get('section_cols')
                
            # Result에 'col' 필드가 있으면 사용, 없으면 message로 전달
            if mode == "failure" and 'colu' in locals() and colu is not None:
                if hasattr(out, "col"):
                    out.col = col1
                    try:
                        out.col2 = float(col2)
                    except:
                        pass
                # elif hasattr(out, "message"):
                #     out.message = f"col={section_col}"
                else:
                    # 필드가 전혀 없으면 로그만 남김
                    self.get_logger().warn("Result에 col/message 필드가 없어 col 값을 담지 못했습니다.")

            self.get_logger().info(f"[{mode}] 좌표 반환: x={cx:.3f}, y={cy:.3f}, z={cz:.3f}")
            if section_col is not None:
                self.get_logger().info(f"[{mode}] section_col={section_col}")
            print(f"[GAP_TIMER] cb_total (cb~succeed): {_time.time()-_tcb0:.2f}s")
            goal_handle.succeed()
            return out

        except subprocess.TimeoutExpired:
            self.get_logger().error("감지 프로세스 타임아웃")
            goal_handle.abort()
            return TargetPose.Result(success=False)

        except subprocess.CalledProcessError as e:
            msg = e.output if isinstance(e.output, str) else (e.output or b"").decode(errors='ignore')
            self.get_logger().error(f"감지 중 예외 발생(exit {e.returncode}):\n{msg}")
            goal_handle.abort()
            return TargetPose.Result(success=False)

        except json.JSONDecodeError as e:
            self.get_logger().error(f"JSON 파싱 실패: {e}")
            goal_handle.abort()
            return TargetPose.Result(success=False)


def main(args=None):
    parser = argparse.ArgumentParser(description='Node2 vision server')
    parser.add_argument('--device', type=str, default='cpu',
                        help='추론 디바이스 (cpu / cuda / cuda:0 등)')
    parser.add_argument('--gpu', action='store_true',
                        help='--device cuda 와 동일 (편의 flag)')
    parsed, ros_args = parser.parse_known_args()
    device = 'cuda' if parsed.gpu else parsed.device

    rclpy.init(args=ros_args or args)
    node = TargetPoseServer(device=device)
    rclpy.spin(node)


if __name__ == '__main__':
    main()
