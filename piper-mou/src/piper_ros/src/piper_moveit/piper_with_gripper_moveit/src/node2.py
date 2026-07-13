"""
ROS2 Action 노드: 외부 감지 파이프라인 라우팅
- mode == "wm"       → run_laundry.py (best.pt)
- mode == "basket"   → run_basket.py (unet_best_multi.pt)
- mode == "failure*" → run_failure_eh.py (unet_best.pt + rgb_only.pth)
- mode == "failure_rgb_diff" / "failure_bg_eh" → run_failure_new.py

모든 모델을 __init__에서 사전 로드하여 매 호출 시 torch.load 생략.
subprocess 대신 in-process 호출로 Python 인터프리터 기동 오버헤드 제거.
"""

import os
import sys
import json
import io
import contextlib
import threading
import argparse
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Pose
from piper_with_gripper_moveit.action import TargetPose

# ====== 경로 상수 ======
PYTHON_BIN = sys.executable or "python3"
TIMEOUT_SEC = 660.0

RUN_LAUNDRY_DIR = "/home/dyros/mcy_ws/Laundry_decision/run_laundry"
RUN_FAILURE_DIR = "/home/dyros/mcy_ws/Laundry_decision/run_bf"

CKPT_PATH_F   = "/home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/unet_best.pt"
CKPT_PATH_B   = "/home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/unet_best_multi.pt"
CKPT_PATH_RGB = "/home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/rgb_only.pth"
CKPT_PATH_WM  = "/home/dyros/mcy_ws/Laundry_decision/run_laundry/models/best.pt"
INTR_JSON     = "/home/dyros/mcy_ws/Laundry_decision/run_bf/intrinsics_depth_640x480.json"
INTR_COLOR    = "/home/dyros/mcy_ws/Laundry_decision/run_bf/intrinsics_color_640x480.json"

FAILURE_ARGS_LIST = [
    "--save_dir", "/home/dyros/mcy_ws/Laundry_decision/run_bf/_dbg",
    "--save_best_only", "--prefix",
]


def _last_json_line(text: str) -> dict:
    """stdout에서 마지막 JSON 라인을 파싱."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") and ln.endswith("}"):
            try:
                return json.loads(ln)
            except Exception:
                pass
    if lines:
        try:
            return json.loads(lines[-1])
        except Exception:
            pass
    raise json.JSONDecodeError("no json line found", doc=(text[-500:] if text else ""), pos=0)


# ── subprocess fallback (사전 로드 실패 시 사용) ──────────────────────
def run_json(cmd, cwd, timeout=TIMEOUT_SEC) -> dict:
    """외부 프로세스 실행 후 JSON 결과 파싱."""
    print("[node] exec:", " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=None, timeout=timeout, text=True,
    )
    out = proc.stdout or ""
    print("[node] output tail:\n" + "\n".join(out.splitlines()[-20:]))
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=out)
    return _last_json_line(out)


class TargetPoseServer(Node):
    def __init__(self, device: str = 'cpu'):
        super().__init__('target_pose_server')
        self.add_basket = self.declare_parameter('add_basket', True).value
        self.device = device
        self.get_logger().info(f"=== NODE2 TARGET POSE SERVER STARTED (add_basket={self.add_basket}, device={self.device}) ===")

        # vision 모듈 사전 로드
        self._mod_failure_eh  = None
        self._mod_basket      = None
        self._mod_failure_new = None
        self._mod_laundry     = None
        self._call_lock = threading.Lock()  # in-process 호출은 순차 실행
        self._preload_all()

        self._action_server = ActionServer(
            self, TargetPose, 'target_pose', self.execute_callback,
            callback_group=ReentrantCallbackGroup()
        )

    # ── 모델 사전 로드 ───────────────────────────────────────────────
    def _preload_all(self):
        """모든 vision 모듈을 import하고 모델 가중치를 메모리에 로드."""
        # sys.path에 vision 스크립트 디렉터리 추가
        for d in (RUN_FAILURE_DIR, RUN_LAUNDRY_DIR):
            if d not in sys.path:
                sys.path.insert(0, d)

        # run_failure_eh (failure_eh, failure_eh_strict 모드)
        try:
            import run_failure_eh as _m
            _m.preload(CKPT_PATH_F, CKPT_PATH_RGB, device=self.device)
            self._mod_failure_eh = _m
            self.get_logger().info(f"[preload] run_failure_eh 완료 (device={self.device})")
        except Exception as e:
            self.get_logger().error(f"[preload] run_failure_eh 실패: {e}")

        # run_basket (basket, basket_eh 모드)
        try:
            import run_basket as _m
            _m.preload(CKPT_PATH_B, device=self.device)
            self._mod_basket = _m
            self.get_logger().info(f"[preload] run_basket 완료 (device={self.device})")
        except Exception as e:
            self.get_logger().error(f"[preload] run_basket 실패: {e}")

        # run_failure_new (failure_rgb_diff, failure_bg_eh 모드)
        try:
            import run_failure_new as _m
            _m.preload(CKPT_PATH_F, device=self.device)
            self._mod_failure_new = _m
            self.get_logger().info(f"[preload] run_failure_new 완료 (device={self.device})")
        except Exception as e:
            self.get_logger().error(f"[preload] run_failure_new 실패: {e}")

        # run_laundry (wm 모드)
        try:
            old_cwd = os.getcwd()
            os.chdir(RUN_LAUNDRY_DIR)
            import run_laundry as _m
            _m.preload(CKPT_PATH_WM, device=self.device)
            self._mod_laundry = _m
            os.chdir(old_cwd)
            self.get_logger().info("[preload] run_laundry 완료")
        except Exception as e:
            self.get_logger().error(f"[preload] run_laundry 실패: {e}")
            try:
                os.chdir(old_cwd)
            except Exception:
                pass

    # ── in-process 호출 헬퍼 ─────────────────────────────────────────
    def _call_inprocess(self, module, argv_list: list, cwd: str) -> dict:
        """
        module.main()을 in-process로 호출.
        sys.argv를 주입하고 stdout을 캡처하여 JSON 결과를 반환.
        """
        import time as _time
        with self._call_lock:
            old_argv = sys.argv[:]
            old_cwd  = os.getcwd()
            sys.argv = [module.__file__] + argv_list
            os.chdir(cwd)
            capture = io.StringIO()
            _t0 = _time.time()
            try:
                with contextlib.redirect_stdout(capture):
                    module.main()
            except SystemExit:
                pass
            except Exception as e:
                self.get_logger().error(f"[inprocess] 예외: {e}")
            finally:
                _infer_dt = _time.time() - _t0
                sys.argv = old_argv
                os.chdir(old_cwd)
                # 예외 등으로 pipeline이 정리되지 않았을 경우 강제 해제
                _pl = getattr(module, '_active_pipeline', None)
                if _pl is not None:
                    try:
                        _pl.stop()
                    except Exception:
                        pass
                    module._active_pipeline = None
            output = capture.getvalue()
        self.get_logger().info(
            f"[INFER_TIMER] {os.path.basename(module.__file__)} : {_infer_dt:.2f}s  (device={self.device})"
        )
        print("[node] output tail:\n" + "\n".join(output.splitlines()[-20:]))
        return _last_json_line(output)

    def _call_laundry_inprocess(self, mode: str, show: bool, argv_list: list) -> dict:
        """run_laundry는 run() 직접 호출 (argparse 우회)."""
        import time as _time
        with self._call_lock:
            old_cwd = os.getcwd()
            os.chdir(RUN_LAUNDRY_DIR)
            capture = io.StringIO()
            _t0 = _time.time()
            try:
                with contextlib.redirect_stdout(capture):
                    self._mod_laundry.run(
                        mode=mode,
                        json_output=True,
                        show=show,
                        device=self.device,
                        rgb_ckpt=CKPT_PATH_WM,
                    )
            except SystemExit:
                pass
            except Exception as e:
                self.get_logger().error(f"[inprocess/laundry] 예외: {e}")
            finally:
                _infer_dt = _time.time() - _t0
                os.chdir(old_cwd)
            output = capture.getvalue()
        self.get_logger().info(
            f"[INFER_TIMER] run_laundry.run(mode={mode}) : {_infer_dt:.2f}s  (device={self.device})"
        )
        print("[node] output tail:\n" + "\n".join(output.splitlines()[-20:]))
        return _last_json_line(output)

    # ── 공통 결과 처리 ───────────────────────────────────────────────
    def _build_result(self, result: dict, mode: str, goal_handle):
        """result dict → TargetPose.Result. 공통 좌표 처리."""
        centers = result.get("centers") or []

        if mode in ["failure", "failure_eh", "wm", "basket"]:
            if result.get("reason") == "rgb_gate_empty":
                self.get_logger().info(f"[{mode}] RGB Gate: Empty Confirmed → Sentinel -99.0")
                sentinel_pose = Pose()
                sentinel_pose.position.z = -99.0
                goal_handle.succeed()
                return TargetPose.Result(success=True, target_pose=sentinel_pose)

        ok = bool(result.get("success", False)) and len(centers) > 0
        if not ok:
            self.get_logger().warn("의류 미검출: Sentinel -99.0 반환")
            sentinel_pose = Pose()
            sentinel_pose.position.z = -99.0
            goal_handle.succeed()
            return TargetPose.Result(success=True, target_pose=sentinel_pose)

        cx, cy, cz = map(float, centers[0])
        detected_area = result.get("area", 9999)
        AREA_MIN = 12000
        if detected_area < AREA_MIN and mode.endswith("_strict"):
            self.get_logger().warn(f"[{mode}] 클러스터 너무 작음 (area={detected_area}) → Sentinel")
            sentinel_pose = Pose()
            sentinel_pose.position.z = -99.0
            goal_handle.succeed()
            return TargetPose.Result(success=True, target_pose=sentinel_pose)
        elif detected_area < AREA_MIN:
            self.get_logger().warn(f"[{mode}] 클러스터 작음 (area={detected_area}), 좌표 그대로 사용")

        target_pose = Pose()
        target_pose.position.x = cx
        target_pose.position.y = cy
        target_pose.position.z = cz
        target_pose.orientation.w = 1.0

        out = TargetPose.Result()
        out.success = True
        out.target_pose = target_pose

        if mode == "failure":
            colu = result.get("occupied_cells")
            if colu:
                try:
                    if hasattr(out, "col"):
                        out.col = colu[0]
                        if len(colu) > 1:
                            try:
                                out.col2 = float(colu[1])
                            except Exception:
                                pass
                except Exception:
                    pass

        self.get_logger().info(f"[{mode}] 좌표 반환: x={cx:.3f}, y={cy:.3f}, z={cz:.3f}")
        goal_handle.succeed()
        return out

    # ── 메인 콜백 ────────────────────────────────────────────────────
    def execute_callback(self, goal_handle):
        self.get_logger().info('target_pose 요청 받음')

        if not goal_handle.request.start:
            self.get_logger().warn('start=False로 요청됨')
            goal_handle.abort()
            return TargetPose.Result(success=False)

        mode = goal_handle.request.mode
        self.get_logger().info(f"세탁물 감지 실행: mode={mode}")

        try:
            result = self._dispatch(mode)
            return self._build_result(result, mode, goal_handle)

        except json.JSONDecodeError as e:
            self.get_logger().error(f"JSON 파싱 실패: {e}")
            goal_handle.abort()
            return TargetPose.Result(success=False)
        except subprocess.TimeoutExpired:
            self.get_logger().error("감지 프로세스 타임아웃")
            goal_handle.abort()
            return TargetPose.Result(success=False)
        except subprocess.CalledProcessError as e:
            msg = e.output if isinstance(e.output, str) else (e.output or b"").decode(errors='ignore')
            self.get_logger().error(f"감지 중 예외 발생(exit {e.returncode}):\n{msg}")
            goal_handle.abort()
            return TargetPose.Result(success=False)
        except Exception as e:
            self.get_logger().error(f"감지 중 예외: {e}")
            goal_handle.abort()
            return TargetPose.Result(success=False)

    def _dispatch(self, mode: str) -> dict:
        """mode별 vision 추론 실행 → result dict."""

        # ── wm ──────────────────────────────────────────────────────
        if mode == "wm":
            if self._mod_laundry is not None:
                return self._call_laundry_inprocess("wm", show=True, argv_list=[])
            # fallback: subprocess
            cmd = [
                PYTHON_BIN,
                os.path.join(RUN_LAUNDRY_DIR, "run_laundry.py"),
                "--mode", "wm", "--json", "--show",
                "--rgb_ckpt", CKPT_PATH_WM,
                "--rgb_thr", "0.60", "--rgb_low_thr", "0.30", "--device", self.device,
            ]
            return run_json(cmd, RUN_LAUNDRY_DIR)

        # ── basket ──────────────────────────────────────────────────
        elif mode == "basket":
            if self._mod_basket is not None:
                argv = [
                    "--mode", "basket", "--ckpt", CKPT_PATH_B,
                    "--intr_json", INTR_COLOR, "--json", "--show",
                    "--diff_mode", "bilateral",
                    "--bg_path", "/home/dyros/mcy_ws/Laundry_decision/run_bf/laundry_bg/bg_basket.npy",
                    "--device", self.device,
                    *FAILURE_ARGS_LIST, "basket",
                ]
                if not self.add_basket:
                    argv.append("--no_basket_mask")
                return self._call_inprocess(self._mod_basket, argv, RUN_FAILURE_DIR)
            cmd = [
                PYTHON_BIN, os.path.join(RUN_FAILURE_DIR, "run_basket.py"),
                "--mode", "basket", "--ckpt", CKPT_PATH_B,
                "--intr_json", INTR_COLOR, "--json", "--show",
                "--diff_mode", "bilateral",
                "--bg_path", "/home/dyros/mcy_ws/Laundry_decision/run_bf/laundry_bg/bg_basket.npy",
                "--device", self.device, *FAILURE_ARGS_LIST, "basket",
            ]
            return run_json(cmd, RUN_FAILURE_DIR)

        # ── basket_eh (run_basket.py로 처리, basket 모드 동일) ────────
        elif mode == "basket_eh":
            if self._mod_basket is not None:
                argv = [
                    "--mode", "basket", "--ckpt", CKPT_PATH_B,
                    "--intr_json", INTR_COLOR, "--json", "--show",
                    "--diff_mode", "bilateral",
                    "--bg_path", "/home/dyros/mcy_ws/Laundry_decision/run_bf/laundry_bg/bg_basket.npy",
                    "--device", self.device, *FAILURE_ARGS_LIST, "basket",
                ]
                return self._call_inprocess(self._mod_basket, argv, RUN_FAILURE_DIR)
            cmd = [
                PYTHON_BIN, os.path.join(RUN_FAILURE_DIR, "run_basket.py"),
                "--mode", "basket", "--ckpt", CKPT_PATH_B,
                "--intr_json", INTR_COLOR, "--json", "--show",
                "--diff_mode", "bilateral",
                "--bg_path", "/home/dyros/mcy_ws/Laundry_decision/run_bf/laundry_bg/bg_basket.npy",
                "--device", self.device, *FAILURE_ARGS_LIST, "basket",
            ]
            return run_json(cmd, RUN_FAILURE_DIR)

        # ── failure ─────────────────────────────────────────────────
        elif mode == "failure":
            run_failure_py = os.path.join(RUN_FAILURE_DIR, "run_failure.py")
            cmd = [
                PYTHON_BIN, run_failure_py,
                "--mode", "failure", "--ckpt", CKPT_PATH_F,
                "--intr_json", INTR_JSON, "--json", "--show",
                "--rgb_ckpt", CKPT_PATH_RGB, "--device", self.device,
                *FAILURE_ARGS_LIST, "failure",
            ]
            return run_json(cmd, RUN_FAILURE_DIR)

        # ── failure_bg ──────────────────────────────────────────────
        elif mode == "failure_bg":
            run_failure_py = os.path.join(RUN_FAILURE_DIR, "run_failure.py")
            cmd = [
                PYTHON_BIN, run_failure_py,
                "--mode", "failure_bg", "--ckpt", CKPT_PATH_F,
                "--intr_json", INTR_JSON, "--json", "--show", "--device", self.device,
            ]
            return run_json(cmd, RUN_FAILURE_DIR)

        # ── failure_eh / failure_eh_strict ──────────────────────────
        elif mode in ("failure_eh", "failure_eh_strict"):
            if self._mod_failure_eh is not None:
                argv = [
                    "--mode", "failure_eh",
                    "--ckpt", CKPT_PATH_F,
                    "--rgb_ckpt", CKPT_PATH_RGB,
                    "--intr_json", INTR_JSON,
                    "--bg_path", "/home/dyros/mcy_ws/Laundry_decision/run_bf/laundry_bg/bg_overcome2.npy",
                    "--size", "384", "--zmax", "0.5", "--thr", "0.45",
                    "--diff_mode", "blend",
                    "--vote", "--vote_k", "10", "--vote_min_area", "150",
                    "--vote_min_dt", "3", "--vote_require_curr", "--stale_sec", "1.0",
                    "--save_dir", "/home/dyros/mcy_ws/Laundry_decision/run_bf/eh_dbg",
                    "--save_best_only", "--prefix", "failure_eh",
                    "--json", "--show",
                    "--dbg_maps", "--dbg_every", "10",
                    "--thr_sweep", "0.35,0.45,0.55", "--device", self.device,
                ]
                return self._call_inprocess(self._mod_failure_eh, argv, RUN_FAILURE_DIR)
            run_failure_eh_py = os.path.join(RUN_FAILURE_DIR, "run_failure_eh.py")
            cmd = [
                PYTHON_BIN, run_failure_eh_py,
                "--mode", "failure_eh", "--ckpt", CKPT_PATH_F,
                "--rgb_ckpt", CKPT_PATH_RGB, "--intr_json", INTR_JSON,
                "--bg_path", "/home/dyros/mcy_ws/Laundry_decision/run_bf/laundry_bg/bg_overcome2.npy",
                "--size", "384", "--zmax", "0.5", "--thr", "0.45",
                "--diff_mode", "blend",
                "--vote", "--vote_k", "10", "--vote_min_area", "150",
                "--vote_min_dt", "3", "--vote_require_curr", "--stale_sec", "1.0",
                "--save_dir", "/home/dyros/mcy_ws/Laundry_decision/run_bf/eh_dbg",
                "--save_best_only", "--prefix", "failure_eh",
                "--json", "--show", "--dbg_maps", "--dbg_every", "10",
                "--thr_sweep", "0.35,0.45,0.55", "--device", self.device,
            ]
            return run_json(cmd, RUN_FAILURE_DIR)

        # ── failure_rgb_diff ─────────────────────────────────────────
        elif mode == "failure_rgb_diff":
            if self._mod_failure_new is not None:
                argv = [
                    "--mode", "failure_rgb_diff",
                    "--bg_path", "/home/dyros/mcy_ws/Laundry_decision/run_bf/laundry_bg/bg_overcome2.npy",
                    "--ckpt", CKPT_PATH_F, "--intr_json", INTR_JSON,
                    "--size", "384", "--zmax", "0.5", "--thr", "0.45",
                    "--diff_mode", "blend",
                    "--vote", "--vote_k", "10", "--vote_min_area", "150",
                    "--vote_min_dt", "3", "--vote_require_curr", "--stale_sec", "1.0",
                    "--save_dir", "/home/dyros/mcy_ws/Laundry_decision/run_bf/eh_dbg",
                    "--save_best_only", "--prefix", "failure_rgb_diff",
                    "--json", "--show", "--device", self.device,
                ]
                return self._call_inprocess(self._mod_failure_new, argv, RUN_FAILURE_DIR)
            run_failure_new_py = os.path.join(RUN_FAILURE_DIR, "run_failure_new.py")
            cmd = [
                PYTHON_BIN, run_failure_new_py,
                "--mode", "failure_rgb_diff",
                "--bg_path", "/home/dyros/mcy_ws/Laundry_decision/run_bf/laundry_bg/bg_overcome2.npy",
                "--ckpt", CKPT_PATH_F, "--intr_json", INTR_JSON,
                "--size", "384", "--zmax", "0.5", "--thr", "0.45",
                "--diff_mode", "blend",
                "--vote", "--vote_k", "10", "--vote_min_area", "150",
                "--vote_min_dt", "3", "--vote_require_curr", "--stale_sec", "1.0",
                "--save_dir", "/home/dyros/mcy_ws/Laundry_decision/run_bf/eh_dbg",
                "--save_best_only", "--prefix", "failure_rgb_diff",
                "--json", "--show", "--device", self.device,
            ]
            return run_json(cmd, RUN_FAILURE_DIR)

        # ── failure_bg_eh ────────────────────────────────────────────
        elif mode == "failure_bg_eh":
            if self._mod_failure_new is not None:
                argv = [
                    "--mode", "failure_bg_eh",
                    "--bg_frames", "20",
                    "--bg_path", "/home/dyros/mcy_ws/Laundry_decision/run_bf/laundry_bg/bg_overcome2.npy",
                    "--ckpt", CKPT_PATH_F, "--intr_json", INTR_JSON,
                    "--size", "384", "--zmax", "0.5",
                    "--save_dir", "/home/dyros/mcy_ws/Laundry_decision/run_bf/_dbg",
                    "--prefix", "failure_bg_eh", "--json", "--show", "--device", self.device,
                ]
                return self._call_inprocess(self._mod_failure_new, argv, RUN_FAILURE_DIR)
            run_failure_new_py = os.path.join(RUN_FAILURE_DIR, "run_failure_new.py")
            cmd = [
                PYTHON_BIN, run_failure_new_py,
                "--mode", "failure_bg_eh", "--bg_frames", "20",
                "--bg_path", "/home/dyros/mcy_ws/Laundry_decision/run_bf/laundry_bg/bg_overcome2.npy",
                "--ckpt", CKPT_PATH_F, "--intr_json", INTR_JSON,
                "--size", "384", "--zmax", "0.5",
                "--save_dir", "/home/dyros/mcy_ws/Laundry_decision/run_bf/_dbg",
                "--prefix", "failure_bg_eh", "--json", "--show", "--device", self.device,
            ]
            return run_json(cmd, RUN_FAILURE_DIR)

        else:
            raise ValueError(f"알 수 없는 mode='{mode}'")


def main(args=None):
    from rclpy.executors import MultiThreadedExecutor
    parser = argparse.ArgumentParser(description='Node2 vision server')
    parser.add_argument('--device', type=str, default='cpu',
                        help='추론 디바이스 (cpu / cuda / cuda:0 등)')
    parser.add_argument('--gpu', action='store_true',
                        help='--device cuda 와 동일 (편의 flag)')
    parsed, ros_args = parser.parse_known_args()
    device = 'cuda' if parsed.gpu else parsed.device

    rclpy.init(args=ros_args or args)
    node = TargetPoseServer(device=device)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()


if __name__ == '__main__':
    main()
