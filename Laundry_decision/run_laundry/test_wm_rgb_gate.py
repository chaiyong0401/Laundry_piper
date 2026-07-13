"""
WM RGB Gate 단독 테스트
세탁기 앞에서 실시간으로 CLOTH/EMPTY 판단 결과를 확인한다.

python test_wm_rgb_gate.py
python test_wm_rgb_gate.py --ckpt models/best.pt --hi_thr 0.60 --lo_thr 0.45
"""
import argparse, time
import cv2
import numpy as np
import pyrealsense2 as rs
from run_laundry import _RGBGate


def enable_auto_exposure(pipeline):
    """모든 센서에 AE/AWB 활성화 (supports() 기반, node2_backup_in_process와 동일)."""
    device = pipeline.get_active_profile().get_device()
    for sensor in device.query_sensors():
        try:
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 1)
            if sensor.supports(rs.option.enable_auto_white_balance):
                sensor.set_option(rs.option.enable_auto_white_balance, 1)
        except Exception as e:
            print(f"[WARN] AE 설정 실패: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',   default='models/best.pt')
    ap.add_argument('--hi_thr', type=float, default=0.60)
    ap.add_argument('--lo_thr', type=float, default=0.45)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    gate = _RGBGate(args.ckpt, device_str=args.device)
    print(f"[INFO] 모델 로드 완료: {args.ckpt}  (hi={args.hi_thr}, lo={args.lo_thr})")

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  15)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    pipeline.start(cfg)

    # AE 명시적 활성화 후 안정화 대기
    enable_auto_exposure(pipeline)
    print("[INFO] AE 활성화 완료. 카메라 워밍업 (60 프레임)...")
    for _ in range(60):
        pipeline.wait_for_frames()

    print("[INFO] 시작. 'r'=EMA 리셋  'q'=종료")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color  = frames.get_color_frame()
            if not color:
                continue

            bgr = np.asanyarray(color.get_data())
            g   = gate.decide(bgr, hi_thr=args.hi_thr, lo_thr=args.lo_thr)

            decision = g['decision']
            cloth_p  = g['cloth_prob']
            ema_p    = g['ema_prob']

            color_map = {'CLOTH': (0, 200, 0), 'EMPTY': (0, 0, 220), 'UNCERTAIN': (0, 165, 255)}
            col = color_map.get(decision, (200, 200, 200))

            vis = bgr.copy()
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 60), (30, 30, 30), -1)
            cv2.putText(vis, f"{decision}", (10, 42),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, col, 2)
            cv2.putText(vis, f"p={cloth_p:.3f}  ema={ema_p:.3f}  hi={args.hi_thr}  lo={args.lo_thr}",
                        (200, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            bar_w = int(ema_p * vis.shape[1])
            cv2.rectangle(vis, (0, vis.shape[0]-10), (bar_w, vis.shape[0]), col, -1)

            cv2.imshow("WM RGB Gate", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                gate.ema_prob = None  # _RGBGate에 reset() 없으므로 직접 초기화
                print("[INFO] EMA 리셋")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
