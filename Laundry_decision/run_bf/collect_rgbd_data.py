"""
RGB + Depth 쌍 수집 스크립트
저장 구조:
  <save_root>/
  ├── cloth/
  │   ├── rgb/       {id}_rgb.png
  │   └── depth_img/ {id}_depth.png  (uint16, mm)
  └── empty/
      ├── rgb/
      └── depth_img/

실행 예시:
  python collect_rgbd_data.py --save_root /home/dyros/mcy_ws/Laundry_decision/dataset_new

키 조작:
  'e' → empty로 저장
  'c' → cloth로 저장
  'q' → 종료
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import argparse
import time


def main(args):
    cloth_rgb_dir   = os.path.join(args.save_root, args.domain, 'cloth', 'rgb')
    cloth_depth_dir = os.path.join(args.save_root, args.domain, 'cloth', 'depth_img')
    empty_rgb_dir   = os.path.join(args.save_root, args.domain, 'empty', 'rgb')
    empty_depth_dir = os.path.join(args.save_root, args.domain, 'empty', 'depth_img')
    for d in [cloth_rgb_dir, cloth_depth_dir, empty_rgb_dir, empty_depth_dir]:
        os.makedirs(d, exist_ok=True)

    # 기존 파일 수 파악 (id 이어서 저장)
    cloth_count = len(os.listdir(cloth_rgb_dir))
    empty_count = len(os.listdir(empty_rgb_dir))
    print(f"기존 cloth: {cloth_count}장, empty: {empty_count}장")
    print("'e'=empty  'c'=cloth  'q'=종료")

    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    align = rs.align(rs.stream.color)

    try:
        pipeline.start(config)
    except Exception as e:
        print(f"카메라 시작 실패: {e}")
        return

    # 워밍업
    for _ in range(30):
        pipeline.wait_for_frames()

    try:
        while True:
            frames      = pipeline.wait_for_frames()
            aligned     = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())   # BGR uint8
            depth_img = np.asanyarray(depth_frame.get_data())   # uint16, mm

            # 화면 표시
            depth_vis = cv2.convertScaleAbs(depth_img, alpha=0.05)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            disp = np.hstack([color_img, depth_color])
            cv2.putText(disp, f"cloth:{cloth_count}  empty:{empty_count}  [e/c/q]",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("RGBD Collector", disp)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('e'):
                sid = empty_count
                cv2.imwrite(os.path.join(empty_rgb_dir,   f"{sid}_rgb.png"),   color_img)
                cv2.imwrite(os.path.join(empty_depth_dir, f"{sid}_depth.png"), depth_img)
                empty_count += 1
                print(f"[empty {sid}] 저장")
            elif key == ord('c'):
                sid = cloth_count
                cv2.imwrite(os.path.join(cloth_rgb_dir,   f"{sid}_rgb.png"),   color_img)
                cv2.imwrite(os.path.join(cloth_depth_dir, f"{sid}_depth.png"), depth_img)
                cloth_count += 1
                print(f"[cloth {sid}] 저장")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\n최종 cloth: {cloth_count}장, empty: {empty_count}장")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_root', default='/home/dyros/mcy_ws/Laundry_decision/dataset_new',
                    help='저장 루트 디렉토리')
    ap.add_argument('--domain', default='dataset_basket',
                    help='도메인 폴더명 (dataset_basket / dataset_failure / dataset_wm 등)')
    main(ap.parse_args())
