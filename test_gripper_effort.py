#!/usr/bin/env python3
"""
그리퍼 토크 제어 가능 여부 확인 실험
실행 전 주의사항:
  1. ROS2 노드(basket2.py 등)가 실행 중이면 CAN 충돌 가능 → 먼저 종료
  2. 로봇 전원 ON, CAN 연결 확인 (can0)
  3. 그리퍼 주변에 손/장애물 없는지 확인

실행:
  python3 /home/dyros/mcy_ws/test_gripper_effort.py
"""

import time
from piper_sdk import C_PiperInterface_V2

CAN_PORT = "can0"

def connect():
    piper = C_PiperInterface_V2(CAN_PORT)
    piper.ConnectPort()
    print("[연결] CAN 포트 연결 완료")
    print("[대기] 로봇 Enable 중...")
    timeout = 5.0
    t0 = time.time()
    while not piper.EnablePiper():
        if time.time() - t0 > timeout:
            print("[경고] Enable 타임아웃. 계속 진행합니다.")
            break
        time.sleep(0.05)
    print("[완료] 로봇 Enable")

    # 그리퍼 독립 초기화
    print("[그리퍼] disable + clear error")
    piper.GripperCtrl(0, 1000, 0x02, 0)
    time.sleep(0.5)
    print("[그리퍼] enable")
    piper.GripperCtrl(0, 1000, 0x01, 0)
    time.sleep(0.5)

    return piper

def print_feedback(piper, label=""):
    msg = piper.GetArmGripperMsgs()
    g = msg.gripper_state
    print(f"  [{label}] angle: {g.grippers_angle} ({g.grippers_angle*0.001:.3f} mm) | "
          f"effort: {g.grippers_effort} ({g.grippers_effort*0.001:.3f} N·m)")

# ────────────────────────────────────────────────────────────
# 실험 1: 피드백 모니터링
#   그리퍼를 열고 닫으면서 angle/effort 피드백 확인
# ────────────────────────────────────────────────────────────
def experiment1(piper):
    print("\n" + "="*60)
    print("실험 1: 피드백 모니터링 (10초간)")
    print("  그리퍼를 열고 닫으면서 angle/effort 피드백을 확인합니다.")
    print("="*60)

    # 열기
    print("\n[1] 그리퍼 열기 (angle=50000 = 50mm, effort=1000 = 1 N·m)")
    piper.GripperCtrl(50000, 1000, 0x01, 0)
    for i in range(20):
        print_feedback(piper, f"열기 {i*0.1:.1f}s")
        time.sleep(0.1)

    # 닫기
    print("\n[2] 그리퍼 닫기 (angle=0, effort=1000 = 1 N·m)")
    piper.GripperCtrl(0, 1000, 0x01, 0)
    for i in range(20):
        print_feedback(piper, f"닫기 {i*0.1:.1f}s")
        time.sleep(0.1)

    print("\n실험 1 완료.")

# ────────────────────────────────────────────────────────────
# 실험 2: 얇은 물체를 끼운 상태에서 effort만 변경
#   → 그리퍼가 더 조여드는지 확인
# ────────────────────────────────────────────────────────────
def experiment2(piper):
    print("\n" + "="*60)
    print("실험 2: effort 변경 실험")
    print("  지금 그리퍼 사이에 얇은 물체(종이, 천 등)를 끼워주세요.")
    print("="*60)

    input("\n  >>> 물체를 끼운 후 Enter를 누르세요...")

    # 먼저 열기
    print("\n[1] 그리퍼 열기")
    piper.GripperCtrl(50000, 1000, 0x01, 0)
    time.sleep(1.5)

    # 약한 토크로 닫기
    print("\n[2] 약한 토크로 닫기 (effort=500 = 0.5 N·m)")
    piper.GripperCtrl(0, 500, 0x01, 0)
    time.sleep(1.5)
    print("  -- effort=500 피드백 --")
    for i in range(10):
        print_feedback(piper, f"  500  {i*0.1:.1f}s")
        time.sleep(0.1)

    # 강한 토크로 변경 (angle은 동일하게 0)
    print("\n[3] 강한 토크로 변경 (effort=3000 = 3 N·m, angle 동일=0)")
    piper.GripperCtrl(0, 3000, 0x01, 0)
    time.sleep(1.5)
    print("  -- effort=3000 피드백 --")
    for i in range(10):
        print_feedback(piper, f"  3000 {i*0.1:.1f}s")
        time.sleep(0.1)

    print("\n  [확인 포인트]")
    print("  - effort 피드백 값이 달라졌는가?")
    print("  - 그리퍼가 물체를 더 강하게 조였는가 (angle 피드백이 줄었는가)?")

    # 열기 (원복)
    print("\n[4] 그리퍼 열기 (원복)")
    piper.GripperCtrl(50000, 1000, 0x01, 0)
    time.sleep(1.0)

    print("\n실험 2 완료.")

# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Piper 그리퍼 토크 제어 실험")
    print("주의: ROS2 노드가 실행 중이면 종료 후 실행하세요.\n")

    choice = input("실험 선택 [1: 피드백 모니터링 / 2: effort 변경 / 12: 둘 다]: ").strip()

    piper = connect()

    if '1' in choice:
        experiment1(piper)
    if '2' in choice:
        experiment2(piper)

    print("\n모든 실험 완료.")
