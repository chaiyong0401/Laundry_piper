# piper-mou (ROS 2 워크스페이스)

Piper 팔 드라이버, MoveIt 설정, XELA 촉각센서 브리지, 세탁물 작업 노드가 들어 있는 ROS 2 Humble 워크스페이스입니다.
전체 시스템 설명은 상위 [`README.md`](../README.md), 전체 실행 순서는 [`사용법`](../사용법)을 참고하세요.

---

## 빌드

```bash
cd Laundry_piper/piper-mou
source /opt/ros/humble/setup.bash
colcon build          # conda가 활성화되어 있으면 catkin_pkg 오류 → conda deactivate 후 빌드
source install/setup.bash
```

## 1. CAN 통신 설정 및 팔 제어 노드

```bash
# CAN 인터페이스 설정 (bitrate: 1 Mbps)
sudo ip link set can0 type can bitrate 1000000
sudo ip link set up can0

# CAN 통신 확인
candump can0

# 제어 노드 실행
ros2 run piper piper_single_ctrl --ros-args -p can_port:=can0 -p gripper_exist:=true -p gripper_val_mutiple:=2

# 모터 활성화 (둘 중 하나)
ros2 service call /enable_srv piper_msgs/srv/Enable "enable_request: true"
ros2 topic pub /enable_flag std_msgs/msg/Bool "data: true"
```

## 2. MoveIt 실행

```bash
# 가상 하드웨어 (플래닝만 확인)
ros2 launch piper_with_gripper_moveit wm_demo.launch.py

# 실제 로봇
ros2 launch piper_with_gripper_moveit wm_demo.launch.py real:=true scene2:=true add_basket:=false
```

## 3. 작업 노드

```bash
cd src/piper_ros/src/piper_moveit/piper_with_gripper_moveit/src
python node2_backup_in_process.py                                   # 비전 (먼저 실행)
python3 basket_ori.py                                               # 바구니 → 세탁기
python Logic_inte_wm.py --mode inference                            # 세탁기 → 바구니
```

> ⓘ ROS 2 Humble + Piper Arm 기준 문서입니다.
