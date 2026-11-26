# Laundry_piper

Piper 6축 로봇팔로 **바구니 ↔ 세탁기** 사이에서 세탁물을 옮기는 시스템입니다.
손목 카메라(RealSense D405)로 파지 위치를 찾고, 그리퍼의 XELA 촉각센서로 **파지 실패·이송 중 낙하**를 실시간으로 감지합니다.

- 실행 명령 순서 → [`사용법`](사용법)
- 핵심 제어 코드 구조 → [`basket_ori_STRUCTURE.md`](piper-mou/src/piper_ros/src/piper_moveit/piper_with_gripper_moveit/src/basket_ori_STRUCTURE.md)
- 촉각 이상감지 패키지 → [`tactile_anomaly/README.md`](tactile_anomaly/README.md)

> **전달본 안내**: 이 코드 묶음에는 학습된 모델·정규화 파일(`.pt`, `.pth`, `.npy`, `.pkl`)과 XELA 센서 실행 파일이 빠져 있습니다. 코드 열람은 가능하지만, 실행하려면 [아래 목록](#4-모델데이터-파일-별도-전달)의 파일이 제자리에 있어야 합니다. 모델이 들어갈 폴더는 빈 폴더로 남겨 두었습니다.
> 외부 패키지(`piper_sdk`, `trac_ik`, `topic_based_ros2_control`)는 빌드에 필요한 부분만 남기고 데모·변경 이력·개발도구 설정은 뺐습니다.

---

## 1. 시스템 구성

```
 [Piper 팔] ──CAN(can0)──> piper_single_ctrl ──/joint_states, /end_pose──┐
                                  ▲                                     │
                                  │ /arm_controller, /gripper_controller│
 wm_demo.launch.py (MoveIt + planning_scene + pose_goal/joint_goal 노드)│
                                  ▲ pose_goal 액션 / FollowJointTrajectory
                                  │                                     ▼
 [D405 카메라] ──> node2_backup_in_process.py ──target_pose 액션──> 작업 노드 (둘 중 하나)
                   (비전: 바구니/세탁기 세탁물 위치)                ├ basket_ori.py    : 바구니 → 세탁기
                                                                    └ Logic_inte_wm.py : 세탁기 → 바구니
 [XELA 촉각] ──CAN(can1)──> xela_server ──> xela_service.py ──/xServTopic──┘
                                                                    (파지·이송 이상감지)
```

| 구성 요소 | 파일 / 패키지 | 역할 |
|---|---|---|
| 로봇 드라이버 | `piper-mou/src/piper_ros/src/piper` | CAN으로 Piper 팔·그리퍼 제어, `/enable_srv`로 모터 활성화 |
| 모션 플래닝 | `piper_with_gripper_moveit` (`launch/wm_demo.launch.py`) | MoveIt, 충돌 환경(`planning_scene*.cpp`: 세탁기·책상·바구니), `pose_goal`/`joint_goal` 액션 서버 |
| 비전 | `piper_with_gripper_moveit/src/node2_backup_in_process.py` | `target_pose` 액션 서버. `mode`에 따라 바구니(`basket`), 바닥 실패물(`failure_eh`), 세탁기 드럼(`wm`) 안의 세탁물 좌표를 반환. 대상이 없으면 `z=-99` |
| 촉각 센서 | `piper-mou/src/xela_sensor`, `xela_server_ros2` | XELA 서버 → ROS 토픽 `/xServTopic` |
| 작업 노드 ① | `piper_with_gripper_moveit/src/basket_ori.py` | 바구니 → 세탁기 투입. 파지 판정(Grasp 모델) + 이송 중 낙하 감지(Transport 모델) + 바닥 낙하물 복구, 드럼 정리(`overcome2`) |
| 작업 노드 ② | `piper_with_gripper_moveit/src/Logic_inte_wm.py` | 세탁기 → 바구니 꺼내기. 문 열기 → 드럼 파지 → 바구니로 이송, 같은 촉각 이상감지 사용 |

### 촉각 이상감지 (두 채널을 OR로 결합)
- **Channel A**: GRU Autoencoder 재구성 오차 → 점진적 이상(빈 파지 등). 모델 정의·전처리는 `train_gru_ae.py`
- **Channel B**: 프레임 간 변화량 규칙 → 순간적인 낙하·미끄러짐. `dual_channel_detector.py`의 `FramewiseDeltaDetector`
- 이송 구간 특징 계산은 `piper_with_gripper_moveit/src/transport_features.py`

---

## 2. 폴더 구조

```
Laundry_piper/
├── README.md                  ← 이 문서
├── 사용법                      ← 터미널별 실행 명령
├── requirements.txt           ← Python 의존성
├── piper_ee2cam.txt           ← 손목 카메라 hand-eye 캘리브레이션 (현재값 + 이전값 기록)
├── train_gru_ae.py            ← GRU-AE 모델 정의/전처리 (작업 노드가 import; 학습 스크립트 겸용)
├── dual_channel_detector.py   ← Channel B 검출기
├── tactile_anomaly/           ← 촉각 이상감지 패키지 (단독 모니터링·C 연동용 API 포함)
├── grasp_new/                 ← [모델] 파지 판정 모델
├── transport_new/             ← [모델] 이송 낙하 감지 모델
├── transport_infer/, wm_05/   ← 실행 중 데이터 저장 위치 (빈 폴더)
├── Laundry_decision/
│   ├── environment.yml        ← conda 환경(rs) 정의
│   ├── run_bf/                ← [모델] 비전 U-Net·RGB 분류 모델, 카메라 intrinsics, 배경
│   └── run_laundry/           ← 바닥 실패물 검출기(detector/) + [모델] models/best.pt
└── piper-mou/                 ← ROS 2 워크스페이스
    └── src/
        ├── piper_ros/src/
        │   ├── piper, piper_msgs, piper_description, piper_sdk   ← 로봇 드라이버·URDF (제조사 제공)
        │   ├── piper_sim          ← URDF 생성에 필요해서 유지 (시뮬레이션은 사용하지 않음)
        │   └── piper_moveit/piper_with_gripper_moveit/
        │       ├── launch/wm_demo.launch.py
        │       ├── config/        ← MoveIt·컨트롤러 설정
        │       ├── action/        ← TargetPose, PoseGoal 등 액션 정의
        │       └── src/           ← ★ basket_ori.py, Logic_inte_wm.py, node2_backup_in_process.py, *.cpp
        ├── topic_based_ros2_control   ← 실기 하드웨어 인터페이스 (외부 패키지)
        ├── trac_ik                    ← IK 솔버 (외부 패키지)
        ├── xela_server_ros2           ← XELA → ROS 브리지
        └── xela_sensor/               ← XELA 서버 실행 파일 (전달본에서는 enable.py, xServ.ini만 포함)
```

**처음 읽을 때 추천 순서**: `사용법` → `basket_ori_STRUCTURE.md` → `basket_ori.py`의 `execute_sequence` → `node2_backup_in_process.py`의 `TargetPoseServer` → `Logic_inte_wm.py`

---

## 3. 환경 및 빌드

- Ubuntu 22.04, ROS 2 Humble, MoveIt 2, Python 3.10
- 하드웨어: Piper 팔 + 그리퍼 (USB-CAN, `can0`), XELA 촉각센서 (USB 시리얼-CAN, `can1`), RealSense D405
- Python: conda 환경 `rs` (`Laundry_decision/environment.yml`) 또는 `pip install -r requirements.txt`

```bash
cd Laundry_piper/piper-mou
source /opt/ros/humble/setup.bash
# conda가 PATH 앞에 있으면 colcon이 conda python을 잡아 "No module named catkin_pkg"로 실패함
#   → conda deactivate 하거나 PATH에서 miniconda3/bin 제거 후 빌드
colcon build
source install/setup.bash
```

빌드 후 실행 순서는 [`사용법`](사용법)을 따릅니다.

---

## 4. 모델·데이터 파일 (별도 전달)

코드에 **절대경로 `/home/user/samsung_laundry/Laundry_piper/...`** 가 하드코딩되어 있습니다. 다른 위치에 설치하면 아래 파일의 경로를 바꿔야 합니다.

| 사용하는 코드 | 파일 (Laundry_piper/ 기준) |
|---|---|
| `node2_backup_in_process.py` (L43–54) | `Laundry_decision/run_bf/dataset_rgb_sam2/unet_best_basket_new.pt`, `unet_best_failure_new.pt`, `rgb_only.pth` |
| | `Laundry_decision/run_bf/intrinsics_color_640x480.json`, `intrinsics_depth_640x480.json` (전달본에 포함) |
| | `Laundry_decision/run_bf/laundry_bg/bg_basket.npy`, `bg_overcome2.npy` |
| | `Laundry_decision/run_laundry/models/best.pt` |
| `basket_ori.py` (L135–159), `Logic_inte_wm.py` (L56, L572, L611) | `piper-mou/.../piper_with_gripper_moveit/src/taxel_minmax_norm.npy` |
| | `grasp_new/grasp_gru_ae.pt`, `grasp_scaler.pkl`, `grasp_threshold.npy`, `grasp_threshold_b.npy`, `grasp_infer_cfg.pkl` |
| | `transport_new/transport_*_raw.*` (Logic_inte_wm, basket_ori `--mode inference`) |
| | `transport_new/codex_model_9d_relative_firstframe/` (basket_ori 기본 모드 `new_9d_relative`) |
| `basket_ori.py`, `Logic_inte_wm.py` | 저장 위치: `transport_infer/`, `wm_05/` (빈 폴더) |
| `basket_ori.py` (L44) | URDF: `piper-mou/src/piper_ros/src/piper_description/urdf/piper_description.xacro` (전달본에 포함) |

---

## 5. 알려진 사항

- `train_gru_ae.py` 안에 예전 작업 경로 `/home/dyros/mcy_ws`가 2곳 남아 있습니다 (학습 스크립트 경로, 로봇 실행에는 영향 없음).
- MoveIt 실행 로그의 `'gripper' is not a chain`, `DepthImageOctomapUpdater` 관련 ERROR는 기존 설정에서 원래 나오는 메시지이며 동작에는 지장이 없습니다.
- `basket_ori.py`의 주의할 점(빈 스텁 `close_door`, 도달 불가 분기 등)은 `basket_ori_STRUCTURE.md` 8절에 정리되어 있습니다.
