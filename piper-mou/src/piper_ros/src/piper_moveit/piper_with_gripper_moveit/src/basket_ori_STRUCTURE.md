# basket_ori.py 구조 지도

> 자동 분석 + 코드 정독으로 작성한 구조 문서. 방대한 3,719줄(클래스 `ControlTower` 1개 + 모듈 함수, 총 53개)을 직접 다 읽지 않고 파악하기 위한 지도입니다.
> 대상 파일: `piper-mou/.../piper_with_gripper_moveit/src/basket_ori.py`

---

## 1. 한눈에 보기

- **역할**: 바구니(basket) → 세탁기(washing machine) 세탁물 파지·이송 로봇의 최상위 제어 노드
- **구성**: `ControlTower(rclpy.Node)` 단일 클래스 (메서드 51개) + 모듈 함수 `build_pin_model`, `main`
- **기본 실행**: `python3 basket_ori.py` → `--mode new_9d_relative`(내부적으로 `inference`), 튜닝값 `threshold_a=0.1259 / infer_stride=10 / consec=2` 자동 적용
- **핵심 메서드 4개가 전체의 절반**: `execute_sequence`(577줄) + `_sensor_processing_worker`(509줄) + `__init__`(325줄) + `overcome2`(309줄)

### 실행 모드
| `--mode` | 내부 run_mode | 설명 |
|---|---|---|
| `normal` | normal | rule-based 판단만, 데이터 수집·이상탐지 없음 |
| `collect` | collect | 파지/이송 시계열 데이터 수집·저장(사람이 라벨링) |
| `inference` | inference | 기존 GRU-AE 이상탐지 |
| `new_11d` | inference | Codex first-frame 11D 모델 |
| **`new_9d_relative`** | inference | **(기본)** stable-baseline relative 9D 모델 |

---

## 2. 스레딩 모델 (중요)

이 노드는 **메인 스레드 + 3개 백그라운드 스레드**로 병렬 동작합니다. 이게 구조 이해의 핵심입니다.

```
main()
 ├─ MultiThreadedExecutor.spin_once() 루프  ← 메인 스레드: ROS 콜백 처리 전담
 │    (sensor_callback, joint_state_callback, ee_pose_callback, state_cb, tick, 마커)
 │
 ├─[daemon thread] execute_sequence()        ← 실제 로봇 작업 시퀀스 (핵심 루프)
 ├─[thread] _sensor_processing_worker()      ← 촉각센서 raw → feature → 이상감지
 └─[thread] _inference_worker()              ← GRU-AE 채널A 추론 (트리거 대기형)
```

- **콜백은 데이터만 큐/버퍼에 넣고**(예: `sensor_callback` → `_raw_sensor_queue`), 무거운 처리는 워커 스레드가 담당 → DDS 블로킹 방지
- `execute_sequence`(작업)와 `_sensor_processing_worker`/`_inference_worker`(감지)는 **이벤트·락·플래그**(`_infer_trigger`, `_insert_trigger`, `_transport_abort`, `_infer_lock`, `contacted`, `z_aligned`)로 통신

---

## 3. 실행 흐름 (기본 = new_9d_relative / inference)

```
main
 └─ ControlTower.__init__          모델·ROS·스레드 셋업
     └─ execute_sequence  ── while rclpy.ok() ──────────────────────────┐
         │                                                               │
         │  [Step1] 그리퍼 open → ba_init 복귀 → 빈그리퍼 기준 캡처       │
         │  [Step2] target_get_ba (바구니 파지좌표)                       │
         │            └ 빈 바구니(z=-99) → overcome2() ───┐              │
         │  [Step2-1] 3단계 접근 (직선하강→수직IK→자유IK)   │              │
         │  [Step2-3] move_forward(Push) → close_gripper   │              │
         │  [파지판정] inference: 이상탐지 결과 / rule fallback           │
         │            실패 → 그리퍼 open, 재시도                          │
         │  [Step3] wm_init 이송 시작 (Model2 녹화, _transport_abort 감시)│
         │            └ 낙하감지 → _floor_recovery_grasp() (최대3회)      │
         │  [Step4] wm_end 도착 → 이송 성공/이상 판정 → 그리퍼 open        │
         │            └ contacted=True → overcome2() ──────┤              │
         │  [복귀] wm_init 복귀 → 다음 사이클 continue ─────┘──────────────┘
         │
         └ 빈 바구니 확정 → 타이밍요약 출력 → home 복귀 → 종료
```

`overcome2`는 **세탁기 드럼에 남은/떨어진 세탁물을 재파지→투입** 반복(빈 드럼 될 때까지)하는 극복 루프입니다.

---

## 4. 역할별 메서드 그룹 (53개 전체)

### 🟦 초기화 / 모델 로딩
| 메서드 | 줄 | 설명 |
|---|---|---|
| `__init__` | L77 (325) | 노드 전체 셋업: ROS 구독/퍼블/액션클라이언트, 타이머(100Hz `tick`, 2Hz 마커), 모델 로딩, 상태·설정 버퍼, 3개 워커 스레드 시작, 캘리브레이션(R_cal·t_cal), 사전정의 관절포즈 |
| `build_pin_model` | L63 (10) | URDF/xacro → Pinocchio 동역학 모델 빌드(중력 설정) |
| `_load_minmax` | L2629 | taxel Min-Max 정규화 파일(48축 min/range) 로드 |
| `_load_gru_ae` | L1179 | **Grasp(Model1)** GRU-AE+scaler+threshold 로드 (6D MRL 고정) |
| `_load_grasp_api_detector` | L1232 | (use_tactile_api) grasp용 DualChannelDetector API 로드 |
| `_load_insert_gru_ae` | L1301 | **Transport(Model2)** GRU-AE 로드. codex 모드면 아래로 위임, 아니면 11d/10d/8d/9d/raw 자동선택(레거시) |
| `_load_insert_gru_ae_codex` | L1384 (146) | **Codex first-frame** transport 모델: new_9d_relative(dim9) / new_11d(dim11) artifact 선택, report.json 하이퍼파라미터, 런타임 override 적용 |
| `_load_insert_api_detector` | L1531 | (use_tactile_api) transport용 DualChannelDetector API 로드 |

### 🟩 모션 / 포즈
| 메서드 | 줄 | 설명 |
|---|---|---|
| `goal` | L540 | 8DOF 관절목표 이동(도달시 skip, MoveIt 실패시 direct JTC fallback) + 그리퍼 옵션 |
| `move_forward` | L646 (95) | Mode9(Cartesian)→ik fallback 하이브리드 전진, 4σ 토크 정지감시, "success/stop/fail" |
| `move_backward` | L942 | 잡은 뒤 후진(현재 Z +15cm, Mode6 직행) |
| `align_joint6` | L744 | joint6만 회전해 EE 수평 정렬(direct JTC) |
| `transform_pose_with_offset` | L2331 | Pose position에 4x4 변환행렬 곱(orientation 유지) |

### 🟨 액션 goal / 대기
| 메서드 | 줄 | 설명 |
|---|---|---|
| `_send_goal` | L422 (116) | FollowJointTrajectory 전송: 이동량 기반 duration 동적계산·폴링·타임아웃/취소·성공판정 |
| `_wait_future` | L591 | executor spin 중 future 폴링 대기(1s 경고) ※일부 호출 timeout 없음 |
| `send_gripper_command` | L573 | joint7/8 값으로 그리퍼 goal |
| `open_gripper` / `close_gripper` | L582 / L587 | 그리퍼 0.03 열기 / 0.0 닫기 |

### 🟥 정지 / 복구 / 극복
| 메서드 | 줄 | 설명 |
|---|---|---|
| `tick` | L809 | (100Hz) rnea 예측토크 vs 실측 4σ 초과 시 정지요청 |
| `_request_stop` | L844 | stop 플래그·사유 세팅 후 정지 override 스레드 발사 |
| `_send_stop_override` | L857 (64) | 현재자세를 속도0 초단시간 trajectory로 즉시정지 |
| `_handle_stop_and_recover` | L923 | stop 요청 시 override 재전송 후 플래그 해제 |
| `recover_robot` | L615 | `/enable_srv` 재호출로 Safety/Torque fault 해제·재활성화 |
| `_trigger_slip_stop` | L1682 | slip 감지 시 goal 취소 없이 contacted/monitor_slip만 기록 |
| `_floor_recovery_grasp` | L2241 (89) | 이송 중 낙하 세탁물 바닥 재파지(attempt마다 z 1cm↓, 최대 3회) |
| `overcome2` | L2657 (309) | **세탁기 드럼 잔여물 재파지→투입 극복 루프**(빈 드럼까지 반복) |

### 🟪 센서 / 콜백 / 정규화
| 메서드 | 줄 | 설명 |
|---|---|---|
| `sensor_callback` | L1714 | `/xServTopic` taxel x/y/z만 추출 → `_raw_sensor_queue`(non-blocking) |
| `_sensor_processing_worker` | L1727 (509) | **촉각 raw→feature→이상감지 스레드** (아래 상세) |
| `joint_state_callback` | L1693 | JointState → 8DOF 리스트 `current_joint_state` |
| `joint_cb` | L793 | `/joint_states` effort 저장(비유한값 방어) |
| `state_cb` | L769 | `/arm_controller/state` desired 위치/속도/가속도 |
| `ee_pose_callback` | L405 | `/end_pose` → 4x4 행렬 + PoseStamped |
| `reset_sensor_data` | L965 | prev/cum 센서값·z_aligned·contacted 초기화 |
| `_capture_empty_gripper_reference` | L971 | 무부하 그리퍼 taxel DC offset·접촉 threshold 캡처 |
| `_reset_transport_channel_b_state` | L1011 | 이송 채널B warmup/baseline 초기화 |
| `_normalize_taxels` | L2646 | raw 48축 Min-Max 정규화 |

### 🟫 이상탐지 추론
| 메서드 | 줄 | 설명 |
|---|---|---|
| `_inference_worker` | L1060 (117) | grasp/transport 채널A(GRU-AE 재구성오차) 스레드, 트리거 대기형, 채널B와 OR 판정 |
| `_run_windowed_inference` | L1266 | grasp: 최근 50프레임 창 정규화·relative → 재구성오차 |
| `_run_insert_windowed_inference` | L1577 | transport: 동일 방식 재구성오차 |

### 🟧 타겟 좌표 계산 (비전)
| 메서드 | 줄 | 설명 |
|---|---|---|
| `target_get_wm` | L2354 (147) | **세탁기** 파지좌표: vision 액션 요청 → base_link 변환 → Z/X/Y 클램프·마진 → 고정 orientation → ff 자세 복귀 |
| `target_get_ba` | L2502 (125) | **바구니** 파지좌표: vision 요청 → 변환 → 바구니 벽/바닥 제약(min_dist) → 현재 EE orientation 사용 |

### 🟦 데이터 저장 / 입력 / 시각화 / 기타
| 메서드 | 줄 | 설명 |
|---|---|---|
| `_save_grasp_data` | L1044 | 파지 시계열 6D/48D raw npy 저장 |
| `_save_insert_data` | L1609 | 이송 시계열 6D/48D raw/abs npy 저장 |
| `_ask_insertion_result` | L1634 | (사람) 이송 성공 y/n/x |
| `_ask_gripper_held` | L1658 | (사람) 파지 유지 y/n/x |
| `_spin_input` | L1023 | executor 충돌 없이 별도 스레드 input() |
| `publish_target_marker` | L3574 | 타겟 SPHERE 마커 발행(RViz) |
| `publish_camera_center_callback` | L3594 | (2Hz) 카메라 중심 위치 마커 |
| `_step_time` / `_print_timing_summary` | L2968 / L2974 | 구간 타이밍 계측·요약 |
| `main` | L3631 | argparse + Executor 구성 + execute_sequence daemon 스레드 실행 |
| `execute_sequence` | L2990 (577) | **코어 작업 루프** (섹션 3 참조) |
| `close_door` | L2237 | ⚠️ 빈 스텁(`pass`, 미구현) — `target_get_ba`에서 호출됨 |

---

## 5. `_sensor_processing_worker` 상세 (L1727, 509줄)

촉각센서 처리·이상감지의 심장부:
1. **입력/중복제거**: `_raw_sensor_queue`에서 16-taxel raw → 직전과 동일하면 skip(~500Hz 재발행 제거), 0.5s 무데이터면 WebSocket 끊김으로 리셋 → `_normalize_taxels`
2. **누적/점프**: 프레임 diff 누적(cum). diff>0.5 "점프" 시 이송 중이면 유지(slip/drop 신호), 아니면 재연결 아티팩트로 리셋+flush
3. **Feature**: norm, theta, MRL, max_norm + 모델별 feature(6D grasp, 9D/8D/10D/11D transport, 9D-relative)
4. **채널 A/B**: grasp·transport 각각 채널A(GRU-AE worker 트리거 or API) + 채널B(퍼프레임 즉시: grasp=MRL delta, transport=contact_drop/force_drop). 연속 이상 임계 도달 시 anomaly 플래그, transport 이상이면 `_transport_abort`
5. **slip/상태판정**: norm 50%↓ 또는 theta 20°변화 시 `_trigger_slip_stop`, max_norm/norm/theta로 `z_aligned`/`contacted` 갱신

---

## 6. 호출 관계 (누가 누구를 부르나)

**진입점(외부 등록/스레드/main)**: `__init__`, `main`, `execute_sequence`, `_sensor_processing_worker`, `_inference_worker`, `sensor_callback`, `joint_state_callback`, `ee_pose_callback`, `state_cb`, `joint_cb`, `tick`, `publish_camera_center_callback`

```
execute_sequence  →  target_get_ba, move_forward, goal, open/close_gripper,
                     _capture_empty_gripper_reference, reset_sensor_data,
                     _floor_recovery_grasp, overcome2, _save_grasp_data,
                     _save_insert_data, _ask_gripper_held, _ask_insertion_result,
                     _handle_stop_and_recover, _reset_transport_channel_b_state,
                     publish_target_marker, _step_time, _print_timing_summary

overcome2         →  target_get_wm, move_forward, goal, align_joint6,
                     open/close_gripper, recover_robot, reset_sensor_data,
                     _handle_stop_and_recover, publish_target_marker, _step_time

_floor_recovery_grasp → target_get_ba, move_forward, move_backward, goal, open/close_gripper, reset_sensor_data

target_get_wm/ba  →  _wait_future, goal, publish_target_marker, transform_pose_with_offset  (ba는 +close_door)

goal              →  _send_goal, open/close_gripper
move_forward      →  _wait_future
_inference_worker →  _run_windowed_inference, _run_insert_windowed_inference
_sensor_processing_worker → _normalize_taxels, _trigger_slip_stop
__init__          →  build_pin_model, _load_minmax, _load_gru_ae, _load_grasp_api_detector,
                     _load_insert_gru_ae, _load_insert_api_detector
```

---

## 7. 모드 게이팅 (기본 실행 시 "안 도는" 코드)

기본값 `new_9d_relative`(=inference)로 돌리면 아래는 조건 불충족으로 **실행되지 않습니다**(죽은 코드 아님, `collect` 모드 전용):
- `execute_sequence` 내 `run_mode == 'collect'` 분기 (L3235·L3497·L3501 부근): `_ask_gripper_held`, `_ask_insertion_result`, `_save_grasp_data`, `_save_insert_data` 등 데이터 수집·라벨링 경로

즉 **정의된 53개 메서드는 전부 어떤 모드에서든 도달 가능(죽은 메서드 0개)**하지만, 한 번의 inference 실행에서 실제로 실행되는 건 그 부분집합입니다.

---

## 8. 알아둘 점 / 주의

- ⚠️ **`close_door`(L2237)는 빈 스텁**(`pass`)인데 `target_get_ba`에서 호출됨 → 문 닫기 동작 없음(의도 확인 필요)
- ⚠️ **`overcome2` 내 `if True:` 뒤 `else` 분기(L2955~)는 도달 불가**(slip 실패 복구 코드가 죽어 있음)
- 이상탐지는 **채널 A(GRU-AE 재구성오차) + 채널 B(퍼프레임 물리 규칙)** 이중구조, 둘을 OR로 결합
- 좌표변환 핵심: `ee2cam`(R_cal,t_cal) → `base2cam = ee_pose @ ee2cam` → base_link 프레임

---
*이 문서는 코드를 변경하지 않습니다. 구조 파악용 참고 자료입니다.*
