# tactile_anomaly

XELA 촉각센서 기반 실시간 이상 감지 패키지.

> 이 문서의 명령은 모두 `Laundry_piper/` 최상위에서 실행합니다. 로봇 파이프라인에서는 `basket_ori.py --use_tactile_api`일 때만 이 패키지의 `DualChannelDetector`를 쓰고, 기본 실행은 최상위 `dual_channel_detector.py`·`train_gru_ae.py`를 씁니다. 여기 설명된 `xela_bridge`/`monitor_bridge`는 단독 모니터링·외부(C) 연동용입니다. 전체 시스템 설명은 최상위 [`README.md`](../README.md)를 참고하세요.

- **Channel A (slow)**: GRU Autoencoder window MSE → 점진적 이상 감지
- **Channel B (fast)**: 프레임별 MRL delta → 충격성 drop/slip 감지

---

## 패키지 구조

```
tactile_anomaly/
├── api_folder/
│   ├── grasp_05/          # 파지 구간 모델
│   └── transport_05/      # 이송 구간 모델
├── xela_bridge.py         # WebSocket → 추론 → TCP 서버
├── monitor_bridge.py      # 터미널 실시간 모니터링 클라이언트
├── detector.py            # DualChannelDetector
├── model.py               # GRUAE
└── preprocess.py          # 전처리 유틸
```

---

## 설치

```bash
pip install torch numpy scikit-learn joblib websocket-client
```

---

## 실시간 모니터링 사용법

### 1. xela_bridge 실행 (추론 서버)

xela_server WebSocket에서 센서 데이터를 받아 추론하고, TCP로 결과를 제공합니다.

```bash
# 파지 구간 모니터링 (preset 기본값 자동 적용)
python3 -m tactile_anomaly.xela_bridge --preset grasp_05_raw

# 이송 구간 모니터링
python3 -m tactile_anomaly.xela_bridge --preset transport_05_raw
```

**Preset 기본값:**

| preset | thr_a_val | anomaly_consec | max_norm_gate |
|---|---|---|---|
| `grasp_05_raw` | 0.4 | 2 | 0.3 |
| 나머지 | - | 3 | 0.3 |

개별 인자로 preset 기본값을 덮어쓸 수 있습니다:

```bash
python3 -m tactile_anomaly.xela_bridge --preset grasp_05_raw --thr_a_val 0.3
```

### 2. monitor_bridge 실행 (터미널 모니터링)

별도 터미널에서 실행합니다.

```bash
python3 tactile_anomaly/monitor_bridge.py
```

**Enter 키 = RESET 신호 전송** — 그리퍼를 닫은 직후 누릅니다.

출력 예시:
```
[Monitor] localhost:9001 연결됨  |  10.0Hz  |  Enter=RESET  |  Ctrl+C 종료
    time  state    channel       score_A    delta_B  log
------------------------------------------------------------------------
16:54:20  IDLE     ch=none  score_A=    0.0000  dB= 0.0000
                             ↑ Enter 누름
16:54:20  RESET                                 [Enter] 브릿지 리셋
16:54:21  IDLE     ch=none  score_A=    0.0000  dB= 0.0000  state -1→0
16:54:21  NORMAL   ch=none  score_A=    0.0251  dB= 0.0000  → 버퍼 완료, 정상 파지
16:54:22  ANOMALY  ch=A     score_A=    0.7100  dB= 0.0000  ⚠ 이상 감지
```

---

## 사용 흐름

### 파지 구간 (grasp_05_raw)

```
1. xela_bridge 실행 → RESET 대기 (IDLE 유지)
2. 그리퍼로 물체 파지
3. Enter 누름 → RESET 전송 → 누적 버퍼 초기화 → 추론 시작
4. IDLE (seq_len=50 프레임 채우는 중)
   → NORMAL : 정상 파지
   → ANOMALY: 빈 파지 또는 이상 감지 (RESET 전까지 유지)
5. 다음 파지 시도 시 Enter 다시 누름
```

> **RESET 타이밍**: 그리퍼가 물체를 **완전히 잡은 직후** Enter를 누릅니다.
> basket_ori.py의 `close_gripper()` → `reset_sensor_data()` 순서와 동일합니다.

### 이송 구간 (transport_05_raw)

```
1. 파지 완료 후 ba_init 위치 도달 시점에 Enter 누름
2. 이송 중 NORMAL/ANOMALY 모니터링
3. ANOMALY = 이송 중 낙하/slip 감지
```

---

## 상태 설명

| 상태 | 색상 | 의미 |
|---|---|---|
| `IDLE` | 회색 | RESET 이후 버퍼 충전 중 (seq_len=50프레임) 또는 RESET 대기 |
| `NORMAL` | 초록 | 정상 파지 (score_A < threshold) |
| `ANOMALY` | 빨강 | 이상 감지 — RESET 전까지 유지 (latch) |

**채널 표시 (`ch`):**

| ch | 의미 |
|---|---|
| `none` | 현재 프레임에서 A/B 모두 threshold 미초과 |
| `A` | Channel A (GRU-AE score_A) 초과 |
| `B` | Channel B (MRL delta_B) 초과 |
| `AB` | 둘 다 초과 |

> ANOMALY latch 상태에서는 현재 프레임이 정상이어도 `ch=none`으로 표시될 수 있습니다.

---

## 주요 파라미터

| 인자 | 설명 | 기본값 |
|---|---|---|
| `--preset` | 모델 프리셋 선택 | `grasp_05` |
| `--thr_a_val` | Channel A threshold 직접 지정 (preset 기본값 덮어쓰기) | preset별 상이 |
| `--anomaly_consec` | ANOMALY latch에 필요한 연속 프레임 수 | preset별 상이 |
| `--max_norm_gate` | 추론 활성화 최소 max_norm (0이면 비활성) | 0.3 |
| `--sensor_id` | XELA 센서 ID | 1 |
| `--tcp_port` | TCP 서버 포트 | 9001 |

### threshold 튜닝 가이드

```
정상 파지 후 score_A가 얼마나 나오는지 먼저 확인 → 그 값의 2~3배를 threshold로 설정

grasp_05_raw 학습 데이터 기준:
  success 학습 score : mean=0.010, p95=0.024
  실제 배포 success  : 0.03 ~ 0.10 (distribution shift 존재)
  failure score      : mean=0.710 (빈 파지 시)

  → thr_a_val=0.4 권장 (실제 성공 패턴과 실패 패턴 사이 충분한 마진)
```

---

## TCP 프로토콜 (C 클라이언트 연동)

```
C → Bridge : 1 byte
  0x01 = RESET  누적 버퍼 + 모델 상태 초기화
  0x02 = GET    최신 추론 결과 요청

Bridge → C : 12 bytes (GET 응답)
  [0]    uint8   anomaly  (0=정상, 1=이상)
  [1]    uint8   channel  (0=none, 1=A, 2=B, 3=AB)
  [2-5]  float32 score_a
  [6-9]  float32 delta_b
  [10]   uint8   state    (0=IDLE, 1=NORMAL, 2=ANOMALY)
  [11]   uint8   reserved
```

basket_ori.py / Logic_inte_wm.py 등에서 연동 시 `close_gripper()` 직후 `0x01` RESET을 전송하면 monitor_bridge의 Enter와 동일하게 동작합니다.

---

## Python API 직접 사용

```python
from tactile_anomaly.detector import DualChannelDetector

detector = DualChannelDetector.from_checkpoint(
    model_path       = 'grasp_05/grasp_gru_ae_raw.pt',
    scaler_path      = 'grasp_05/grasp_scaler_raw.pkl',
    threshold_a_path = 'grasp_05/grasp_threshold_raw.npy',
    threshold_b_path = 'grasp_05/grasp_threshold_b_raw.npy',
    infer_cfg_path   = 'grasp_05/grasp_infer_cfg_raw.pkl',
)

detector.reset()  # 에피소드 시작 시 반드시 호출

for frame_6d, raw_frame_48d in tactile_stream:
    result = detector.update(frame_6d, raw_frame=raw_frame_48d)
    # result: {'anomaly': bool, 'channel': str, 'score_a': float, 'delta_b': float}
```

---

## 모델 재학습

```bash
# grasp 모델 재학습 (relative=none 권장)
python3 train_gru_ae.py --data_dir grasp_05 --use_raw --model grasp --windowed --eval

# 학습 후 api_folder에 복사
cp grasp_05/grasp_gru_ae_raw.pt      tactile_anomaly/api_folder/grasp_05/
cp grasp_05/grasp_scaler_raw.pkl     tactile_anomaly/api_folder/grasp_05/
cp grasp_05/grasp_threshold_raw.npy  tactile_anomaly/api_folder/grasp_05/
cp grasp_05/grasp_threshold_b_raw.npy tactile_anomaly/api_folder/grasp_05/
cp grasp_05/grasp_infer_cfg_raw.pkl  tactile_anomaly/api_folder/grasp_05/
```

> `relative=none` + StandardScaler 조합이 `relative=first_frame`보다 failure 감지율이 높습니다.
> (grasp_05_raw 기준: relative=none → TP 100%, FP 12% / relative=first_frame → TP 53%)
