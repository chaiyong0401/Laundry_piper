# run_laundry.py — **WM 모드 사용 설명서** 

RealSense D435i로 **세탁기 드럼 내부**를 관찰해 포인트클라우드 기반으로 **의류 클러스터**를 검출하고,  
그 중심을 **그립 포인트(X, Y, Z)**로 산출

---

## 1) 빠른 실행 예시

```bash
python run_laundry.py \
  --mode wm \
  --rgb_ckpt /home/cheon/Documents/gd_0808/models/best.pt \
  --rgb_thr 0.60 --rgb_low_thr 0.45 \
  --json
```
- `--json`을 넣으면 GUI 없이 **마지막 줄에 JSON 한 줄**을 출력하고 종료(ros2 action에서 json 파싱할 때 사용)
- GUI 모드(기본): 실행 후 창이 뜨고, **키보드 `d`**를 눌러 감지를 수행

---

## 2) 출력(JSON 스키마)

성공/빈 판단은 RGB 게이트 및 3D 클러스터링 결과에 따릅니다.

```jsonc
// EMPTY로 판단 시(예: RGB 게이트로 빈으로 확신)
{"success": true, "centers": [], "log": "rgb-empty-skip p=0.12 ema=0.18 thr=0.60"}

// 의류 감지 성공 시(카메라 좌표계, 미터 단위)
{
  "success": true,
  "log": "origin center : [[...], ...], filtered center : [[...], ...]",
  "centers": [[X, Y, Z], ...]   // 한 개 이상일 수 있으나, 보통 정렬 후 1개를 사용, 현재 정렬을 cylinder의 중앙에 가까운 점 산출하게 되어있음. 
}
```
- `centers`는 최종 보정 후 **카메라 좌표계**로 변환된 좌표
  (ICP 변환 행렬 `T_icp`의 역행렬을 적용하여 ref→camera로 변환)

---

## 3) 주요 파이프라인 (WM)

1. **RGB 게이트(선택)** — `_RGBGate`
   - state_dict(`.pth`) 체크포인트 로딩
   - 이미지넷 정규화 후 2‑클래스(빈/의류) 분류
   - **지수이동평균(EMA)** 확률로 히스테리시스 판단
     - `ema ≥ rgb_thr` → CLOTH
     - `ema < rgb_low_thr` → EMPTY
     - 그 외 구간 → 보수적으로 CLOTH 처리(빈 오판 방지)
   - WM 모드에서 `decision == "EMPTY"`이고 `ema < (rgb_thr - 0.05)`면 **조기 스킵** 및 빈 JSON 반환

2. **포인트클라우드 획득**
   - `get_current_pointcloud(pipeline, (Z_MIN, Z_MAX))`
   - 전역 Z crop: `Z_MIN=0.47`, `Z_MAX=0.90`, 너무 가까워 세탁기 입구 부분이 감지되거나, 너무 먼 부분 crop

3. **기하 참조(실린더) 생성 및 정합**
   - `create_z_axis_cylinder(radius=0.29, z:[0.47,0.90])`로 **드럼 참조 모델** 생성
   - y<0 영역만 골라 **ICP 정합**(`run_icp_y_negative_only`) → 현재 pcd 정렬

4. **후처리/ROI**
   - voxel 1cm 다운샘플 → Z 범위 필터
   - XY **반경 필터**: `FILTER_RADIUS = 0.30`
   - **적응형 이상치 탐지**: `detect_anomalies_adaptive(...)`
   - **날개(회전 물체) 제거**: 검출된 날개 클러스터를 120°, 240° 회전 복제해 근접한 점 제거
   - DBSCAN 기반 클러스터링 및 **컬러 분할**: `split_colorful_clusters(...)`
   - 각 최종 클러스터의 **평균**을 grasp 후보로 사용

5. **보정 및 좌표계 변환**
   - 후보들을 **XY norm ** 기준 정렬(가까운 순)
   - Heuristics: 의류 중심을 더 잘잡기 위한 파라미터, 파이퍼가 좀더 깊숙히 들어갈 수 있도록 조절
     - `Z > 0.7` → `Z = 0.7` 상한
     - `Y ≤ 0.24` → `Y += 0.04`
     - `Y > 0.29` → `Y = 0.29`
   - `C_cam = inv(T_icp) * C_ref`로 ref→camera 변환 후 JSON 출력

6. **시각화**
   - `visualize_pointcloud_with_garments_open3d(...)`로 ref/다운샘플/선택점/클러스터를 Open3D로 표시 -> debugging 해서 실제 점이 어디 찍혔는지 확인할 때 사용

---

## 4) 명령행 옵션

| 옵션 | 기본값 | 설명 |
|---|---:|---|
| `--mode` | (필수) | `wm`만 사용(본 문서) |
| `--json` | - | 마지막 줄에 JSON만 출력하고 종료 |
| `--rgb_ckpt` | `models/best.pt` | RGB 게이트 분류기 체크포인트(TorchScript 또는 state_dict) |
| `--rgb_thr` | 0.35 | CLOTH 상한 임계값(EMA ≥ thr → CLOTH) |
| `--rgb_low_thr` | 0.45 | EMPTY 하한 임계값(EMA < low_thr → EMPTY) ※ 히스테리시스 |
| (코드 상수) `Z_MIN` | 0.47 | Z 크롭 하한 |
| (코드 상수) `Z_MAX` | 0.90 | Z 크롭 상한 |
| (코드 상수) `CYL_RADIUS` | 0.29 | 참조 실린더 반경 |
| (코드 상수) `FILTER_RADIUS` | 0.30 | XY ROI 반경 |

> 주의: `CFG["wm"]["ref_pcd"]`가 정의돼 있지만 WM 모드에서는 **직접 생성한 실린더**를 참조로 사용합니다.

---

## 5) 키 조작 (GUI 모드)

- **`d`**: 감지 실행(1회)
- **`q`**: 종료

> `--json`을 사용하면 키 입력 없이 자동으로 감지 

---

## 6) 체크포인트 포맷 (RGB 게이트)

- TorchScript(`torch.jit.load`) 우선 로딩, 실패 시 state_dict 로드
- state_dict 사용 시 **백본 선택**: `resnet18` 기본
  - 체크포인트의 `args["model"]`를 읽어 자동 선택(없으면 `resnet18`)
- 클래스 매핑: 기본 `{0:"empty",1:"cloth"}`, ckpt에 `class_to_idx`/메타 json이 있으면 갱신

---

## 7) 트러블슈팅

- **RGB 게이트 로드 실패**: TorchScript 형식/경로 확인. 실패 시 state_dict 경로가 맞는지, 모델명이 일치하는지 확인.
- **빈인데도 감지됨**: `--rgb_thr`를 올리거나 `--rgb_low_thr`를 낮춰 히스테리시스 폭을 넓히세요.
- **좌표가 이상함**: RealSense depth‑to‑meter 스케일, Z 크롭 범위(`Z_MIN/Z_MAX`), Y 보정 로직 확인.
- **Open3D 창이 뜨지 않게 하려면**: 원격/Headless 환경에서는 해당 시각화 호출을 주석 처리하거나 X forwarding 설정이 필요합니다.
- **의류가 드럼 경계 근처**: `FILTER_RADIUS`(0.30)와 DBSCAN 파라미터(eps=0.037, min_samples=30)를 조절.

---


---

## 9) 구현 포인트 

- RGB 게이트 EMA: `ema = decay*ema + (1-decay)*p`, `decay=0.7`
- 조기 스킵 로그 예: `[wm] rgb-empty-skip p=0.12 ema=0.18 thr=0.60`
- ICP: y<0 절반 포인트만으로 정합(드럼 대칭/가림 대응)
- 날개 제거: 날개 클러스터를 120°, 240° 회전 복제 → 근접 점 제거로 false positive 감소
- 최종 센터 보정: Z 상한/ Y 범위 보정으로 집게 충돌-risk 감소
