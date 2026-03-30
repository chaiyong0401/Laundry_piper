# RGB‑D Laundry Grasp Detection
**run_basket.py / run_failure.py**  
RealSense D435i의 RGB‑D 스트림을 받아 경량UNet(+SE)로 의류 마스크를 예측하고, 중심/방향/깊이로 **그립 포인트(XYZ)**를 산출.  

---

## 1) 요구사항
- 학습 체크포인트: `*.pt`/`*.pth`
- 카메라 내정수 JSON (예: `intrinsics_color_640x480.json`) 

> **정렬과 내정수 매칭 중요!**   
> - `run_failure.py, run_basket.py`:  **항상 color 정렬**을 사용. **color intrinsics 권장**.

---

## 2) 빠른 실행 예시

### A. 바구니(basket) — 시각화 모드
```bash
python run_basket.py \
  --ckpt /home/cheon/Documents/run_bf/dataset_rgb_sam2/unet_best_multi.pt \
  --intr_json /home/cheon/Documents/run_bf/intrinsics_color_640x480.json \
  --mode basket \
  --size 384 --thr 0.45 \
  --zmax_basket 0.6 --zmax_failure 0.5 \
  --diff_mode bilateral \
  --vote --vote_k 10 --vote_min_area 150 --vote_min_dt 3 --vote_require_curr \
  --bg_path /home/cheon/Documents/run_bg/laundry_bg/bg_basket.npy
```

### B. 실패(failure) — 시각화 모드
```bash
python run_failure.py \
  --mode failure \
  --ckpt /home/cheon/Documents/run_bf/dataset_rgb_sam2/unet_best.pt \
  --intr_json /home/cheon/Documents/run_bf/intrinsics_color_640x480.json \
  --size 384 --zmax 0.5 --thr 0.45 \
  --diff_mode blend \
  --vote --vote_k 10 --vote_min_area 150 --vote_min_dt 3 --vote_require_curr \
  --stale_sec 1.0 \
  --save_dir /home/cheon/Documents/run_bf/_dbg \
  --save_best_only --prefix failure --show
```

### C. JSON 모드(ROS2 연동) — basket/failure
```bash
# basket / JSON 한 줄 출력
python run_basket.py \
  --ckpt /home/cheon/Documents/run_bf/dataset_rgb_sam2/unet_best_multi.pt \
  --intr_json /home/cheon/Documents/run_bf/intrinsics_color_640x480.json \
  --mode basket --json --max_wait_sec 3.0 \
  --vote --vote_k 10

# failure / JSON 한 줄 출력
python run_failure.py \
  --mode failure --json \
  --ckpt /home/cheon/Documents/run_bf/dataset_rgb_sam2/unet_best.pt \
  --intr_json /home/cheon/Documents/run_bf/intrinsics_color_640x480.json \
  --size 384 --zmax 0.5 --thr 0.45 \
  --diff_mode blend \
  --vote --vote_k 10 --vote_min_area 150 --vote_min_dt 3 --vote_require_curr \
  --stale_sec 1.0 \
  --save_dir /home/cheon/Documents/run_bf/_dbg \
  --save_best_only --prefix failure
```

### D. 배경 저장(failure_bg)
```bash
python run_failure.py \
  --mode failure_bg \
  --ckpt /home/cheon/Documents/run_bf/dataset_rgb_sam2/unet_best.pt \
  --intr_json /home/cheon/Documents/run_bf/intrinsics_color_640x480.json \
  --bg_frames 20 \
  --bg_path /home/cheon/Documents/run_bf/laundry_bg/bg_failure.npy
```

---

## 3) 공통 파이프라인 개요
1. **RealSense 정렬**: `run_basket`는 `--align` 선택, `run_failure`는 color 고정.  
2. **입력 구성**:  
   - `diff`: 배경/양방향필터 기반 깊이 차 (`--diff_mode`), [0,0.1]m → [0,1] 정규화  
   - `z`: 깊이(m)를 `zmax`로 클리핑/정규화  
   - `rgb(3)`: 0~1 정규화  
   - `domain_plane`(basket 전용): basket=0, failure=1 상수 채널 → 총 **6채널**(basket) / **5채널**(failure)
3. **추론**: TinyUNet(SE) → 1채널 로짓 → 시그모이드/이진화(`--thr`) → 오픈(3×3) → 최대 연결요소
4. **중심/방향**: DT(distance transform) 최대값을 중심(u,v), PCA로 주방향(θ) 계산
5. **깊이 샘플링**: 중심 주변 5×5 유효 깊이 중앙값 → Z(m)
6. **픽셀→미터**: `uvz_to_xyz(u,v,z,fx,fy,ppx,ppy)`
7. **투표/게이팅**: 버퍼(`--vote_k`) 중앙값/원형평균, `--stale_sec` 내에서만 유효 처리
8. **출력**: 시각화 또는 JSON 1줄

---

## 4) JSON 스키마

### run_basket.py
```jsonc
{
  "success": true,
  "empty": false,
  "mode": "basket",
  "thr": 0.45,
  "zmax": 0.6,
  "centers": [[0.004, -0.020, 0.368]],  // 카메라 좌표계 [X,Y,Z] (m), 1개
  "pixel_center": [320, 240],           // [u,v] (px)
  "theta": -0.34,                       // 라디안
  "area": 1540,                         // 최대 연결요소 픽셀수
  "dt": 7.2                             // DT 피크 값(픽셀)
}
```

### run_failure.py
```jsonc
{
  "success": true,
  "centers": [[0.004, -0.020, 0.368]],  // 카메라 좌표계 [X,Y,Z] (m), 1개
  "grid": {"nx": 4, "ny": 1},
  "section_idx": 1, "section_row": 0, "section_col": 1,
  "occupied_cells": [0,1],              // min_px 이상 픽셀 포함 셀 인덱스들
  "center_uv": [320, 240]
}
// 타임아웃: {"success": false, "reason": "timeout"}
```

> 두 스크립트 모두 **마지막 줄**에만 JSON을 출력합니다. 외부에서는 **마지막 한 줄만 파싱**하면 됩니다.

---

## 5) 공통 옵션
basket은 6채널, failure은 5채널 -> failure과 basket을 같이 학습 시킬려했으나, failure에서는 견고하게 작동하지 못함. 
| 옵션 | 기본값 | 설명 |
|---|---:|---|
| `--ckpt` | (필수) | UNet(SE) 체크포인트(.pt/.pth). 내부에 `model`, `c_in`, `base`, `use_se` 메타 포함  |
| `--intr_json` | (필수) | 카메라 내정수 JSON: `fx, fy, ppx/cx, ppy/cy`를 해석 |
| `--size` | 384 | 입력 크기 |
| `--thr` | 0.45 | 시그모이드 임계값 |
| `--device` | `cuda` | `cuda` 또는 `cpu` |
| `--diff_mode` | `blend`(failure)/`bilateral`(예시) | `bg`/`bilateral`/`min`/`blend` |
| `--vote` | - | 투표 기반 안정화 활성화 |
| `--vote_k` | 10 | 투표 버퍼 길이 |
| `--vote_min_area` | 150 | 최소 영역(픽셀) |
| `--vote_min_dt` | 3.0 | 최소 DT 피크(픽셀) |
| `--vote_require_curr` | - | 현재 프레임이 조건 만족해야만 표 반영 |
| `--stale_sec` | 1.0 | 마지막 유효 검출 이후 유효기간 |

### run_basket 전용
| 옵션 | 기본값 | 설명 |
|---|---:|---|
| `--mode` | `basket` | `basket` 또는 `failure` (도메인 채널과 zmax 선택에 사용) |
| `--zmax_basket` | 0.6 | basket 장면 Z 상한(m) |
| `--zmax_failure` | 0.5 | failure 장면 Z 상한(m) |
| `--align` | `color` | `color` 또는 `depth` 정렬 선택 |
| `--bg_path` | `/home/.../bg_failure.npy`* | 배경 depth 경로(.npy). 예시처럼 **작업별로 구분 경로** 권장 |
| `--json` | - | JSON 모드 활성화 |
| `--max_wait_sec` | 3.0 | JSON 모드 최대 대기 시간 |

\* 코드 기본값은 failure용 경로로 되어 있습니다. basket 전용 배경을 쓰려면 `bg_basket.npy` 등으로 **명시 지정**하세요.

### run_failure 전용
| 옵션 | 기본값 | 설명 |
|---|---:|---|
| `--mode` | `failure` | `failure_bg`(배경 캡처) 또는 `failure`(검출) |
| `--zmax` | 0.5 | Z 상한(m) |
| `--show` | - | 시각화 창 표시 |
| `--save_dir` | `/home/.../_dbg` | 디버그 저장 폴더 |
| `--save_all` | - | 프레임마다 저장 |
| `--save_best_only` | - | 베스트 프레임만 저장 |
| `--save_on_success_only` | - | 성공시에만 저장 |
| `--prefix` | `failure` | 파일 접두어 |
| `--bg_frames` | 20 | 배경 캡처 프레임 수 |
| `--bg_path` | `/home/.../bg_failure.npy` | 배경 depth 경로(.npy) |
| `--max_sec` | 30.0 | 전체 최대 처리 시간(시각화/JSON 공통) |
| `--grid` | 3 | 정사각 그리드 기본(nx=ny=grid) |
| `--grid_w`/`--grid_h` | 4 / 1 | 가로/세로 칸 수(설정 시 `--grid` 무시) |
| `--cell_px` | 80 | 셀 점유 판정 최소 픽셀 수 |
| `--draw_grid` | - | 시각화 그리드/섹션 표시 |

---

## 6) 단축키(시각화 모드)
- 공통: `q` 종료, `]`/`[` 임계값 `thr` ±0.02, `o` 오버레이 토글  
- `run_basket`: `b` 배경 캡처/저장(`--bg_path` 지정 시 저장)  
- `run_failure`: `--draw_grid` 옵션으로 그리드/섹션 시각화

> `run_failure.py`는 시작 후 내부적으로 **초기 30루프 웜업**이 있어 첫 수 초 동안 결과가 지연될 수 있습니다.

---

## 7) 디버그 output
- `run_basket.py`: `--save_dir` 지정 시 프레임 단위로 `*_rgb.png`, `*_depth.png`, `*_prob.png`, `*_mask.png`, `*_overlay.png`, `*_raw.npz` 저장. `--save_best_only`로 성공 시점만 저장 가능.

- `run_failure.py`: `--save_dir`와 `--save_*` 플래그로 `*_rgb.jpg`, `*_mask.png`, `*_overlay.jpg`, `*_meta.json` 저장. `--save_best_only` 사용

---

## 8) 트러블슈팅
- **배경 불러오기 실패**: 경로/권한 확인. 없으면 basket `b` 키 또는 failure의 `failure_bg` 모드로 캡처. 일단 저장은 해둠. 
- **JSON 파싱 문제**: 마지막 줄만. 중간 디버그 출력은 무시.
- **검출 불안정**: `--vote_k`↑, `--vote_min_area`/`--vote_min_dt` 조정, `--thr` 조금 낮추기, `--diff_mode`를 `blend`로 테스트.

---
