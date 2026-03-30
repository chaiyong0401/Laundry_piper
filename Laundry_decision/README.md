# run_bf 
basket, wm 입구 감지하는 run_basket.py, run_failure.py 포함

# ROS2 Action Router for Laundry Detection. (node2.py)
외부 감지 파이프라인(run_laundry/run_basket/run_failure)을 **하나의 ROS2 Action 서버**로 라우팅하는 노드.

- Action 이름: `target_pose`  
- Action 타입: `piper_with_gripper_moveit/TargetPose`  
- 동작: Goal로 `{start: true, mode: "wm|basket|failure|failure_bg"}`를 받으면 해당 스크립트를 실행하고, JSON 결과를 파싱해 `Pose`를 반환

---

## 주요 기능
- `mode == "wm"`        → `run_laundry.py --mode wm --json`
- `mode == "basket"`    → `run_basket.py  --mode basket  --ckpt ... --intr_json ... --json`
- `mode == "failure"`   → `run_failure.py --mode failure --ckpt ... --intr_json ... --json`
- `mode == "failure_bg"`→ `run_failure.py --mode failure_bg --ckpt ... --intr_json ... --json`
- 외부 스크립트의 **표준출력**에서 마지막 JSON 라인을 파싱: `{"success": bool, "centers": [[x,y,z], ...], ...}`
- 첫 번째 center를 `geometry_msgs/Pose`로 래핑해 Result로 반환

---

## 요구사항
- 외부 스크립트 존재:
  - `RUN_LAUNDRY_PY`: `/home/cheon/Documents/run_laundry/run_laundry.py`
  - `RUN_BASKET_PY` : `/home/cheon/Documents/run_bf/run_basket.py`
  - `RUN_FAILURE_PY`: `/home/cheon/Documents/run_bf/run_failure.py`
- 각 스크립트는 **마지막 출력 라인**에 JSON을 한 줄로 출력해야 함(예: `print(json.dumps(result))`)

---

## 설정(코드 상단 상수) 각 폴더의 구조에 맞춰서 변경 필요. 
```python
PYTHON_BIN   = sys.executable or "python3"
TIMEOUT_SEC  = 660.0

RUN_LAUNDRY_DIR = "/home/cheon/Documents/run_laundry"
RUN_LAUNDRY_PY  = os.path.join(RUN_LAUNDRY_DIR, "run_laundry.py")

RUN_FAILURE_DIR = "/home/cheon/Documents/run_bf"
RUN_FAILURE_PY  = os.path.join(RUN_FAILURE_DIR, "run_failure.py")

RUN_BASKET_DIR  = "/home/cheon/Documents/run_bf"
RUN_BASKET_PY   = os.path.join(RUN_BASKET_DIR, "run_basket.py")

CKPT_PATH_F = "/home/cheon/Documents/run_bf/dataset_rgb_sam2/unet_best.pt"
CKPT_PATH_B = "/home/cheon/Documents/run_bf/dataset_rgb_sam2/unet_best_multi.pt"
INTR_JSON   = "/home/cheon/Documents/run_bf/intrinsics_color_640x480.json"
```


> **주의:** intrinsics color 사용 

---
python node2.py


## Goal 전송 예시
```
# wm
ros2 action send_goal /target_pose piper_with_gripper_moveit/action/TargetPose "{start: true, mode: 'wm'}"

# basket
ros2 action send_goal /target_pose piper_with_gripper_moveit/action/TargetPose "{start: true, mode: 'basket'}"

# failure
ros2 action send_goal /target_pose piper_with_gripper_moveit/action/TargetPose "{start: true, mode: 'failure'}"

# failure_bg
ros2 action send_goal /target_pose piper_with_gripper_moveit/action/TargetPose "{start: true, mode: 'failure_bg'}"
```

> `TargetPose.action`

---

## 외부 스크립트의 JSON 계약
- **필수**: `success: bool`, `centers: [[x,y,z], ...]`
- **선택**: `occupied_cells` (failure 모드일 때)
- 예시 출력(한 줄):
```json
{"success": true, "centers": [[0.12, -0.04, 0.53]], "occupied_cells": [123.0, 456.0]}
```
노드는 첫 번째 center만 사용 `target_pose.position`에 대입, `orientation.w = 1.0`로 반환

---


# RGB-SAM2 Click Tool (Depth-guided), (rgb_sam2_click.py)

세탁기/바구니 등 실내 RGB-D 데이터셋에서 **클릭 기반 SAM2 세그멘테이션**을 수행하고,  
선택적으로 **빈 씬(empty)과의 depth 차(diff)를 저장**하는 라벨링 도구.

- 좌클릭: 긍정 점(포함) 추가  
- 우클릭: 부정 점(제외) 추가  
- `e`: SAM2 실행 → 마스크 생성/갱신  
- `o`/`k`: 마스크 열기/닫기(모폴로지)  
- `s`: 결과 저장  
- `n`/`v`: 다음/이전 샘플  
- `m`: 마스크 초기화, `c`: 클릭 초기화, `q`: 종료

---

---

## 데이터 디렉터리 구조

### A) `in_root` 바로 아래에 폴더가 있는 경우
```
in_root/
 ├─ rgb/        # RGB 이미지(.png/.jpg)
 └─ depth_img/  # Depth 이미지(.png, uint16=mm 또는 float32/64=m)
```

### B) `in_root/<domain>` 구조를 쓰는 경우
```
in_root/
 └─ dataset_wm/cloth/
     ├─ rgb/
     └─ depth_img/
```

빈 씬(empty) depth는 자동/수동으로 찾음:
```
in_root/
 └─ dataset_wm/empty/
     └─ depth_img/
```

> Depth 파일은 `uint16(mm)` 또는 `float(m)` 지원. `uint16`이고 최대값이 50보다 크면 **자동으로 m 단위로 /1000** 변환함.

---

## command

```bash
python rgb_sam2_click.py   --in_root /home/cheon/Documents/train_all/dataset   --out_root /home/cheon/Documents/train_all/dataset_rgb_sam2/wm   --sam2_ckpt /home/cheon/Documents/train_all/checkpoints/sam2.1_hiera_base_plus.pt   --sam2_cfg  configs/sam2.1/sam2.1_hiera_b+.yaml   --device cuda   --enable_zmask --zmax 0.65   --save_diff_cur   --ref_match cached --match_down 4 --ref_cache /home/cheon/Documents/run_failure/ref_cache_basket.json   --align affine   --depth_suffix _depth --rgb_suffix _rgb --strip_suffix --rgb_exts .png,.jpg   --domain dataset_wm/cloth   --empty_domain dataset_wm/empty
```

---

## 사용 방법 (워크플로우)

1. 프로그램 시작 → 첫 depth 샘플이 뜸. (RGB가 없으면 depth colormap으로 대체 표시)
2. 관심 물체에 **좌클릭**(포함) / **우클릭**(제외) 점을 몇 개 찍기
3. `e` 키로 SAM2 실행 → 최고 점수 마스크 선택 후 누적 병합  
   - `--enable_zmask`면 `0 < z ≤ zmax` 영역으로도 마스크 클리핑
4. 필요하면 `o`(open) / `k`(close)로 잡티 정리, `m`으로 마스크 초기화, `c`로 클릭 초기화
5. `s`로 저장 → 라벨/메타/디프/디버그 이미지가 생성
6. `n` 다음 / `v` 이전 샘플로 이동

---

## 출력물

저장하면 `out_root/samples/<stem>/`에 생성:

```
out_root/
 └─ samples/<stem>/
     ├─ mask.png          # 최종 이진 마스크(0/255)
     ├─ dbg_overlay.png   # 엣지(빨간색) 오버레이 시각화
     ├─ rgb.png           # 현재 프레임 RGB 저장(BGR→RGB→BGR 경유, 최종 BGR PNG)
     ├─ cur.npy           # (선택) 현재 depth (float32, meters)
     ├─ diff.npy          # (선택) |cur - aligned_ref| depth 차이
     └─ meta.json         # 클릭/경로/zmax/ref_path 등 메타데이터
```

- `diff.npy`, `cur.npy`는 `--save_diff_cur`일 때만 저장
- `ref_cache` 경로를 주면 빈 씬 매칭 인덱스를 json에 누적 저장

---

## 키보드 단축키

| 키 | 기능 |
|---|---|
| 좌클릭 | 긍정 점 추가(포함) |
| 우클릭 | 부정 점 추가(제외) |
| `e` | SAM2 실행/갱신 |
| `o` | 모폴로지 **open**(노이즈 제거) |
| `k` | 모폴로지 **close**(구멍 메움) |
| `m` | 마스크 초기화 |
| `c` | 클릭 초기화 |
| `s` | 결과 저장 |
| `n` / `v` | 다음 / 이전 샘플 |
| `q` | 종료 |

---

## 주요 옵션 설명

### 입력/출력
- `--in_root` (필수): 입력 루트
- `--out_root` (필수): 출력 루트(샘플들이 이 아래에 저장)

### SAM2
- `--sam2_cfg`, `--sam2_ckpt` (필수): SAM2 설정/가중치 경로
- `--device`: `cuda` 권장. 없으면 자동 판별

### 도메인/폴더명
- `--domain`: `in_root/<domain>` 구조를 사용할 때 하위 폴더명  
  예) `dataset_wm/cloth` (기본값은 코드에 정의된 choices 참고)
- `--rgb_dirname`, `--depth_dirname`: 기본 `rgb`, `depth_img`

### RGB/Depth 매칭
- `--depth_suffix`, `--rgb_suffix`: 파일명 스템 매칭 시 사용할 접미어(예: `000_depth.png` ↔ `000_rgb.jpg`)
- `--strip_suffix`: depth 접미어를 제거한 스템도 후보로 사용
- `--rgb_exts`: RGB 확장자 후보 리스트

### Z-mask
- `--enable_zmask`: depth 범위로 마스크 클리핑 활성화
- `--zmax`: 최대 z(m). 0 < z ≤ zmax 만 사용

### Empty 참조/정합/차이 저장
- `--save_diff_cur`: `cur.npy`, `diff.npy` 저장. **빈 씬 depth 폴더 필수**
- `--empty_dir`: 빈 씬 루트를 직접 지정(그 아래 `depth_dirname`가 있어야 함)
- `--empty_domain`: 자동 탐색 시 사용할 빈 씬 폴더명(예: `dataset_wm/empty`)
- `--ref_match`: 빈 씬 선택 방식
  - `cached`: 캐시 있으면 그대로, 없으면 저해상도 MAE로 베스트 선택 후 캐시
  - `first`: 첫 번째만 사용
  - `idx`: `--ref_idx`로 지정한 인덱스 사용
  - `best_ecc`: 캐시 무시하고 저해상도 MAE로 매번 베스트
- `--ref_cache`: 매칭 결과 캐시 JSON 경로
- `--match_down`: 매칭을 위한 다운스케일 배수(4~8 권장)
- `--align`: `affine`(ECC) 또는 `none`  
  - `affine`이면 `cv2.findTransformECC`로 깊이를 정합 후 차이 계산

---

## 내부 동작 포인트

- **Depth 로딩**: `uint16`이면 mm→m 자동 변환(/1000). float32/64는 그대로 사용
- **SAM2 추론과 누적**: 클릭 → `predictor.predict()` → 최고 점수 마스크 선택 → 기존 마스크와 OR 병합  
  `--enable_zmask`이면 z 범위로 마스크 클리핑
- **엣지 표시**: `dbg_overlay.png`에는 Canny 엣지를 빨간색으로 오버레이
- **빈 씬 선택**: 저해상도(다운스케일) depth로 MAE를 구해 가장 가까운 empty를 고름(캐시 가능)
