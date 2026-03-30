"""
vis mode: 
python run_basket.py \
  --ckpt /home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/unet_best_multi.pt \
  --intr_json /home/dyros/mcy_ws/Laundry_decision/run_bf/intrinsics_color_640x480.json \
  --mode basket \
  --size 384 --thr 0.45 \
  --zmax_basket 0.6 --zmax_failure 0.5 \
  --diff_mode bilateral \
  --vote --vote_k 10 --vote_min_area 150 --vote_min_dt 3 --vote_require_curr \
  --bg_path /home/dyros/mcy_ws/Laundry_decision/run_bg/laundry_bg/bg_basket.npy \ 

  python run_basket.py \
  --ckpt /home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/unet_best_multi.pt \
  --intr_json /home/dyros/mcy_ws/Laundry_decision/run_bf/intrinsics_color_640x480.json \
  --mode basket \
  --size 384 --thr 0.45 \
  --zmax_basket 0.6 --zmax_failure 0.5 \
  --diff_mode bilateral \
  --vote --vote_k 10 --vote_min_area 150 --vote_min_dt 3 --vote_require_curr \
  --bg_path /home/dyros/mcy_ws/Laundry_decision/run_bg/laundry_bg/bg_basket.npy \ 
  

--json mode: for ros2
python realtime_grasp_runner_rgb2_gated.py \
  --ckpt .../unet_best_multi.pt --intr_json .../intrinsics_depth_640x480.json \
  --mode failure --json --max_wait_sec 3.0 --vote --vote_k 10
"""
from pathlib import Path  
import os, json, time, argparse, math
from collections import deque
import numpy as np
import cv2
import torch, torch.nn as nn

from sklearn.cluster import MeanShift
from collections import defaultdict

# ---------------- Model (TinyUNet) ----------------
class SE(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, max(1, ch // r), bias=False), nn.ReLU(True),
            nn.Linear(max(1, ch // r), ch, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        b,c,_,_ = x.shape
        w = self.fc(self.avg(x).view(b,c)).view(b,c,1,1)
        return x * w

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.se = SE(out_ch) if use_se else nn.Identity()
    def forward(self, x):
        x = self.block(x)
        return self.se(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=True):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, use_se)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, use_se)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dh = x2.size(2)-x1.size(2); dw = x2.size(3)-x1.size(3)
        x1 = nn.functional.pad(x1, [dw//2, dw-dw//2, dh//2, dh-dh//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class TinyUNet(nn.Module):
    def __init__(self, c_in=6, base=32, use_se=True):
        super().__init__()
        self.inc = DoubleConv(c_in, base, use_se)
        self.d1 = Down(base, base*2, use_se)
        self.d2 = Down(base*2, base*4, use_se)
        self.d3 = Down(base*4, base*8, use_se)
        self.u1 = Up(base*8, base*4, use_se)
        self.u2 = Up(base*4, base*2, use_se)
        self.u3 = Up(base*2, base, use_se)
        self.outc = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.d1(x1); x3 = self.d2(x2); x4 = self.d3(x3)
        x = self.u1(x4, x3); x = self.u2(x, x2); x = self.u3(x, x1)
        return self.outc(x)

# --------------- Camera helpers -------------------
def load_intrinsics(path):
    J = json.load(open(path, 'r'))
    fx = float(J.get('fx') or J.get('intrinsics',{}).get('fx'))
    fy = float(J.get('fy') or J.get('intrinsics',{}).get('fy'))
    ppx = float(J.get('ppx') or J.get('cx') or J.get('intrinsics',{}).get('ppx'))
    ppy = float(J.get('ppy') or J.get('cy') or J.get('intrinsics',{}).get('ppy'))
    return fx, fy, ppx, ppy

def uvz_to_xyz(u, v, z, fx, fy, ppx, ppy):
    X = (u - ppx) * z / fx
    Y = (v - ppy) * z / fy
    return X, Y, z

# --------------- Pre/Post process -----------------
def compute_diff(depth_m, bg_m, mode='min'):
    """
    mode: 'bg' | 'bilateral' | 'min' | 'blend'
    """
    depth = depth_m.astype(np.float32)
    valid = depth > 0

    base = cv2.bilateralFilter(depth, d=5, sigmaColor=0.05, sigmaSpace=7)
    diff_bi = np.abs(base - depth); diff_bi[~valid] = 0.0

    diff_bg = np.zeros_like(depth, np.float32)
    have_bg = (bg_m is not None)
    if have_bg:
        bg = bg_m.astype(np.float32)
        both = valid & (bg > 0)
        diff_bg[both] = np.abs(bg[both] - depth[both])

    if mode == 'bg':
        diff = diff_bg
    elif mode == 'bilateral':
        diff = diff_bi
    elif mode == 'blend':
        diff = diff_bi.copy()
        if have_bg:
            both = (diff_bg > 0)
            diff[both] = diff_bg[both]
    else:  # 'min'
        diff = diff_bi.copy()
        if have_bg:
            both = (diff_bg > 0)
            diff[both] = np.minimum(diff_bg[both], diff_bi[both])

    return (np.clip(diff, 0.0, 0.1) / 0.1).astype(np.float32)

def preprocess_inputs(depth_m, color_rgb, bg_m, size, zmax, mode_name, diff_mode):
    """
    return 6ch tensor [diff, z, rgb(3), domain_plane] resized to (size,size)
    - diff_mode: CLI에서 받은 모드를 그대로 사용
    """
    diff = compute_diff(depth_m, bg_m, mode=diff_mode)
    z = np.clip(depth_m, 0.0, zmax) / max(1e-6, zmax)

    diff_r = cv2.resize(diff, (size, size), interpolation=cv2.INTER_NEAREST)
    z_r    = cv2.resize(z,    (size, size), interpolation=cv2.INTER_NEAREST)

    if color_rgb is None:
        rgb_r = np.zeros((size, size, 3), dtype=np.float32)
    else:
        rgb = color_rgb.astype(np.float32) / 255.0
        rgb_r = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)

    x = np.stack([diff_r, z_r], axis=0)
    x = np.concatenate([x, np.transpose(rgb_r, (2,0,1))], axis=0)  # (5,H,W)

    dom_id = 0 if mode_name=='basket' else 1  # basket=0, failure=1
    dom_plane = np.full((1, size, size), float(dom_id), dtype=np.float32)
    x = np.concatenate([x, dom_plane], axis=0)  # (6,H,W)
    return x.astype(np.float32)

def postprocess_mask(logits, thr, out_hw):
    prob = torch.sigmoid(logits)[0,0].cpu().numpy()
    m = (prob >= thr).astype(np.uint8)
    H, W = out_hw
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))) # debug_mcy

    # debug_mcy
    kernel_size = 65
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    m = cv2.morphologyEx(m,cv2.MORPH_OPEN, kernel)

    # debug_mcy_color
    # kernel_size = 5
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)

    return m

def largest_cc(mask):
    num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    if num <= 1: return None
    best_label, best_area = None, 0
    for lb in range(1, num):
        area = (labels==lb).sum()
        if area > best_area:
            best_area, best_label = area, lb
    return (labels==best_label).astype(np.uint8)


def find_all_detections(mask, min_area=150, min_dt=3.0):
    """
    마스크에서 min_area와 min_dt 기준을 만족하는 모든 객체를 찾아
    [ {center, theta, area, dt_max}, ... ] 리스트를 반환합니다. (디버그 프린트 추가됨)
    """
    num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    detections = []
    
    # ------------------ [디버그 프린트 1] ------------------
    # 딥러닝 마스크에서 찾은 '날것'의 덩어리 갯수
    print(f"\n[Debug] Total raw components found (incl. noise): {num - 1}")
    # ----------------------------------------------------

    if num <= 1:
        return detections # 빈 리스트 반환

    for lb in range(1, num): # 0번(배경)은 제외
        cc = (labels == lb).astype(np.uint8)
        area = int(cc.sum())
        
        # ------------------ [디버그 프린트 2] ------------------
        # 각 덩어리의 면적을 검사 전에 출력
        print(f"  > Checking component {lb}: area={area} (min_area={min_area})")
        # ----------------------------------------------------
        
        # 1. 최소 면적 검사
        if area < min_area:
            print(f"    └─ REJECTED (Area < {min_area})") # 면적 탈락
            continue
        
        # 2. 거리 변환(dt) 검사
        dt = cv2.distanceTransform(cc, cv2.DIST_L2, 3)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(dt)
        cy, cx = maxLoc[1], maxLoc[0]
        dt_max = float(maxVal)

        # ------------------ [디버그 프린트 3] ------------------
        # 면적을 통과한 객체의 dt_max 값 출력
        print(f"    └─ Checking dt_max={dt_max:.2f} (min_dt={min_dt})")
        # ----------------------------------------------------

        if dt_max < min_dt:
            print(f"      └─ REJECTED (dt_max < {min_dt})") # dt_max 탈락
            continue

        # 3. 각도(theta) 계산 (생략)
        ys, xs = np.where(cc > 0)
        if len(xs) < 5:
            theta = 0.0
        else:
            x = xs.astype(np.float32); y = ys.astype(np.float32)
            x -= x.mean(); y -= y.mean()
            cov_xx = (x*x).mean(); cov_yy = (y*y).mean(); cov_xy = (x*y).mean()
            theta = 0.5 * math.atan2(2*cov_xy, (cov_xx - cov_yy))
        
        print(f"      └─ PASSED. (Added to detections list)") # 최종 통과
        
        detections.append({
            "center": (cx, cy),
            "theta": theta,
            "area": area,
            "dt_max": dt_max
        })
    
    # 신뢰도(dt_max)가 높은 순으로 정렬
    detections.sort(key=lambda d: d['dt_max'], reverse=True)
    
    return detections

# debug_clustering
def find_clusters_by_color(color_rgb, depth_m, mask, 
                           fx, fy, ppx, ppy, 
                           min_area_per_cluster=150,
                           color_bandwidth=30):
    """
    하나의 큰 마스크(mask) 내부를 색상으로 분리(MeanShift)하여,
    각 색상 클러스터의 3D 중심점을 찾습니다.
    """
    
    # 1. 마스크에 해당하는 픽셀의 (u,v) 좌표와 RGB 값 추출
    v_coords, u_coords = np.where(mask > 0)
    if len(u_coords) == 0:
        return []

    # 픽셀이 너무 많으면 MeanShift가 느리므로, 샘플링
    max_pixels = 50000
    if len(u_coords) > max_pixels:
        indices = np.random.choice(len(u_coords), max_pixels, replace=False)
        v_coords, u_coords = v_coords[indices], u_coords[indices]

    # 2. 색상 정보 추출 (RGB -> Lab 변환)
    # Lab 색상 공간의 'a'(green-red)와 'b'(blue-yellow) 채널만 사용
    pixels_rgb = color_rgb[v_coords, u_coords].reshape(-1, 1, 3)
    pixels_lab = cv2.cvtColor(pixels_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    pixels_ab = pixels_lab[:, 1:].astype(np.float32) # 'a'와 'b' 채널

    # 3. 색상 클러스터링 (MeanShift)
    # bandwidth 값 튜닝이 중요 (예: 20~40)
    ms = MeanShift(bandwidth=color_bandwidth, bin_seeding=True)
    ms.fit(pixels_ab)
    labels = ms.labels_ # 각 픽셀이 몇 번 클러스터(색상 그룹)에 속하는지
    unique_labels = np.unique(labels)
    
    print(f"[ColorCluster] Found {len(unique_labels)} raw color clusters.")

    # 4. 각 클러스터(색상 그룹)의 3D 중심점 계산
    final_centers_xyz = []
    
    # (u,v) 좌표와 label을 딕셔너리로 그룹화
    pixel_groups = defaultdict(list)
    for u, v, label in zip(u_coords, v_coords, labels):
        pixel_groups[label].append((u, v))

    for label in unique_labels:
        pixels_uv = pixel_groups[label]
        
        # 5. 클러스터 필터링 (너무 작은 노이즈 제거)
        if len(pixels_uv) < min_area_per_cluster:
            print(f"  > Cluster {label} REJECTED (size {len(pixels_uv)} < {min_area_per_cluster})")
            continue
        
        # 6. 클러스터의 2D 중심(median) 찾기
        arr_uv = np.array(pixels_uv)
        u_med = int(np.median(arr_uv[:, 0]))
        v_med = int(np.median(arr_uv[:, 1]))
        
        # 7. 2D 중심점 -> 3D 좌표 변환
        y0, y1 = max(0, v_med-2), min(depth_m.shape[0], v_med+3)
        x0, x1 = max(0, u_med-2), min(depth_m.shape[1], u_med+3)
        local_depth = depth_m[y0:y1, x0:x1]
        valid_depths = local_depth[local_depth > 0]
        
        if valid_depths.size > 0:
            z_m = float(np.median(valid_depths))
            X, Y, Z = uvz_to_xyz(u_med, v_med, z_m, fx, fy, ppx, ppy)
            final_centers_xyz.append((X, Y, Z))
            print(f"  > Cluster {label} ACCEPTED. Center: ({X:.3f}, {Y:.3f}, {Z:.3f})")

    return final_centers_xyz
# --- [새 함수 끝] ---

def analyze_mask(mask):
    """return center(x,y), angle(rad), area(px), dt_max(px)."""
    cc = largest_cc(mask)
    if cc is None:
        return None, None, 0, 0.0
    dt = cv2.distanceTransform(cc, cv2.DIST_L2, 3)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(dt)
    cy, cx = maxLoc[1], maxLoc[0]
    dt_max = float(maxVal)
    area = int(cc.sum())

    ys, xs = np.where(cc>0)
    if len(xs) < 5:
        theta = 0.0
    else:
        x = xs.astype(np.float32); y = ys.astype(np.float32)
        x -= x.mean(); y -= y.mean()
        cov_xx = (x*x).mean(); cov_yy = (y*y).mean(); cov_xy = (x*y).mean()
        theta = 0.5 * math.atan2(2*cov_xy, (cov_xx - cov_yy))
    return (cx, cy), theta, area, dt_max

def circ_mean_theta(thetas):
    if len(thetas)==0: return 0.0
    s = sum([math.sin(t) for t in thetas])
    c = sum([math.cos(t) for t in thetas])
    return math.atan2(s, c)

def draw_overlay(color_bgr, mask, center, theta, xyz=None, thr=0.5, voted=False, stale=False, votes_len=0, vote_k=0):
    out = color_bgr.copy()
    if mask is not None:
        overlay = (np.dstack([np.zeros_like(mask), mask*255, np.zeros_like(mask)])).astype(np.uint8)
        overlay = cv2.GaussianBlur(overlay, (5,5), 0)
        out = cv2.addWeighted(out, 1.0, overlay, 0.35, 0)
    if center is not None:
        cx, cy = center
        cv2.circle(out, (int(cx), int(cy)), 5, (0,0,255), -1)
        L = 40
        x2 = int(cx + L*math.cos(theta)); y2 = int(cy + L*math.sin(theta))
        x3 = int(cx - L*math.cos(theta)); y3 = int(cy - L*math.sin(theta))
        cv2.line(out, (x3,y3), (x2,y2), (255,0,0), 2)
    t = f"thr={thr:.2f}"
    if voted: t += f" | VOTE {votes_len}/{vote_k}"
    if xyz is not None:
        t += f" | XYZ=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}) m"
    cv2.putText(out, t, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    if stale:
        cv2.putText(out, "STALE (no valid detection)", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
    return out
# --- ADD (헬퍼 함수) ---
def _save_debug(args, rgb_bgr=None, depth_m=None, prob=None, mask=None,
                pixel_center=None, centers_m=None, score=None, theta=None):
    if not args.save_dir:
        return
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    tag = f"{args.prefix}_{ts}"

    # 이미지/맵 저장
    if rgb_bgr is not None:
        cv2.imwrite(str(out_dir / f"{tag}_rgb.png"), rgb_bgr)
    if depth_m is not None:
        # 깊이 시각화: 0~1 m 구간 가정(원하면 zmax로 스케일)
        d8 = np.clip((depth_m / max(1e-6, 1.0)) * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{tag}_depth.png"), d8)
    if prob is not None:
        p8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{tag}_prob.png"), p8)
    if mask is not None:
        m8 = (mask > 0).astype(np.uint8) * 255
        cv2.imwrite(str(out_dir / f"{tag}_mask.png"), m8)

    # 오버레이 (선택)
    if (rgb_bgr is not None) and (pixel_center is not None):
        ov = rgb_bgr.copy()
        u, v = int(pixel_center[0]), int(pixel_center[1])
        cv2.drawMarker(ov, (u, v), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.imwrite(str(out_dir / f"{tag}_overlay.png"), ov)

    # 원시데이터 모음
    npz_dict = {}
    if depth_m is not None: npz_dict['depth_m'] = depth_m
    if prob    is not None: npz_dict['prob'] = prob
    if mask    is not None: npz_dict['mask'] = mask.astype(np.uint8)
    if centers_m is not None: npz_dict['centers_m'] = np.array(centers_m, dtype=np.float32)
    if pixel_center is not None: npz_dict['pixel_center'] = np.array(pixel_center, dtype=np.float32)
    if score   is not None: npz_dict['score'] = np.array([score], dtype=np.float32)
    if theta   is not None: npz_dict['theta'] = np.array([theta], dtype=np.float32)
    if npz_dict:
        np.savez_compressed(str(out_dir / f"{tag}_raw.npz"), **npz_dict)


# def _emit_result(args, success, centers_m, pixel_center=None, theta=None, score=None, extra=None):
#     """
#     centers_m: [[x,y,z], ...] (meters)
#     pixel_center: [u,v] (pixels)
#     theta: float or None (deg)
#     score: float or None
#     extra: dict or None (원하면 추가 필드 넣기)
#     """
#     result = {
#         "success": bool(success),
#         "centers": centers_m if centers_m is not None else [],
#         "pixel_center": pixel_center if pixel_center is not None else None,
#         "theta": theta,
#         "score": score,
#     }
#     if isinstance(extra, dict):
#         result.update(extra)
    
#     if args.json:
#         # 마지막 줄 JSON 1줄만 출력: node.py가 이걸 파싱
#         print(json.dumps(result, ensure_ascii=False))
#     else:
#         # 사람이 볼 로그(선택)
#         print("[basket] success:", result["success"],
#               "centers:", result["centers"],
#               "pixel_center:", result["pixel_center"],
#               "theta:", result["theta"], "score:", result["score"])

# --------------- Main -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='/home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/unet_best_multi.pt')
    ap.add_argument('--intr_json', default='intrinsics_color_640x480.json')
    ap.add_argument('--mode', choices=['basket','failure','basket_eh'], default='basket')
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--zmax_basket', type=float, default=0.6)
    ap.add_argument('--zmax_failure', type=float, default=0.5)
    ap.add_argument('--thr', type=float, default=0.45)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--bg_frames', type=int, default=20)
    ap.add_argument('--no_bg_auto', action='store_true')
    ap.add_argument('--diff_mode', choices=['bg','bilateral','min','blend'], default='blend')
    ap.add_argument('--align', choices=['color','depth'], default='color')
    ap.add_argument('--bg_path', default='/home/dyros/mcy_ws/Laundry_decision/bf/laundry_bg/bg_failure.npy')

    # voting & gating (첫 번째 스타일)
    ap.add_argument('--vote', action='store_true')
    ap.add_argument('--vote_k', type=int, default=10)
    ap.add_argument('--vote_min_area', type=int, default=150)
    ap.add_argument('--vote_min_dt', type=float, default=3.0)
    ap.add_argument('--vote_require_curr', action='store_true')
    ap.add_argument('--stale_sec', type=float, default=1.0)

    # JSON one-shot
    ap.add_argument('--json', action='store_true', help='창 없이 결과만 JSON으로 출력 후 종료')
    ap.add_argument('--show', action='store_true', help='JSON 모드에서도 시각화 창 띄우기') # Added --show
    ap.add_argument('--save_dir', default=None,
                        help='디버그 아웃 저장 폴더(이미지/npz 등)')
    ap.add_argument('--save_best_only', action='store_true',
                        help='베스트 프레임만 저장(기본: 처리한 프레임마다 저장)')
    ap.add_argument('--prefix', default='basket',
                        help='디버그 파일 접두어')
    ap.add_argument('--max_wait_sec', type=float, default=3.0, help='--json 모드 최대 대기')

    # debug_mcy color
    # ap.add_argument('--color_bandwidth', type=float, default=30.0,
    #                     help='색상 클러스터링 민감도 (값이 작을수록 잘게 쪼갬)')
    ###

    args = ap.parse_args()

    dev = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # load ckpt & build net
    ckpt = torch.load(args.ckpt, map_location='cpu')
    meta = ckpt
    c_in = int(meta.get('c_in', 6))
    base = int(meta.get('base', 32))
    use_se = bool(meta.get('use_se', True))
    net = TinyUNet(c_in=c_in, base=base, use_se=use_se).to(dev)
    net.load_state_dict(meta['model'])
    net.eval()

    fx, fy, ppx, ppy = load_intrinsics(args.intr_json)

    try:
        import pyrealsense2 as rs
    except Exception as e:
        msg = {"success": False, "error": f"pyrealsense2 not found: {e}"}
        print(json.dumps(msg) if args.json else str(msg))
        return

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    # [WARMUP] Discard frames for auto-exposure stabilization
    print("[INFO] Camera Warmup: 30 frames...")
    for _ in range(30):
        pipeline.wait_for_frames()
    align_target = rs.stream.color if args.align=='color' else rs.stream.depth
    align = rs.align(align_target)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # zmax per mode
    zmax = args.zmax_basket if args.mode=='basket' else args.zmax_failure

    # background 준비(옵션)
    bg_m = None
    if args.bg_path and os.path.exists(args.bg_path):
        try:
            bg_m = np.load(args.bg_path).astype(np.float32)
            print(f"[INFO] loaded background from {args.bg_path}")
        except Exception as e:
            print(f"[WARN] failed to load bg from {args.bg_path}: {e}")

    if (bg_m is None) and (not args.no_bg_auto):
        
        frames = []
        t_end = time.time() + min(args.max_wait_sec, 1.0) if args.json else None
        n_cap = args.bg_frames if not args.json else 8
        for _ in range(n_cap):
            fs = pipeline.wait_for_frames()
            fs = align.process(fs)
            d = np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32) * depth_scale
            frames.append(d)
            if args.json and (time.time() > t_end):
                break
        if frames:
            bg_m = np.median(np.stack(frames,0), axis=0).astype(np.float32)

    overlay_on = (not args.json) or args.show # Modifed overlay logic
    thr = float(args.thr)

    
    buf = deque(maxlen=max(1, args.vote_k))
    last_ok_ts = -1e9
    t_deadline = time.time() + args.max_wait_sec if args.json else None
    result_printed = False

    # debug_mcy_test // find all detections
    # while True:
    #     fs = pipeline.wait_for_frames()
    #     fs = align.process(fs)
    #     dfrm = fs.get_depth_frame()
    #     cfrm = fs.get_color_frame()
    #     if not dfrm or not cfrm:
    #         continue

    #     depth = np.asanyarray(dfrm.get_data()).astype(np.float32) * depth_scale
    #     color_bgr = np.asanyarray(cfrm.get_data())          # BGR
    #     color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)  # ★ RGB로 변환 (중요)

    #     H, W = depth.shape
    #     x_np = preprocess_inputs(
    #         depth, color_rgb, bg_m, args.size, zmax,
    #         mode_name=args.mode, diff_mode=args.diff_mode
    #     )
    #     x = torch.from_numpy(x_np).unsqueeze(0).to(dev)

    #     with torch.no_grad():
    #         z = net(x.float())

    #     mask = postprocess_mask(z, thr, (H, W))
        
    #     # 1. 'analyze_mask' 대신 'find_all_detections' 호출 (인수 사용)
    #     detections = find_all_detections(mask, args.vote_min_area, args.vote_min_dt)

    #     # --- [사용자 요청] 감지 결과 프린트 ---
    #     if len(detections) > 0:
    #         print(f"[{time.strftime('%H:%M:%S')}] Valid Objects Found: {len(detections)}")
    #         for i, det in enumerate(detections):
    #             print(f"    > Obj {i}: center={det['center']}, area={det['area']}, dt_max={det['dt_max']:.2f}")
    #     # --- [프린트 끝] ---

    #     # --- 현재 프레임 ---
    #     all_xyz_coords = []         # 이번 프레임에서 감지된 모든 3D 좌표
    #     best_det_this_frame = None  # 이번 프레임의 '최적' 객체 (voting/시각화용)
    #     accepted = False            # 이번 프레임이 유효했는지 (voting용)

    #     if len(detections) > 0:
    #         # 2. 감지된 모든 객체의 3D 좌표 계산
    #         for i, det in enumerate(detections):
    #             center = det["center"]
    #             cx, cy = center
    #             y0, y1 = max(0, cy-2), min(H, cy+3); x0, x1 = max(0, cx-2), min(W, cx+3)
    #             local = depth[y0:y1, x0:x1]
    #             val = local[local > 0]
                
    #             if val.size > 0:
    #                 z_m = float(np.median(val))
    #                 X, Y, Z = uvz_to_xyz(cx, cy, z_m, fx, fy, ppx, ppy)
    #                 xyz = (X, Y, Z)
    #                 all_xyz_coords.append(list(xyz))
                    
    #                 # 3. '최적' 객체 1개 선정 (기존 로직 호환용)
    #                 if i == 0: # 정렬되어 있으므로 0번이 '최적'
    #                     accepted = True
    #                     last_ok_ts = time.time()
    #                     best_det_this_frame = {
    #                         "center": center, "theta": det["theta"], "xyz": xyz,
    #                         "area": det["area"], "dt_max": det["dt_max"]
    #                     }
    #                     if args.save_dir and not args.save_best_only:
    #                         _save_debug(
    #                             args, rgb_bgr=color_bgr, depth_m=depth, prob=None, mask=mask,
    #                             pixel_center=center, centers_m=[list(xyz)],
    #                             score=float(det["dt_max"]),
    #                             theta=(float(det["theta"]) if det["theta"] is not None else None),
    #                         )
        
    #     final_center = None
    #     final_theta  = None
    #     final_xyz    = None
    #     is_stale = (time.time() - last_ok_ts) > args.stale_sec

    #     if args.vote:
    #         if args.vote_require_curr:
    #             # '최적' 객체 1개만 voting에 사용
    #             if accepted and best_det_this_frame:
    #                 det = best_det_this_frame
    #                 buf.append((det["center"][0], det["center"][1], det["theta"], det["xyz"][2]))
    #         else:
    #             if not is_stale and accepted and best_det_this_frame:
    #                 det = best_det_this_frame
    #                 buf.append((det["center"][0], det["center"][1], det["theta"], det["xyz"][2]))
            
    #         if (len(buf) >= args.vote_k) and (not is_stale):
    #             arr = np.array(buf, dtype=np.float32)  # [N,4]
    #             cx_med = float(np.median(arr[:,0])); cy_med = float(np.median(arr[:,1]))
    #             th_mean = float(circ_mean_theta(list(arr[:,2])))
    #             z_med  = float(np.median(arr[:,3]))
    #             final_center = (int(round(cx_med)), int(round(cy_med)))
    #             final_theta  = th_mean
    #             X, Y, Z = uvz_to_xyz(final_center[0], final_center[1], z_med, fx, fy, ppx, ppy)
    #             final_xyz = (X, Y, Z)
    #     else:
    #         # voting 사용 안 하면 '최적' 객체로 즉시 성공
    #         if accepted and best_det_this_frame:
    #             final_center = best_det_this_frame["center"]
    #             final_theta  = best_det_this_frame["theta"]
    #             final_xyz    = best_det_this_frame["xyz"]

    #     # --- JSON 모드 (사용자 명령어에는 해당 없음) ---
    #     if args.json:
    #         ok = (final_center is not None) and (final_xyz is not None) and (not is_stale)
    #         timeout = (time.time() > t_deadline)
            
    #         if ok and args.save_dir and args.save_best_only:
    #             _save_debug(
    #                 args, rgb_bgr=color_bgr, depth_m=depth, prob=None, mask=mask,
    #                 pixel_center=final_center, centers_m=[list(final_xyz)],
    #                 score=float(best_det_this_frame["dt_max"] if accepted else 0.0),
    #                 theta=(float(final_theta) if final_theta is not None else None),
    #             )

    #         if ok or timeout:
    #             area_now = best_det_this_frame["area"] if accepted else 0
    #             dt_now = best_det_this_frame["dt_max"] if accepted else 0.0
                
    #             out = {
    #                 "success": bool(ok), "empty": (not ok), "mode": args.mode,
    #                 "thr": thr, "zmax": zmax,
    #                 "centers": ([list(final_xyz)] if ok else []),
    #                 "pixel_center": (list(final_center) if ok else []),
    #                 "theta": (float(final_theta) if ok else 0.0),
    #                 "area": int(area_now),
    #                 "dt": float(dt_now),
    #                 # --- 갯수 확인을 위한 추가 필드 ---
    #                 "object_count": len(all_xyz_coords) if ok else 0,
    #                 "all_centers": all_xyz_coords if ok else []
    #             }
    #             print(json.dumps(out, ensure_ascii=False))
    #             result_printed = True
    #             break

    #     # --- 화면 모드 (사용자 명령어에 해당됨) ---
    #     if overlay_on:
    #         voted_flag = args.vote and (len(buf) > 0) and (not is_stale)
            
    #         # 시각화는 '최적' 객체(best) 또는 '최종' 객체(final)를 기준으로 함
    #         show_center = final_center if (final_center is not None) else (best_det_this_frame["center"] if accepted else None)
    #         show_theta  = final_theta if (final_theta is not None) else (best_det_this_frame["theta"] if accepted else 0.0)
    #         show_xyz    = final_xyz if (final_xyz is not None) else (best_det_this_frame["xyz"] if accepted else None)

    #         view = draw_overlay(color_bgr, mask, show_center, show_theta,
    #                             show_xyz, thr=thr, voted=voted_flag,
    #                             stale=is_stale, votes_len=len(buf), vote_k=args.vote_k)
    #         cv2.imshow("realtime_grasp_rgb2_gated", view)
    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord('q'):
    #             break
    #         # ... (이하 키 입력 부분은 동일) ...
    #         elif key == ord('o'):
    #             overlay_on = not overlay_on
    #         elif key == ord('b'):
    #             print("[INFO] capturing background...")
    #             frames = []
    #             for _ in range(15):
    #                 fs2 = pipeline.wait_for_frames()
    #                 fs2 = align.process(fs2)
    #                 d2 = np.asanyarray(fs2.get_depth_frame().get_data()).astype(np.float32) * depth_scale
    #                 frames.append(d2)
    #             bg_m = np.median(np.stack(frames,0), axis=0).astype(np.float32)
    #             try:
    #                 if args.bg_path:
    #                     os.makedirs(os.path.dirname(args.bg_path), exist_ok=True)
    #                     np.save(args.bg_path, bg_m)
    #                     print(f"[INFO] background saved to {args.bg_path}")
    #             except Exception as e:
    #                 print(f"[WARN] failed to save bg: {e}")
    #             print("[INFO] background updated.")
    #         elif key == ord(']'):
    #             thr = min(0.99, thr+0.02); print(f"[thr] {thr:.2f}")
    #         elif key == ord('['):
    #             thr = max(0.01, thr-0.02); print(f"[thr] {thr:.2f}")
        
    #     # --- if vis success, break ---
    #     # if (final_center is not None) and (final_xyz is not None) and (not is_stale) and (not args.json):
    #     #     print("[OK] detection:", {"center": final_center, "xyz": final_xyz, "theta": float(final_theta or 0.0)})
    #     #     break

    #     # JSON 모드에서 타임아웃 처리
    #     if args.json and (time.time() > t_deadline):
    #         break

# debug_mcy // analyze_mask
    while True:
        fs = pipeline.wait_for_frames()
        fs = align.process(fs)
        dfrm = fs.get_depth_frame()
        cfrm = fs.get_color_frame()
        if not dfrm or not cfrm:
            if args.json and (time.time() > t_deadline):
                break
            continue

        depth = np.asanyarray(dfrm.get_data()).astype(np.float32) * depth_scale
        color_bgr = np.asanyarray(cfrm.get_data())          # BGR
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)  # ★ RGB로 변환 (중요)

        H, W = depth.shape
        x_np = preprocess_inputs(
            depth, color_rgb, bg_m, args.size, zmax,
            mode_name=args.mode, diff_mode=args.diff_mode
        )
        x = torch.from_numpy(x_np).unsqueeze(0).to(dev)

        with torch.no_grad():
            z = net(x.float())

        mask = postprocess_mask(z, thr, (H, W))
        center, theta, area, dt_max = analyze_mask(mask)

        # --- 현재 프레임 ---
        accepted = False
        xyz = None
        if center is not None and area >= args.vote_min_area and dt_max >= args.vote_min_dt:
            cx, cy = center
            y0, y1 = max(0, cy-2), min(H, cy+3)
            x0, x1 = max(0, cx-2), min(W, cx+3)
            local = depth[y0:y1, x0:x1]
            val = local[local > 0]
            if val.size > 0:
                z_m = float(np.median(val))
                X, Y, Z = uvz_to_xyz(cx, cy, z_m, fx, fy, ppx, ppy)
                xyz = (X, Y, Z)
                accepted = True
                last_ok_ts = time.time()
            if args.save_dir and not args.save_best_only and accepted:
                _save_debug(
                    args,
                    rgb_bgr=color_bgr,
                    depth_m=depth,
                    prob=None,
                    mask=mask,
                    pixel_center=center,                  # (cx, cy)
                    centers_m=([list(xyz)] if xyz else None),  # [[X,Y,Z]]
                    score=float(dt_max),
                    theta=(float(theta) if theta is not None else None),
                )
        
        final_center = None
        final_theta  = None
        final_xyz    = None
        is_stale = (time.time() - last_ok_ts) > args.stale_sec

        if args.vote:
            if args.vote_require_curr:
                if accepted and xyz is not None:
                    buf.append((center[0], center[1], theta, xyz[2]))  # (cx,cy,theta,z)
            else:
                
                if not is_stale and xyz is not None:
                    buf.append((center[0], center[1], theta, xyz[2]))
            
            if (len(buf) >= args.vote_k) and (not is_stale):
                arr = np.array(buf, dtype=np.float32)  # [N,4]
                cx_med = float(np.median(arr[:,0])); cy_med = float(np.median(arr[:,1]))
                th_mean = float(circ_mean_theta(list(arr[:,2])))
                z_med  = float(np.median(arr[:,3]))
                final_center = (int(round(cx_med)), int(round(cy_med)))
                final_theta  = th_mean
                X, Y, Z = uvz_to_xyz(final_center[0], final_center[1], z_med, fx, fy, ppx, ppy)
                final_xyz = (X, Y, Z)
        else:
            # voting 사용 안 하면 accepted 프레임에서 즉시 성공
            if accepted and xyz is not None:
                final_center = center
                final_theta  = theta
                final_xyz    = xyz

        # --- JSON 모드: 성공/타임아웃 시 즉시 출력 ros2 통신에 사용---
            if args.json:
                ok = (final_center is not None) and (final_xyz is not None) and (not is_stale)
                timeout = (time.time() > t_deadline)

                
                if ok and args.save_dir and args.save_best_only:
                    _save_debug(
                        args,
                        rgb_bgr=color_bgr,
                        depth_m=depth,
                        prob=None,
                        mask=mask,
                        pixel_center=final_center,
                        centers_m=[list(final_xyz)],
                        score=float(dt_max),
                        theta=(float(final_theta) if final_theta is not None else None),
                    )

                if ok or timeout:
                    out = {
                        "success": bool(ok),
                        "empty": (not ok),
                        "mode": args.mode,
                        "thr": thr,
                        "zmax": zmax,
                        "centers": ([list(final_xyz)] if ok else []),
                        "pixel_center": (list(final_center) if ok else []),
                        "theta": (float(final_theta) if ok else 0.0),
                        "area": (int(area) if accepted else 0),
                        "dt": (float(dt_max) if accepted else 0.0),
                    }
                    print(out['centers'])
                    #debug
                    if out['centers']:
                        if out['centers'][0][2] <0.55:
                            out['centers'][0][2] +=0.05
                    print(out['centers'])
                    
                    #piper test 
                    #out['centers'] = [[0.004, -0.02, 0.368]]

                    
                    print(json.dumps(out, ensure_ascii=False))
                    result_printed = True
                    save_path = "/home/dyros/mcy_ws/Laundry_decision/run_bf/eh_basket_dbg/run/debugging.png"
                    if args.show:
                        with open("/home/dyros/mcy_ws/Laundry_decision/vis_debug.log", "a") as f:
                            f.write(f"[{time.time()}] Attempting to show success window\n")
                        view = draw_overlay(color_bgr, mask, final_center, final_theta, final_xyz, thr=thr)
                        cv2.imshow("realtime_grasp_rgb2_gated", view)
                        cv2.imwrite(save_path, view)
                        print("[INFO] Showing result for 3 seconds...")
                        cv2.waitKey(3000)
                    break

        # --- 화면 모드: 현재 프레임 오버레이 표시 ---
        if overlay_on:
            voted_flag = args.vote and (len(buf) > 0) and (not is_stale)
            show_center = final_center if (final_center is not None) else center
            show_theta  = final_theta if (final_theta is not None) else (theta if theta is not None else 0.0)
            show_xyz    = final_xyz if (final_xyz is not None) else xyz
            view = draw_overlay(color_bgr, mask, show_center, show_theta,
                                show_xyz, thr=thr, voted=voted_flag,
                                stale=is_stale, votes_len=len(buf), vote_k=args.vote_k)
            cv2.imshow("realtime_grasp_rgb2_gated", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('o'):
                overlay_on = not overlay_on
            elif key == ord('b'):
                print("[INFO] capturing background...")
                frames = []
                for _ in range(15):
                    fs2 = pipeline.wait_for_frames()
                    fs2 = align.process(fs2)
                    d2 = np.asanyarray(fs2.get_depth_frame().get_data()).astype(np.float32) * depth_scale
                    frames.append(d2)
                bg_m = np.median(np.stack(frames,0), axis=0).astype(np.float32)
                try:
                    if args.bg_path:
                        os.makedirs(os.path.dirname(args.bg_path), exist_ok=True)
                        np.save(args.bg_path, bg_m)
                        print(f"[INFO] background saved to {args.bg_path}")
                except Exception as e:
                    print(f"[WARN] failed to save bg: {e}")
                print("[INFO] background updated.")
            elif key == ord(']'):
                thr = min(0.99, thr+0.02); print(f"[thr] {thr:.2f}")
            elif key == ord('['):
                thr = max(0.01, thr-0.02); print(f"[thr] {thr:.2f}")
        
        # --- if vis success, break ---
        
        if (final_center is not None) and (final_xyz is not None) and (not is_stale) and (not args.json):
            print("[OK] detection:", {"center": final_center, "xyz": final_xyz, "theta": float(final_theta or 0.0)})
            break

        # JSON 모드에서 타임아웃 처리
        if args.json and (time.time() > t_deadline):
            break

    pipeline.stop()
    if not args.json:
        cv2.destroyAllWindows()
    if args.json and not result_printed:
        print(json.dumps({"success": False, "empty": True, "mode": args.mode}))
        return
    
    #debug_mcy color
    # while True:
    #     fs = pipeline.wait_for_frames()
    #     fs = align.process(fs)
    #     dfrm = fs.get_depth_frame()
    #     cfrm = fs.get_color_frame()
    #     if not dfrm or not cfrm:
    #         if args.json and (time.time() > t_deadline):
    #             break
    #         continue

    #     depth = np.asanyarray(dfrm.get_data()).astype(np.float32) * depth_scale
    #     color_bgr = np.asanyarray(cfrm.get_data())
    #     color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    #     H, W = depth.shape
    #     x_np = preprocess_inputs(
    #         depth, color_rgb, bg_m, args.size, zmax,
    #         mode_name=args.mode, diff_mode=args.diff_mode
    #     )
    #     x = torch.from_numpy(x_np).unsqueeze(0).to(dev)

    #     with torch.no_grad():
    #         z = net(x.float())

    #     # 1. 딥러닝 마스크 생성
    #     mask = postprocess_mask(z, thr, (H, W))

    #     # 2. 마스크 내부를 색상으로 분리하여 3D 좌표 리스트 받기
    #     all_xyz_coords = find_clusters_by_color(
    #         color_rgb, depth, mask, 
    #         fx, fy, ppx, ppy,
    #         min_area_per_cluster=args.vote_min_area,
    #         color_bandwidth=args.color_bandwidth # 인자값 사용
    #     )
        
    #     object_count = len(all_xyz_coords)
        
    #     if object_count > 0:
    #         accepted = True
    #         last_ok_ts = time.time()
    #         if args.save_dir and not args.save_best_only:
    #              _save_debug(
    #                 args, rgb_bgr=color_bgr, depth_m=depth, mask=mask,
    #                 centers_m=all_xyz_coords, # 모든 좌표 저장
    #             )
    #     else:
    #         accepted = False

        
    #     final_xyz_list = []
    #     is_stale = (time.time() - last_ok_ts) > args.stale_sec

    #     if args.vote:
    #         if args.vote_require_curr:
    #             if accepted:
    #                 buf.append(1) # 유효 프레임 카운트
    #         else:
    #             if not is_stale:
    #                 buf.append(1)
            
    #         if (len(buf) >= args.vote_k) and (not is_stale):
    #             final_xyz_list = all_xyz_coords # VOTE 성공 시 현재 프레임의 모든 좌표 사용
    #     else:
    #         # voting 사용 안 하면 즉시 성공
    #         if accepted:
    #             final_xyz_list = all_xyz_coords

    #     # --- JSON 모드: 성공/타임아웃 시 즉시 출력 ---
    #     if args.json:
    #         ok = (len(final_xyz_list) > 0)
    #         timeout = (time.time() > t_deadline)
            
    #         if ok and args.save_dir and args.save_best_only:
    #             _save_debug(
    #                 args, rgb_bgr=color_bgr, depth_m=depth, mask=mask,
    #                 centers_m=final_xyz_list
    #             )

    #         if ok or timeout:
    #             out = {
    #                 "success": bool(ok),
    #                 "empty": (not ok),
    #                 "mode": args.mode,
    #                 "thr": thr,
    #                 "zmax": zmax,
    #                 "centers": [list(xyz) for xyz in final_xyz_list] if ok else [], # 모든 좌표
    #                 "pixel_center": [], # (새 로직에서는 의미 없음)
    #                 "theta": 0.0,       # (새 로직에서는 의미 없음)
    #                 "area": 0,          # (새 로직에서는 의미 없음)
    #                 "dt": 0.0,          # (새 로직에서는 의미 없음)
    #                 "object_count": len(final_xyz_list) if ok else 0
    #             }
                
    #             # (debug_mcy의 z 보정 로직 - 필요시 주석 해제)
    #             # if ok:
    #             #     for center in out['centers']:
    #             #         if center[2] < 0.55:
    #             #             center[2] += 0.05
                
    #             print(json.dumps(out, ensure_ascii=False))
    #             result_printed = True
    #             break

    #     # --- 화면 모드: 현재 프레임 오버레이 표시 ---
    #     if overlay_on:
    #         voted_flag = args.vote and (len(buf) > 0) and (not is_stale)
            
    #         # 시각화는 마스크 + (첫 번째 객체의 XYZ)만 표시
    #         show_xyz = final_xyz_list[0] if (len(final_xyz_list) > 0) else (all_xyz_coords[0] if accepted else None)

    #         view = draw_overlay(color_bgr, mask, 
    #                             None, 0.0, # (2D 중심/각도 없음)
    #                             show_xyz, thr=thr, voted=voted_flag,
    #                             stale=is_stale, votes_len=len(buf), vote_k=args.vote_k)
            
    #         # (갯수 표시 추가)
    #         cv2.putText(view, f"Objects Found: {object_count}", (10, 55),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
    #         cv2.imshow("realtime_grasp_rgb2_gated", view)
    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord('q'):
    #             break
    #         elif key == ord('o'):
    #             overlay_on = not overlay_on
    #         elif key == ord('b'):
    #             print("[INFO] capturing background...")
    #             frames = []
    #             for _ in range(15):
    #                 fs2 = pipeline.wait_for_frames()
    #                 fs2 = align.process(fs2)
    #                 d2 = np.asanyarray(fs2.get_depth_frame().get_data()).astype(np.float32) * depth_scale
    #                 frames.append(d2)
    #             bg_m = np.median(np.stack(frames,0), axis=0).astype(np.float32)
    #             try:
    #                 if args.bg_path:
    #                     os.makedirs(os.path.dirname(args.bg_path), exist_ok=True)
    #                     np.save(args.bg_path, bg_m)
    #                     print(f"[INFO] background saved to {args.bg_path}")
    #             except Exception as e:
    #                 print(f"[WARN] failed to save bg: {e}")
    #             print("[INFO] background updated.")
    #         elif key == ord(']'):
    #             thr = min(0.99, thr+0.02); print(f"[thr] {thr:.2f}")
    #         elif key == ord('['):
    #             thr = max(0.01, thr-0.02); print(f"[thr] {thr:.2f}")
        
    #     # --- if vis success, break ---
    #     if (len(final_xyz_list) > 0) and (not args.json):
    #         print(f"[OK] detection ({len(final_xyz_list)} objects):", final_xyz_list)
    #         break

    #     # JSON 모드에서 타임아웃 처리
    #     if args.json and (time.time() > t_deadline):
    #         break

    # pipeline.stop()
    # if not args.json:
    #     cv2.destroyAllWindows()
    # if args.json and not result_printed:
    #     # (JSON 출력 수정)
    #     print(json.dumps({"success": False, "empty": True, "mode": args.mode, "object_count": 0}))
    #     return

    
    
if __name__ == '__main__':
    main()