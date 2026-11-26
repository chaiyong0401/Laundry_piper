"""
node2_backup_in_process.py

subprocess 방식 대신 모델·카메라·배경을 프로세스 상주시키는 in-process 버전.

제거된 오버헤드 (매 호출마다 발생하던 것):
  - Python 인터프리터 시작          ~1s
  - torch / cv2 / pyrealsense2 import  ~1s
  - 모델 가중치 디스크 로드           ~1-2s
  - RealSense 카메라 초기화 + warmup   ~1-2s
  합계: 4-6s 고정 비용 → 0s (startup 1회만)

모드 대응:
  basket         → detect_basket()  : TinyUNet(c_in=6) + 단일 프레임
  basket_eh      → detect_basket()  : 동일
  failure_eh     → detect_failure_eh(strict=False) : RGB gate + TinyUNet + vote_k=1 (단발)
  failure_eh_strict → detect_failure_eh(strict=True)  : vote_k=3, vote_min_area/dt 강화

사용:
  python node2_backup_in_process.py [--device cpu|cuda] [--gpu] [--show]
"""

import os, sys, json, math, collections, threading
import time as _time
import argparse
from typing import Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from geometry_msgs.msg import Pose
from piper_with_gripper_moveit.action import TargetPose

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
CKPT_PATH_B   = "/home/user/samsung_laundry/Laundry_piper/Laundry_decision/run_bf/dataset_rgb_sam2/unet_best_basket_new.pt"
CKPT_PATH_F   = "/home/user/samsung_laundry/Laundry_piper/Laundry_decision/run_bf/dataset_rgb_sam2/unet_best_failure_new.pt"
CKPT_PATH_RGB = "/home/user/samsung_laundry/Laundry_piper/Laundry_decision/run_bf/dataset_rgb_sam2/rgb_only.pth"
INTR_COLOR    = "/home/user/samsung_laundry/Laundry_piper/Laundry_decision/run_bf/intrinsics_color_640x480.json"
INTR_DEPTH    = "/home/user/samsung_laundry/Laundry_piper/Laundry_decision/run_bf/intrinsics_depth_640x480.json"
BG_BASKET     = "/home/user/samsung_laundry/Laundry_piper/Laundry_decision/run_bf/laundry_bg/bg_basket.npy"
BG_FAILURE    = "/home/user/samsung_laundry/Laundry_piper/Laundry_decision/run_bf/laundry_bg/bg_overcome2.npy"


# ── wm 모드 상수 (run_laundry.py CFG["wm"] 와 동일) ─────────────────────────
WM_RGB_CKPT      = "/home/user/samsung_laundry/Laundry_piper/Laundry_decision/run_laundry/models/best.pt"
WM_DETECTOR_DIR  = "/home/user/samsung_laundry/Laundry_piper/Laundry_decision/run_laundry"
WM_Z_MIN         = 0.35
WM_Z_MAX         = 0.65
WM_CYL_RADIUS    = 0.29
WM_FILTER_RADIUS = 0.285
WM_EPS           = 0.037
WM_MIN_SAMPLES   = 45
WM_RGB_HI_THR    = 0.65
WM_RGB_LO_THR    = 0.60


# ── wm in-process 헬퍼 (run_laundry.py에서 이식) ─────────────────────────────
def _wm_estimate_center(pcd, o3d):
    pts = np.asarray(pcd.points)
    mask = (pts[:, 2] > 0.4) & (pts[:, 2] < 1.0)
    pcd_z = o3d.geometry.PointCloud()
    pcd_z.points = o3d.utility.Vector3dVector(pts[mask])
    pcd_z, _ = pcd_z.remove_radius_outlier(nb_points=10, radius=0.02)
    pcd_z, _ = pcd_z.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    return np.mean(np.asarray(pcd_z.points), axis=0)

def _wm_create_cylinder(o3d, radius, z_min, z_max, radial_res=200, height_res=150):
    zs = np.linspace(z_min, z_max, height_res)
    ths = np.linspace(0, 2 * np.pi, radial_res)
    pts = [[radius * np.cos(t), radius * np.sin(t), z] for z in zs for t in ths]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    return pc

def _wm_icp(curr_pcd, ref_pcd, o3d, init_T=None):
    if init_T is None:
        init_T = np.eye(4)
    ref_pts  = np.asarray(ref_pcd.points)
    curr_pts = np.asarray(curr_pcd.points)
    ref_h  = o3d.geometry.PointCloud()
    curr_h = o3d.geometry.PointCloud()
    ref_h.points  = o3d.utility.Vector3dVector(ref_pts[ref_pts[:, 1] < 0])
    curr_h.points = o3d.utility.Vector3dVector(curr_pts[curr_pts[:, 1] < 0])
    result = o3d.pipelines.registration.registration_icp(
        curr_h.voxel_down_sample(0.005), ref_h.voxel_down_sample(0.005),
        0.05, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )
    curr_pcd.transform(result.transformation)
    return curr_pcd, result.transformation

def _wm_rotate_cluster(cluster, angle_deg):
    a = np.radians(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    out = cluster.copy()
    out[:, :2] = cluster[:, :2] @ R.T
    return out

def _wm_filter_radius(pts, cx, cy, radius):
    return np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2) <= radius


# ── 시각화 헬퍼 (run_failure_eh.py draw_overlay와 동일) ───────────────────────
def _draw_overlay(color_bgr, mask_u8, center, theta, xyz=None, alpha=0.35, info_text=None):
    out = color_bgr.copy()
    if mask_u8 is not None:
        m = mask_u8.astype(np.uint8)
        if m.max() == 1:
            m = (m * 255).astype(np.uint8)
        overlay = np.dstack([np.zeros_like(m), m, np.zeros_like(m)])  # green channel
        overlay = cv2.GaussianBlur(overlay, (5, 5), 0)
        out = cv2.addWeighted(out, 1.0, overlay, alpha, 0)
    if center is not None:
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(out, (cx, cy), 6, (0, 0, 255), -1)
        L = 40
        th = theta if theta is not None else 0.0
        cv2.line(out, (int(cx - L * np.cos(th)), int(cy - L * np.sin(th))),
                      (int(cx + L * np.cos(th)), int(cy + L * np.sin(th))), (255, 0, 0), 2)
    parts = []
    if xyz is not None:
        parts.append(f"XYZ=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})m")
    if info_text:
        parts.append(info_text)
    if parts:
        cv2.putText(out, " | ".join(parts), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out

# ── TinyUNet (basket용: c_in=6) ───────────────────────────────────────────────
class SE(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(ch, max(1, ch//r), bias=False), nn.ReLU(True),
            nn.Linear(max(1, ch//r), ch, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.shape
        return x * self.fc(self.avg(x).view(b, c)).view(b, c, 1, 1)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.se = SE(out_ch) if use_se else nn.Identity()
    def forward(self, x): return self.se(self.block(x))

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=True):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, use_se)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=True):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, use_se)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dh = x2.size(2)-x1.size(2); dw = x2.size(3)-x1.size(3)
        x1 = F.pad(x1, [dw//2, dw-dw//2, dh//2, dh-dh//2])
        return self.conv(torch.cat([x2, x1], 1))

class TinyUNet(nn.Module):
    def __init__(self, c_in=6, base=32, use_se=True):
        super().__init__()
        self.inc  = DoubleConv(c_in, base, use_se)
        self.d1   = Down(base, base*2, use_se); self.d2 = Down(base*2, base*4, use_se)
        self.d3   = Down(base*4, base*8, use_se)
        self.u1   = Up(base*8, base*4, use_se); self.u2 = Up(base*4, base*2, use_se)
        self.u3   = Up(base*2, base, use_se)
        self.outc = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3)
        x=self.u1(x4,x3); x=self.u2(x,x2); x=self.u3(x,x1)
        return self.outc(x)

# ── RGB Gate (ResNet18 기반, failure_eh용) ────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _rgb_eval_transform():
    return T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                      T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

def _create_backbone(model_name="resnet18", num_classes=2):
    if model_name == "resnet18":
        from torchvision.models import resnet18
        m = resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == "mobilenet_v3_small":
        from torchvision.models import mobilenet_v3_small
        m = mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"unknown model: {model_name}")
    return m

class RGBGate:
    def __init__(self, ckpt_path: str, device: torch.device, ema_decay: float = 0.7):
        self.device    = device
        self.tfm       = _rgb_eval_transform()
        self.ema_decay = ema_decay
        self.ema_prob: Optional[float] = None
        self.ts = self.model = None
        self.cloth_idx = 1
        self.idx_to_class = {0: "empty", 1: "cloth"}

        # TorchScript 우선 시도
        try:
            self.ts = torch.jit.load(ckpt_path, map_location=device); self.ts.eval()
            meta_path = ckpt_path + ".json"
            if os.path.exists(meta_path):
                meta = json.load(open(meta_path))
                self.idx_to_class = {int(k):v for k,v in meta.get("idx_to_class",{}).items()}
                cloth_list = [k for k,v in self.idx_to_class.items() if v=="cloth"]
                self.cloth_idx = cloth_list[0] if cloth_list else 1
            print(f"[RGBGate] TorchScript loaded: {ckpt_path}")
            return
        except Exception:
            pass

        # state_dict fallback
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        args_d = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
        self.model = _create_backbone(args_d.get("model", "resnet18"), 2).to(device).eval()
        new_sd = {k[6:] if k.startswith("model.") else k: v for k,v in state_dict.items()}
        self.model.load_state_dict(new_sd, strict=False)
        cls2idx = ckpt.get("class_to_idx", {"empty":0,"cloth":1}) if isinstance(ckpt,dict) else {"empty":0,"cloth":1}
        self.idx_to_class = {v:k for k,v in cls2idx.items()}
        cloth_list = [k for k,v in self.idx_to_class.items() if v=="cloth"]
        self.cloth_idx = cloth_list[0] if cloth_list else 1
        print(f"[RGBGate] state_dict loaded: {ckpt_path}")

    def reset(self):
        """호출 간 EMA 초기화 (subprocess 방식과 동일한 동작)"""
        self.ema_prob = None

    @torch.inference_mode()
    def decide(self, bgr_img, hi_thr=0.60, lo_thr=0.55) -> dict:
        rgb    = bgr_img[..., ::-1]
        x      = self.tfm(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        logits = self.ts(x) if self.ts is not None else self.model(x)
        prob   = torch.softmax(logits, 1)[0]
        pred_idx  = int(prob.argmax())
        cloth_p   = float(prob[self.cloth_idx]) if 0 <= self.cloth_idx < prob.numel() else float(prob.max())
        self.ema_prob = (cloth_p if self.ema_prob is None
                         else self.ema_decay*self.ema_prob + (1-self.ema_decay)*cloth_p)
        # hi_thr 미만은 전부 EMPTY (lo_thr~hi_thr 애매 구간도 안전하게 EMPTY 처리)
        decision = "CLOTH" if self.ema_prob >= hi_thr else "EMPTY"
        return {"pred_label": self.idx_to_class.get(pred_idx, str(pred_idx)),
                "cloth_prob": cloth_p, "ema_prob": float(self.ema_prob), "decision": decision}

# ── 공통 이미지 처리 ──────────────────────────────────────────────────────────
def _load_intrinsics(path):
    J = json.load(open(path))
    fx  = float(J.get('fx') or J.get('intrinsics',{}).get('fx'))
    fy  = float(J.get('fy') or J.get('intrinsics',{}).get('fy'))
    ppx = float(J.get('ppx') or J.get('cx') or J.get('intrinsics',{}).get('ppx'))
    ppy = float(J.get('ppy') or J.get('cy') or J.get('intrinsics',{}).get('ppy'))
    return fx, fy, ppx, ppy

def _uvz_to_xyz(u, v, z, fx, fy, ppx, ppy):
    return (u-ppx)*z/fx, (v-ppy)*z/fy, z

def _compute_diff(depth_m, bg_m, mode='blend'):
    d = depth_m.astype(np.float32); valid = d > 0
    base    = cv2.bilateralFilter(d, 5, 0.05, 7)
    diff_bi = np.abs(base - d); diff_bi[~valid] = 0.0
    diff_bg = np.zeros_like(d)
    if bg_m is not None:
        bg = bg_m.astype(np.float32); both = valid & (bg > 0)
        diff_bg[both] = np.abs(bg[both] - d[both])
    if mode == 'blend':
        diff = diff_bi.copy()
        if bg_m is not None:
            m = diff_bg > 0; diff[m] = diff_bg[m]
    elif mode == 'bg':        diff = diff_bg
    elif mode == 'bilateral': diff = diff_bi
    else:  # min
        diff = diff_bi.copy()
        if bg_m is not None:
            m = diff_bg > 0; diff[m] = np.minimum(diff_bg[m], diff_bi[m])
    return (np.clip(diff, 0.0, 0.1) / 0.1).astype(np.float32)

def _preprocess_basket(depth_m, color_rgb, bg_m, size=384, zmax=0.35):
    """6ch tensor [diff, z, R, G, B, domain=0] — basket_backup.py와 동일"""
    diff   = _compute_diff(depth_m, bg_m, 'blend')
    z_     = np.clip(depth_m, 0, zmax) / max(1e-6, zmax)
    diff_r = cv2.resize(diff, (size, size), interpolation=cv2.INTER_NEAREST)
    z_r    = cv2.resize(z_,   (size, size), interpolation=cv2.INTER_NEAREST)
    rgb_r  = cv2.resize(color_rgb.astype(np.float32)/255., (size, size), cv2.INTER_LINEAR)
    dom    = np.zeros((1, size, size), np.float32)  # domain=0 (basket)
    x = np.concatenate([np.stack([diff_r, z_r], 0),
                         np.transpose(rgb_r, (2,0,1)), dom], axis=0)
    return x.astype(np.float32)

def _preprocess_failure(depth_m, color_rgb, bg_m, size=384, zmax=0.5):
    """6ch tensor [diff, z, R, G, B, domain=1] — failure domain (run_basket_backup.py와 동일, domain만 다름)"""
    diff   = _compute_diff(depth_m, bg_m, 'blend')
    z_     = np.clip(depth_m, 0, zmax) / max(1e-6, zmax)
    diff_r = cv2.resize(diff, (size, size), interpolation=cv2.INTER_NEAREST)
    z_r    = cv2.resize(z_,   (size, size), interpolation=cv2.INTER_NEAREST)
    rgb_r  = cv2.resize(color_rgb.astype(np.float32)/255., (size, size), cv2.INTER_LINEAR)
    dom    = np.ones((1, size, size), np.float32)   # domain=1 (failure)
    x = np.concatenate([np.stack([diff_r, z_r], 0),
                         np.transpose(rgb_r, (2,0,1)), dom], axis=0)
    return x.astype(np.float32)

def _postprocess_mask_failure(logits, thr, out_hw):
    """failure TinyUNet 후처리 — basket의 65px ellipse 없이 간단한 open만 적용"""
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    m = (prob >= thr).astype(np.uint8)
    H, W = out_hw
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return m

def _postprocess_mask_basket(logits, thr, out_hw):
    """run_basket_backup.py postprocess_mask와 동일 (65px ellipse open 포함)"""
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    m = (prob >= thr).astype(np.uint8)
    H, W = out_hw
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    k65 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k65)
    return m

def _largest_cc(mask) -> Tuple[Optional[np.ndarray], int]:
    num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    if num <= 1: return None, 0
    best_lbl, best_area = None, 0
    for lb in range(1, num):
        a = int((labels == lb).sum())
        if a > best_area: best_area, best_lbl = a, lb
    return (labels == best_lbl).astype(np.uint8), best_area

def _analyze_mask(mask):
    """(center, theta, area, dt_max) — analyze_mask + find_center 통합"""
    cc, area = _largest_cc(mask)
    if cc is None: return None, 0.0, 0, 0.0
    dt = cv2.distanceTransform(cc, cv2.DIST_L2, 3)
    dt_max = float(dt.max())
    ys, xs = np.where(cc > 0)
    if len(xs) >= 5:
        x_ = xs.astype(np.float32) - xs.mean()
        y_ = ys.astype(np.float32) - ys.mean()
        theta = 0.5 * math.atan2(2*(x_*y_).mean(), ((x_*x_).mean()-(y_*y_).mean()))
    else:
        theta = 0.0
    # find_center: centroid (basket_backup.py 동일)
    cx, cy = int(xs.mean()), int(ys.mean())
    return (cx, cy), theta, int(area), dt_max

def _analyze_mask_multi(mask, min_area=150, min_dt=3.0, max_centers=5):
    """마스크에서 조건을 만족하는 모든 CC의 centroid 반환 (면적 내림차순).
    반환: [(area, dt_max, cx, cy), ...]
    """
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    results = []
    for lb in range(1, num):
        area = int(stats[lb, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cc = (labels == lb).astype(np.uint8)
        dt_max = float(cv2.distanceTransform(cc, cv2.DIST_L2, 3).max())
        if dt_max < min_dt:
            continue
        cx = int(round(centroids[lb, 0]))
        cy = int(round(centroids[lb, 1]))
        results.append((area, dt_max, cx, cy))
    results.sort(key=lambda r: r[0], reverse=True)
    return results[:max_centers]

# ── failure_eh 전용: depth heuristic ─────────────────────────────────────────
def _fill_holes(mask_255: np.ndarray) -> np.ndarray:
    inv = 255 - (mask_255 > 0).astype(np.uint8) * 255
    ff  = inv.copy()
    cv2.floodFill(ff, np.zeros((mask_255.shape[0]+2, mask_255.shape[1]+2), np.uint8), (0,0), 255)
    return cv2.bitwise_or((mask_255 > 0).astype(np.uint8)*255, 255-ff)

def _depth_to_sky_mask(depth_m, zmax=0.5, blue_max=35, lift=80, u8_thr=102,
                        close1=5, blur_k=7, thr_blur=127, close2=7) -> np.ndarray:
    """run_failure_eh.py _save_color_depth_depthonly_blue_to_sky와 동일 로직 (파일 저장 제거).

    원본과의 핵심 차이:
    - depth=0 픽셀도 u8=0 → blue lift → u8=80 < thr=102 이므로 마스크에 포함됨 (& valid 없음)
    - fill_holes 는 close2 이후 한 번만 (중간 fill_holes 없음)
    """
    # depth → u8 (depth=0 포함, clip만)
    u8 = (np.clip(depth_m, 0.0, zmax) / max(1e-6, zmax) * 255).astype(np.uint8)
    H, W = u8.shape; left = W // 8
    xs = np.tile(np.arange(W), (H, 1))
    # blue → sky lift (valid 체크 없이 원본과 동일)
    m_blue = (u8 <= blue_max) & (xs >= left)
    u8[m_blue] = lift
    # binary mask (valid 체크 없이 원본과 동일)
    mask = ((u8 < u8_thr) & (xs >= left)).astype(np.uint8) * 255
    # close1 → blur → threshold → close2 → fill_holes (원본 순서)
    k1 = max(1, close1 | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1)))
    blurred = cv2.GaussianBlur(mask, (blur_k | 1, blur_k | 1), 0)
    _, mask = cv2.threshold(blurred, thr_blur, 255, cv2.THRESH_BINARY)
    k2 = max(1, close2 | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2)))
    mask = _fill_holes(mask)
    mask[:, :left] = 0
    return mask

def _grasp_from_mask(mask_255):
    """run_failure_eh.py grasp_from_mask와 동일"""
    cc, area = _largest_cc((mask_255 > 0).astype(np.uint8))
    if cc is None: return None, 0.0, 0.0, 0
    ys, xs = np.where(cc > 0)
    cx = float(xs.mean()); cy = float(ys.mean())
    cx = min(cx + float(xs.max()-xs.min()+1)/8.0, cc.shape[1]-1)
    dt_peak = float(cv2.distanceTransform(cc, cv2.DIST_L2, 3).max())
    if len(xs) >= 5:
        x_ = xs.astype(np.float32)-xs.mean(); y_ = ys.astype(np.float32)-ys.mean()
        theta = 0.5*math.atan2(2*(x_*y_).mean(), ((x_*x_).mean()-(y_*y_).mean()))
    else:
        theta = 0.0
    return (int(round(cx)), int(round(cy))), theta, dt_peak, area

# ── VisionEngine ──────────────────────────────────────────────────────────────
class VisionEngine:
    """
    startup 시 1회 초기화.
    이후 detect_basket() / detect_failure_eh()를 반복 호출.
    """

    def __init__(self, device: str = 'cpu', log_fn=print, show: bool = False):
        self._log  = log_fn
        self._dev  = torch.device(device)
        self._lock = threading.Lock()   # 동시 호출 방지
        self._cur_exposure: Optional[str] = None
        self._show = show

        # ── 모델 로드 ─────────────────────────────────────────────────────────
        self._log("[VisionEngine] 모델 로드 중...")
        t0 = _time.time()
        self._basket_net  = self._load_unet(CKPT_PATH_B, c_in_default=6)
        self._failure_net = self._load_unet(CKPT_PATH_F, c_in_default=6)
        self._rgb_gate    = RGBGate(CKPT_PATH_RGB, self._dev)
        self._log(f"[VisionEngine] 모델 로드 완료: {_time.time()-t0:.2f}s")

        # ── 배경 이미지 ───────────────────────────────────────────────────────
        self._bg_basket  = self._load_npy(BG_BASKET,  "bg_basket")
        self._bg_failure = self._load_npy(BG_FAILURE, "bg_failure")

        # ── intrinsics ────────────────────────────────────────────────────────
        self._intr_color = _load_intrinsics(INTR_COLOR)
        self._intr_depth = _load_intrinsics(INTR_DEPTH)

        # failure_eh 고정 exposure 파라미터
        params_path = BG_FAILURE.replace('.npy', '_params.json')
        self._bg_failure_params: dict = {}
        if os.path.exists(params_path):
            try:
                self._bg_failure_params = json.load(open(params_path))
            except Exception:
                pass

        # ── RealSense 카메라 시작 ─────────────────────────────────────────────
        self._log("[VisionEngine] RealSense 초기화 중...")
        t0 = _time.time()
        self._pipeline, self._align, self._depth_scale, self._rs_profile = self._start_camera()
        self._log(f"[VisionEngine] 카메라 초기화 완료: {_time.time()-t0:.2f}s")

        # 초기 exposure: basket 모드(auto), warmup 30 frames
        self._set_exposure('auto', flush_n=30)

        # ── wm 모드 (open3d + detector) 초기화 ───────────────────────────────
        self._wm_gate  = None
        self._wm_o3d   = None
        self._wm_funcs = None
        if os.path.isfile(WM_RGB_CKPT):
            try:
                self._wm_gate = RGBGate(WM_RGB_CKPT, self._dev)
                import open3d as _o3d
                self._wm_o3d = _o3d
                if WM_DETECTOR_DIR not in sys.path:
                    sys.path.insert(0, WM_DETECTOR_DIR)
                from detector.utils import (get_current_pointcloud, to_open3d_pcd,
                                            split_clusters_by_color_lab)
                from detector.zscore_detector import (detect_anomalies_adaptive,
                                                      cluster_and_get_targets,
                                                      split_colorful_clusters)
                self._wm_funcs = {
                    'get_pc':     get_current_pointcloud,
                    'to_pcd':     to_open3d_pcd,
                    'split_lab':  split_clusters_by_color_lab,
                    'anomalies':  detect_anomalies_adaptive,
                    'cluster':    cluster_and_get_targets,
                    'split_color': split_colorful_clusters,
                }
                self._log("[VisionEngine] wm in-process 모드 초기화 완료")
            except Exception as e:
                self._log(f"[VisionEngine][WARN] wm 초기화 실패: {e}")

        self._log("[VisionEngine] 준비 완료")

    # ── 초기화 헬퍼 ───────────────────────────────────────────────────────────
    def _load_unet(self, path, c_in_default=6):
        ckpt = torch.load(path, map_location='cpu')
        c_in   = int(ckpt.get('c_in', c_in_default))
        base   = int(ckpt.get('base', 32))
        use_se = bool(ckpt.get('use_se', True))
        net = TinyUNet(c_in=c_in, base=base, use_se=use_se).to(self._dev)
        net.load_state_dict(ckpt['model']); net.eval()
        return net

    def _load_npy(self, path, label) -> Optional[np.ndarray]:
        if os.path.exists(path):
            try:
                arr = np.load(path).astype(np.float32)
                self._log(f"[VisionEngine] {label} 로드 완료: {path}")
                return arr
            except Exception as e:
                self._log(f"[VisionEngine][WARN] {label} 로드 실패: {e}")
        return None

    def _start_camera(self):
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipeline.start(cfg)
        align = rs.align(rs.stream.color)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        return pipeline, align, depth_scale, profile

    def _set_exposure(self, mode: str, flush_n: int = 15):
        """mode 변경 시만 실행. flush_n 프레임으로 노출 안정화.

        이 카메라는 query_sensors()가 'Stereo Module' 하나만 반환하고
        is_color_sensor()=False 이므로, supports(option) 기반으로 판단.
        """
        if mode == self._cur_exposure:
            return
        import pyrealsense2 as rs
        device = self._rs_profile.get_device()
        for sensor in device.query_sensors():
            try:
                if mode == 'auto':
                    # AE 지원 센서는 모두 켜기 (run_laundry.py 동일 방식)
                    if sensor.supports(rs.option.enable_auto_exposure):
                        sensor.set_option(rs.option.enable_auto_exposure, 1)
                    if sensor.supports(rs.option.enable_auto_white_balance):
                        sensor.set_option(rs.option.enable_auto_white_balance, 1)
                else:  # fixed (failure_eh)
                    if sensor.supports(rs.option.enable_auto_exposure):
                        sensor.set_option(rs.option.enable_auto_exposure, 0)
                    if sensor.supports(rs.option.enable_auto_white_balance):
                        sensor.set_option(rs.option.enable_auto_white_balance, 0)
                    exp = float(self._bg_failure_params.get('exposure', 500.0))
                    if sensor.supports(rs.option.exposure):
                        sensor.set_option(rs.option.exposure, exp)
                    if 'gain' in self._bg_failure_params and sensor.supports(rs.option.gain):
                        sensor.set_option(rs.option.gain, float(self._bg_failure_params['gain']))
                    if 'white_balance' in self._bg_failure_params:
                        try:
                            if sensor.supports(rs.option.white_balance):
                                sensor.set_option(rs.option.white_balance,
                                                  float(self._bg_failure_params['white_balance']))
                        except Exception:
                            pass
            except Exception as e:
                self._log(f"[VisionEngine][WARN] exposure 설정 실패: {e}")
        if mode == 'auto':
            self._log("[VisionEngine] exposure → auto")
        else:
            exp = float(self._bg_failure_params.get('exposure', 500.0))
            self._log(f"[VisionEngine] exposure → fixed (exp={exp}, "
                      f"gain={self._bg_failure_params.get('gain','?')}, "
                      f"wb={self._bg_failure_params.get('white_balance','?')})")
        self._cur_exposure = mode
        self._log(f"[VisionEngine] 노출 안정화 {flush_n}프레임 flush 중...")
        for _ in range(flush_n):
            self._pipeline.wait_for_frames()

    def _get_frame(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fs = self._pipeline.wait_for_frames()
        fs = self._align.process(fs)
        depth     = np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32) * self._depth_scale
        color_bgr = np.asanyarray(fs.get_color_frame().get_data())
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        return depth, color_bgr, color_rgb

    # ── wm 감지 (세탁기 내부 세탁물, in-process) ─────────────────────────────
    def detect_wm(self, hi_thr: float = WM_RGB_HI_THR,
                  lo_thr: float = WM_RGB_LO_THR) -> dict:
        with self._lock:
            if self._wm_gate is None or self._wm_funcs is None:
                return {"success": False, "centers": [], "reason": "wm_not_initialized"}

            # wm은 항상 auto-exposure 재안정화 (이전 상태 무관하게 60프레임 flush)
            self._cur_exposure = None   # 강제 재설정 트리거
            self._set_exposure('auto', flush_n=60)

            fn  = self._wm_funcs
            o3d = self._wm_o3d

            # RGB gate: 5프레임 EMA 누적 (단발 판단의 노이즈 억제)
            self._wm_gate.reset()
            g = None
            for i in range(5):
                _, color_bgr, _ = self._get_frame()
                g = self._wm_gate.decide(color_bgr, hi_thr=hi_thr, lo_thr=lo_thr)
                if self._show:
                    disp = color_bgr.copy()
                    col  = (0, 255, 0) if g['decision'] == 'CLOTH' else (0, 0, 255)
                    cv2.putText(disp,
                                f"[wm gate {i+1}/5] {g['decision']}  "
                                f"ema={g['ema_prob']:.2f}  cloth_p={g['cloth_prob']:.2f}",
                                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2, cv2.LINE_AA)
                    cv2.imshow("wm RGB gate", disp)
                    cv2.waitKey(80)   # 5 * 80ms = ~400ms 동안 실시간 갱신

            self._log(f"[wm] RGB gate: {g['decision']} cloth_p={g['cloth_prob']:.2f} "
                      f"ema={g['ema_prob']:.2f}")
            if g["decision"] == "EMPTY" and g["ema_prob"] < (hi_thr - 0.05):
                if self._show:
                    cv2.waitKey(1500)   # EMPTY 결과를 1.5초간 보여준 뒤 닫기
                    cv2.destroyWindow("wm RGB gate")
                return {"success": False, "centers": [], "reason": "rgb_gate_empty"}

            # 포인트 클라우드 캡처
            t0 = _time.time()
            curr_pts, color_img, intr = fn['get_pc'](self._pipeline, (WM_Z_MIN, WM_Z_MAX))
            curr_pcd = fn['to_pcd'](curr_pts)
            self._log(f"[wm] 포인트 수: {len(curr_pts)} "
                      f"z=[{curr_pts[:,2].min():.3f},{curr_pts[:,2].max():.3f}]"
                      if len(curr_pts) > 0 else "[wm] 포인트 없음")

            if len(curr_pts) < 100:
                return {"success": False, "centers": [], "reason": "too_few_points"}

            # 드럼 중심 추정 + ICP 정합
            center   = _wm_estimate_center(curr_pcd, o3d)
            cx, cy   = float(center[0]), float(center[1])
            self._log(f"[wm] 드럼 중심 추정: cx={cx:.3f}, cy={cy:.3f}")
            ref_pcd  = _wm_create_cylinder(o3d, WM_CYL_RADIUS, WM_Z_MIN, WM_Z_MAX)
            ref_pts  = np.asarray(ref_pcd.points)
            init_T   = np.eye(4); init_T[0, 3] = -cx; init_T[1, 3] = -cy
            aligned_pcd, T_icp = _wm_icp(curr_pcd, ref_pcd, o3d, init_T=init_T)

            # 다운샘플 + z 필터 + 반지름 필터
            down       = aligned_pcd.voxel_down_sample(0.01)
            pts_down   = np.asarray(down.points)
            z_mask     = ((pts_down[:, 2] >= WM_Z_MIN) & (pts_down[:, 2] <= WM_Z_MAX))
            pts_clipped = pts_down[z_mask]
            r_mask     = _wm_filter_radius(pts_clipped, 0.0, 0.0, WM_FILTER_RADIUS)
            pts_roi    = pts_clipped[r_mask]

            # 이상치 마스크
            ab_mask, _, _ = fn['anomalies'](pts_roi, ref_pts)
            pts_roi = pts_roi[ab_mask]

            # 날개 탐지 + 제거
            _, _, wings = fn['cluster'](pts_roi, np.ones(len(pts_roi), dtype=bool),
                                        eps=WM_EPS, min_samples=WM_MIN_SAMPLES)
            if wings:
                rotated = []
                for w in wings:
                    rotated += [_wm_rotate_cluster(w, 120), _wm_rotate_cluster(w, 240)]
                all_wing_pts = np.vstack(rotated + wings)
                def _near_wing(pt):
                    d = all_wing_pts - pt
                    return np.any((np.linalg.norm(d[:, :2], axis=1) < 0.04)
                                  & (np.abs(d[:, 2]) < 0.15))
                mask_keep = np.array([not _near_wing(p) for p in pts_roi])
            else:
                mask_keep = np.ones(len(pts_roi), dtype=bool)
            final_pts = pts_roi[mask_keep]
            self._log(f"[wm] 날개 제거 후 포인트: {len(final_pts)}")

            # 최종 클러스터링
            mask, _, _  = fn['anomalies'](final_pts, ref_pts)
            centers, clusters, _ = fn['cluster'](final_pts, mask,
                                                  eps=WM_EPS, min_samples=WM_MIN_SAMPLES)
            # color_thresh=1은 거의 모든 클러스터가 MeanShift 재분할을 타서 매우 느려짐 (수 초 단위)
            # → 색상 재분할이 필요 없다고 판단, 기본값(15)으로 복원
            clusters = fn['split_color'](clusters, color_img, intr,
                                         method="mean_shift", color_thresh=15)
            centers = [np.mean(c, axis=0) for c in clusters]

            if not centers:
                self._log("[wm] 클러스터 없음")
                return {"success": False, "centers": [], "reason": "no_clusters"}

            # 좌표 후처리: 정렬 + clamp + ICP 역변환
            centers.sort(key=lambda c: abs(c[0]))
            for c in centers:
                if c[2] >= 0.7: c[2] = 0.7
                if c[1] > 0.29: c[1] = 0.29
            C_ref = np.hstack([centers, np.ones((len(centers), 1))])
            C_cam = (np.linalg.inv(T_icp) @ C_ref.T).T[:, :3]

            self._log(f"[wm] 완료: {len(C_cam)} 클러스터, {_time.time()-t0:.2f}s")
            return {"success": True, "centers": [c.tolist() for c in C_cam]}

    # ── basket 감지 ───────────────────────────────────────────────────────────
    def detect_basket(self, size=384, thr=0.45, zmax=0.35, max_wait_sec=3.0,
                       vote_min_area=150, vote_min_dt=3.0) -> dict:
        """
        TinyUNet(c_in=6) + 단일 유효 프레임 즉시 반환 (vote 없음).
        run_basket_backup.py --no_basket_mask --max_wait_sec 5 와 동일 동작.
        """
        with self._lock:
            self._set_exposure('auto')   # basket은 auto-exposure
            t0 = _time.time()
            fx, fy, ppx, ppy = self._intr_color

            while True:
                elapsed = _time.time() - t0
                if elapsed > max_wait_sec:
                    return {"success": False, "empty": True, "mode": "basket",
                            "centers": [], "pixel_center": [], "theta": 0.0,
                            "area": 0, "dt": 0.0}

                depth, _, color_rgb = self._get_frame()
                H, W = depth.shape

                x_np = _preprocess_basket(depth, color_rgb, self._bg_basket, size, zmax)
                x    = torch.from_numpy(x_np).unsqueeze(0).to(self._dev)
                with torch.no_grad():
                    logits = self._basket_net(x.float())

                mask  = _postprocess_mask_basket(logits, thr, (H, W))
                candidates = _analyze_mask_multi(mask, min_area=vote_min_area, min_dt=vote_min_dt)

                if not candidates:
                    continue

                # 3D 좌표 변환 (유효한 depth가 있는 후보만)
                centers_3d = []
                for area_c, dt_c, cx_c, cy_c in candidates:
                    local = depth[max(0,cy_c-2):cy_c+3, max(0,cx_c-2):cx_c+3]
                    val   = local[local > 0]
                    if val.size == 0:
                        continue
                    z_m = float(np.median(val))
                    X, Y, Z = _uvz_to_xyz(cx_c, cy_c, z_m, fx, fy, ppx, ppy)
                    if Z < 0.55:
                        Z += 0.05
                    centers_3d.append([X, Y, Z])

                if not centers_3d:
                    continue

                # theta: 가장 큰 CC 기준 (기존 동작 유지)
                _, theta, _, _ = _analyze_mask(mask)
                first_area, first_dt, first_cx, first_cy = candidates[0]

                return {"success": True, "empty": False, "mode": "basket",
                        "centers": centers_3d, "pixel_center": [first_cx, first_cy],
                        "theta": float(theta), "area": int(first_area), "dt": float(first_dt)}

    # ── failure_eh 감지 ───────────────────────────────────────────────────────
    def detect_failure_eh(self, vote_k=10, vote_min_area=150, vote_min_dt=3.0,
                           max_sec=30.0, zmax=0.5, strict=False) -> dict:
        """
        RGB gate → depth heuristic → vote_k 프레임 누적.
        run_failure_eh.py --vote --vote_k 10 --vote_require_curr 와 동일 동작.

        strict=True: failure_eh_strict (vote_min_area=300, vote_min_dt=5)
        """
        if strict:
            vote_min_area, vote_min_dt = 300, 5.0

        with self._lock:
            self._set_exposure('auto', flush_n=30)

            self._rgb_gate.reset()       # EMA 초기화 (subprocess 방식과 동일)
            votes     = collections.deque(maxlen=max(1, vote_k))
            t0        = _time.time()
            last_ok_t = -1e9
            fx, fy, ppx, ppy = self._intr_color  # aligned depth → color frame 기준 intrinsics

            while True:
                now = _time.time()
                if now - t0 > max_sec:
                    return {"success": False, "reason": "timeout"}

                depth, color_bgr, color_rgb = self._get_frame()
                H, W = depth.shape

                # ── RGB gate: 빈 세탁기 빠른 감지 ─────────────────────────────
                g = self._rgb_gate.decide(color_bgr, hi_thr=0.60, lo_thr=0.55)
                if g["decision"] == "EMPTY":
                    return {"success": False, "reason": "rgb_gate_empty", "object_count": 0}

                # ── TinyUNet (failure domain=1) 세탁물 segmentation ──────────
                x_np  = _preprocess_failure(depth, color_rgb, self._bg_failure, zmax=zmax)
                x_t   = torch.from_numpy(x_np).unsqueeze(0).to(self._dev)
                with torch.no_grad():
                    logits = self._failure_net(x_t.float())
                mask_u8 = _postprocess_mask_failure(logits, thr=0.45, out_hw=(H, W))

                # TinyUNet mask → basket과 동일하게 _analyze_mask (단순 centroid, offset 없음)
                center, theta, area, dt_peak = _analyze_mask(mask_u8)

                accepted = False
                xyz: Optional[tuple] = None
                if center is not None and area >= vote_min_area and dt_peak >= vote_min_dt:
                    cx, cy = center
                    local  = depth[max(0,cy-2):cy+3, max(0,cx-2):cx+3]
                    val    = local[local > 0]
                    z_m    = float(np.median(val)) if val.size > 0 else 0.07
                    X, Y, Z = _uvz_to_xyz(cx, cy, z_m, fx, fy, ppx, ppy)
                    xyz = (X, Y, Z); last_ok_t = now; accepted = True
                else:
                    # TinyUNet mask 불충분 → depth heuristic fallback
                    # fallback은 _grasp_from_mask (cx+width/8 offset 포함) 유지
                    mask_heu = _depth_to_sky_mask(depth, zmax=zmax)
                    center_h, _, dt_h, area_h = _grasp_from_mask(mask_heu)
                    if center_h is not None and area_h >= vote_min_area and dt_h >= vote_min_dt:
                        cx, cy = center_h
                        local  = depth[max(0,cy-2):cy+3, max(0,cx-2):cx+3]
                        val    = local[local > 0]
                        z_m    = float(np.median(val)) if val.size > 0 else 0.07
                        X, Y, Z = _uvz_to_xyz(cx, cy, z_m, fx, fy, ppx, ppy)
                        xyz = (X, Y, Z); last_ok_t = now; accepted = True

                # ── vote (vote_require_curr=True) ────────────────────────────
                if accepted and xyz is not None:
                    votes.append(xyz)

                # ── 시각화 (--show) ───────────────────────────────────────────
                if self._show:
                    info = f"area={area} dt={dt_peak:.1f} votes={len(votes)}/{vote_k}"
                    view = _draw_overlay(color_bgr, mask_u8, center, theta if center else 0.0,
                                         xyz, info_text=info)
                    cv2.imshow("failure_eh detect", view)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        return {"success": False, "reason": "user_quit"}

                if len(votes) >= vote_k and (now - last_ok_t) <= 1.0:
                    arr = np.array(votes, np.float32)
                    med = np.median(arr, axis=0).tolist()
                    med[2] += 0.01   # 1cm offset (run_failure_eh.py 동일)
                    if self._show:
                        cv2.destroyAllWindows()
                    return {"success": True, "centers": [med]}

    # ── 종료 ──────────────────────────────────────────────────────────────────
    def shutdown(self):
        try:
            self._pipeline.stop()
            self._log("[VisionEngine] 카메라 종료 완료")
        except Exception:
            pass


# ── ROS2 Action Server ────────────────────────────────────────────────────────
class TargetPoseServer(Node):
    def __init__(self, device: str = 'cpu', show: bool = False):
        super().__init__('target_pose_server')
        self.get_logger().info(f"=== IN-PROCESS NODE2 STARTED (device={device}, show={show}) ===")
        self.device = device       

        self._vision = VisionEngine(device=device, log_fn=self.get_logger().info, show=show)
        self._basket_centers_cache: list = []  # basket 감지 결과 후보 캐시

        self._action_server = ActionServer(
            self, TargetPose, 'target_pose', self.execute_callback)
        self.get_logger().info("=== Action Server 준비 완료 ===")

    def execute_callback(self, goal_handle):
        self.get_logger().info('target_pose 요청 받음')

        if not goal_handle.request.start:
            goal_handle.abort()
            return TargetPose.Result(success=False)

        mode = goal_handle.request.mode
        self.get_logger().info(f"mode={mode}")

        try:
            if mode in ("basket", "basket_eh"):
                result_dict = self._vision.detect_basket()
                self._basket_centers_cache = result_dict.get("centers") or []
                self.get_logger().info(f"[basket] 후보 {len(self._basket_centers_cache)}개 캐시")
            elif mode.startswith("basket:"):
                # 캐시된 N번째 후보 반환 (재감지 없음)
                try:
                    idx = int(mode.split(":", 1)[1])
                except (ValueError, IndexError):
                    self.get_logger().warn(f"[basket:N] 잘못된 인덱스: {mode}")
                    goal_handle.abort()
                    return TargetPose.Result(success=False)
                if idx < len(self._basket_centers_cache):
                    center = self._basket_centers_cache[idx]
                    self.get_logger().info(
                        f"[basket:{idx}] 캐시 후보 반환: {center} "
                        f"(전체 {len(self._basket_centers_cache)}개)"
                    )
                    result_dict = {"success": True, "centers": [center], "area": 99999}
                else:
                    self.get_logger().warn(
                        f"[basket:{idx}] 후보 없음 (캐시={len(self._basket_centers_cache)}개)"
                    )
                    result_dict = {"success": False, "centers": []}
            elif mode in ("failure_eh", "failure_eh_strict"):
                strict = (mode == "failure_eh_strict")
                vote_k = 3  # failure_eh/strict 모두 3프레임 median
                result_dict = self._vision.detect_failure_eh(strict=strict, vote_k=vote_k)
            elif mode == "wm":
                result_dict = self._vision.detect_wm()
            else:
                self.get_logger().error(f"지원하지 않는 mode='{mode}'")
                goal_handle.abort()
                return TargetPose.Result(success=False)

        except Exception as e:
            self.get_logger().error(f"inference 예외: {e}")
            import traceback; traceback.print_exc()
            goal_handle.abort()
            return TargetPose.Result(success=False)

        return self._dict_to_result(goal_handle, mode, result_dict)

    def _dict_to_result(self, goal_handle, mode, d: dict):
        """result dict → TargetPose.Result. 빈 경우 sentinel z=-99.0 반환."""
        reason  = d.get("reason", "")
        centers = d.get("centers") or []
        is_empty = (reason in ("rgb_gate_empty", "timeout") or
                    not d.get("success", False) or not centers)

        if is_empty:
            sentinel = Pose(); sentinel.position.z = -99.0
            res = TargetPose.Result(success=True, target_pose=sentinel)
            goal_handle.succeed()
            return res

        cx, cy, cz = map(float, centers[0])
        target_pose = Pose()
        target_pose.position.x = cx
        target_pose.position.y = cy
        target_pose.position.z = cz
        target_pose.orientation.w = 1.0
        self.get_logger().info(f"[{mode}] 좌표 반환: x={cx:.3f}, y={cy:.3f}, z={cz:.3f}")

        res = TargetPose.Result(success=True, target_pose=target_pose)
        goal_handle.succeed()
        return res

    def destroy_node(self):
        self._vision.shutdown()
        super().destroy_node()


# ── main ──────────────────────────────────────────────────────────────────────
def main(args=None):
    parser = argparse.ArgumentParser(description='In-process vision node (node2)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='추론 디바이스 (cpu / cuda / cuda:0)')
    parser.add_argument('--gpu', action='store_true',
                        help='--device cuda 와 동일 (편의 flag)')
    parser.add_argument('--show', action='store_true',
                        help='failure_eh 감지 중 cv2.imshow로 마스크·좌표 실시간 시각화')
    parsed, ros_args = parser.parse_known_args()
    device = 'cuda' if parsed.gpu else parsed.device

    rclpy.init(args=ros_args or args)
    node = TargetPoseServer(device=device, show=parsed.show)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
