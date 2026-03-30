'''
vis mode:
python run_failure.py \
  --mode failure --json \
  --ckpt /home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/unet_best.pt \
  --intr_json /home/dyros/mcy_ws/Laundry_decision/run_bf/intrinsics_depth_640x480.json \
  --size 384 --zmax 0.5 --thr 0.45 \
  --diff_mode blend \
  --vote --vote_k 10 --vote_min_area 150 --vote_min_dt 3 --vote_require_curr \
  --stale_sec 1.0 \
  --save_dir /home/dyros/mcy_ws/Laundry_decision/run_bf/_dbg \
  --save_best_only --prefix failure --show
  
  

  python run_failure.py \
  --mode failure \
  --ckpt /home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/unet_best.pt \
  --intr_json /home/dyros/mcy_ws/Laundry_decision/run_bf/intrinsics_depth_640x480.json \
  --size 384 --zmax 0.5 --thr 0.45 \
  --diff_mode blend \
  --vote --vote_k 10 --vote_min_area 150 --vote_min_dt 3 --vote_require_curr \
  --stale_sec 1.0 \
  --save_dir /home/dyros/mcy_ws/Laundry_decision/run_bf/_dbg \
  --save_best_only --prefix failure
'''

import os, json, time, argparse, math, collections
import numpy as np
import cv2
import torch, torch.nn as nn

import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image

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
    def __init__(self, c_in=5, base=32, use_se=True):
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

# run_failure.py의 TinyUNet 클래스 뒤에 추가

# === run_laundry.py에서 가져온 RGB Gate 로직 ===
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _rgb_eval_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def _create_backbone(model_name: str = "resnet18", num_classes: int = 2) -> nn.Module:
    if model_name == "resnet18":
        from torchvision.models import resnet18
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif model_name == "mobilenet_v3_small":
        from torchvision.models import mobilenet_v3_small
        m = mobilenet_v3_small(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    # (필요 시 'efficientnet_b0' 등 다른 모델 추가)
    else:
        raise ValueError(f"unknown model: {model_name}")

### debug rgb_gate ###
class _RGBGate:
    def __init__(self, ckpt_path: str, device_str: str = "cuda", ema_decay: float = 0.7):
        import os, json
        assert os.path.isfile(ckpt_path), f"ckpt not found: {ckpt_path}"
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.tfm = _rgb_eval_transform()
        self.ema_decay = ema_decay
        self.ema_prob = None
        self.ts = None
        self.model = None
        self.idx_to_class = {0: "empty", 1: "cloth"}
        self.cloth_idx = 1

        # 1) TorchScript 로드 시도
        try:
            self.ts = torch.jit.load(ckpt_path, map_location=self.device)
            self.ts.eval()
            meta_path = ckpt_path + ".json"
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if "idx_to_class" in meta:
                    self.idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
                    self.cloth_idx = [k for k, v in self.idx_to_class.items() if v == "cloth"]
                    self.cloth_idx = self.cloth_idx[0] if len(self.cloth_idx) else 1
            print(f"[rgb-gate] TorchScript loaded: {ckpt_path}")
            return
        except Exception:
            self.ts = None

        # 2) state_dict 로드
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        state_dict = None
        args = {}
        
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
            args = ckpt.get("args", {})
        else:
            # Checkpoint itself is likely the state_dict
            state_dict = ckpt
            args = {}

        model_name = args.get("model", "resnet18")
        self.model = _create_backbone(model_name, num_classes=2).to(self.device).eval()

        # Handle "model." prefix if present (common in Lightning or wrapper saves)
        # We need keys to match the backbone (conv1, bn1, ...)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict, strict=False)

        cls2idx = {}
        if isinstance(ckpt, dict):
             cls2idx = ckpt.get("class_to_idx", {"empty": 0, "cloth": 1})
        else:
             cls2idx = {"empty": 0, "cloth": 1}

        self.idx_to_class = {v: k for k, v in cls2idx.items()}
        self.cloth_idx = [k for k, v in self.idx_to_class.items() if v == "cloth"]
        self.cloth_idx = self.cloth_idx[0] if len(self.cloth_idx) else 1
        print(f"[rgb-gate] state_dict loaded: {ckpt_path}, cloth_idx={self.cloth_idx}")

    @torch.inference_mode()
    def _predict(self, bgr_img):
        import numpy as np
        assert bgr_img is not None and bgr_img.ndim == 3 and bgr_img.shape[2] == 3
        rgb = bgr_img[..., ::-1] # BGR -> RGB
        x = self.tfm(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        if self.ts is not None:
            logits = self.ts(x)
        else:
            logits = self.model(x)
        prob = torch.softmax(logits, dim=1)[0]
        pred_idx = int(prob.argmax().item())
        cloth_prob = float(prob[self.cloth_idx].item()) if 0 <= self.cloth_idx < prob.numel() else float(prob.max().item())
        return pred_idx, cloth_prob

    @torch.inference_mode()
    def decide(self, bgr_img, hi_thr: float = 0.60, lo_thr: float = 0.45):
        pred_idx, p = self._predict(bgr_img)
        self.ema_prob = p if self.ema_prob is None else (self.ema_decay * self.ema_prob + (1 - self.ema_decay) * p)
        if self.ema_prob >= hi_thr:
            decision = "CLOTH"
        elif self.ema_prob < lo_thr:
            decision = "EMPTY"
        else:
            decision = "CLOTH" # 보수적
        return {
            "pred_label": self.idx_to_class.get(pred_idx, str(pred_idx)),
            "cloth_prob": float(p),
            "ema_prob": float(self.ema_prob),
            "decision": decision
        }
# === RGB Gate 로직 끝 ===

# --------------- Camera / geometry helpers -------------------
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

# ---- Grid helpers -------------------------------------------------
def _compute_grid_index(u, v, W, H, nx, ny):
    """픽셀(u,v)이 nx×ny 그리드에서 속한 (idx,row,col) 반환"""
    cell_w = max(1, W // nx)
    cell_h = max(1, H // ny)
    col = int(np.clip(u // cell_w, 0, nx - 1))
    row = int(np.clip(v // cell_h, 0, ny - 1))
    idx = int(row * nx + col)
    return idx, row, col

def _mask_to_occupied_cells(mask_u8, W, H, nx, ny, min_px=80):
    """마스크 내 픽셀 카운트가 min_px 이상인 셀 idx 목록"""
    if mask_u8 is None:
        return []
    m = mask_u8.astype(np.uint8)
    ys, xs = np.where(m > 0)
    if xs.size == 0:
        return []
    cell_w = max(1, W // nx)
    cell_h = max(1, H // ny)
    cols = np.clip(xs // cell_w, 0, nx - 1)
    rows = np.clip(ys // cell_h, 0, ny - 1)
    idxs = rows * nx + cols
    uniq, cnts = np.unique(idxs, return_counts=True)
    return [int(i) for i, c in zip(uniq, cnts) if c >= int(min_px)]
# --------------- debug save ----------------

def draw_overlay(color_bgr, mask_u8, center, theta, xyz=None, alpha=0.35, thr_text=None):
    out = color_bgr.copy()
    if mask_u8 is not None:
        m = mask_u8
        if m.dtype != np.uint8: m = m.astype(np.uint8)
        if m.max() == 1: m = (m * 255).astype(np.uint8)
        overlay = np.dstack([np.zeros_like(m), m, np.zeros_like(m)])  # green
        overlay = cv2.GaussianBlur(overlay, (5,5), 0)
        out = cv2.addWeighted(out, 1.0, overlay, alpha, 0)

    if center is not None:
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(out, (cx, cy), 6, (0,0,255), -1)
        if theta is None: theta = 0.0
        L = 40
        x2 = int(cx + L*np.cos(theta)); y2 = int(cy + L*np.sin(theta))
        x3 = int(cx - L*np.cos(theta)); y3 = int(cy - L*np.sin(theta))
        cv2.line(out, (x3,y3), (x2,y2), (255,0,0), 2)

    text = []
    if thr_text is not None: text.append(f"thr={thr_text:.2f}")
    if xyz is not None: text.append(f"XYZ=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}) m")
    if text:
        cv2.putText(out, " | ".join(text), (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return out

def save_debug_bundle(color_bgr, mask_u8, center, theta, xyz,
                      out_dir, prefix="failure", tag="best", stamp=None,
                      save_raw=True, save_mask=True, save_overlay=True, meta_extra=None):
    os.makedirs(out_dir, exist_ok=True)
    if stamp is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
    base = f"{prefix}_{tag}_{stamp}"

    if save_raw and color_bgr is not None:
        cv2.imwrite(os.path.join(out_dir, base + "_rgb.jpg"), color_bgr)

    if save_mask and mask_u8 is not None:
        m = mask_u8
        if m.dtype != np.uint8: m = m.astype(np.uint8)
        if m.max() == 1: m = (m * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, base + "_mask.png"), m)

    if save_overlay and color_bgr is not None:
        overlay = draw_overlay(color_bgr, mask_u8, center, theta, xyz=xyz)
        cv2.imwrite(os.path.join(out_dir, base + "_overlay.jpg"), overlay)

    meta = {
        "center_uv": [float(center[0]), float(center[1])] if center is not None else None,
        "theta_rad": float(theta) if theta is not None else None,
        "xyz_m": [float(x) for x in xyz] if xyz is not None else None,
        "tag": tag, "prefix": prefix, "timestamp": stamp,
    }
    if meta_extra: meta.update(meta_extra)
    with open(os.path.join(out_dir, base + "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
# --------------- Pre/Post process -----------------
def compute_diff(depth_m, bg_m, mode='min'):
    """
    mode:
      - 'bg'         : |bg - depth| 만 사용
      - 'bilateral'  : |bilateral(depth) - depth| 만 사용 (배경 없어도 동작)
      - 'min'        : 픽셀단위 min(bg_diff, bi_diff)
      - 'blend'      : 배경이 유효한 픽셀은 bg_diff, 나머지는 bi_diff
    """
    depth = depth_m.astype(np.float32)
    valid = depth > 0

    # bilateral 기반 diff — 32F 유지!
    base = cv2.bilateralFilter(depth, d=5, sigmaColor=0.05, sigmaSpace=7)
    diff_bi = np.abs(base - depth)
    diff_bi[~valid] = 0.0

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

def preprocess_inputs(depth_m, color_rgb, bg_m, size, zmax, diff_mode='min'):
    """return 5ch tensor [diff, z, rgb] resized to (size,size)"""
    diff = compute_diff(depth_m, bg_m, mode=diff_mode)
    z = np.clip(depth_m, 0.0, zmax) / max(1e-6, zmax)

    diff_r = cv2.resize(diff, (size, size), interpolation=cv2.INTER_NEAREST)
    z_r    = cv2.resize(z,    (size, size), interpolation=cv2.INTER_NEAREST)

    if color_rgb is None:
        rgb_r = np.zeros((size, size, 3), dtype=np.float32)
    else:
        rgb = color_rgb.astype(np.float32) / 255.0
        rgb_r = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)

    x = np.stack([diff_r, z_r], axis=0)  # (2, H, W)
    x = np.concatenate([x, np.transpose(rgb_r, (2,0,1))], axis=0)  # (5, H, W)
    return x.astype(np.float32)

def postprocess_mask(logits, thr, out_hw):
    prob = torch.sigmoid(logits)[0,0].cpu().numpy()
    m = (prob >= thr).astype(np.uint8)
    H, W = out_hw
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    return m, prob

def largest_cc(mask):
    num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    if num <= 1: return None, 0
    best_label, best_area = None, 0
    for lb in range(1, num):
        area = (labels==lb).sum()
        if area > best_area:
            best_area, best_label = area, lb
    return (labels==best_label).astype(np.uint8), int(best_area)

def grasp_from_mask(mask):
    """return (cx,cy), angle(rad), dt_peak"""
    cc, area = largest_cc(mask)
    if cc is None: return None, None, 0.0, 0

    dt = cv2.distanceTransform(cc, cv2.DIST_L2, 3)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dt)
    cy, cx = maxLoc[1], maxLoc[0]
    dt_peak = float(maxVal)

    ys, xs = np.where(cc>0)
    if len(xs) < 5:
        theta = 0.0
    else:
        x = xs.astype(np.float32); y = ys.astype(np.float32)
        x -= x.mean(); y -= y.mean()
        cov_xx = (x*x).mean(); cov_yy = (y*y).mean(); cov_xy = (x*y).mean()
        theta = 0.5 * math.atan2(2*cov_xy, (cov_xx - cov_yy))  # radians
    return (cx, cy), theta, dt_peak, area

def draw_overlay(color_bgr, mask, center, theta, xyz, thr=0.6, info_text=""): ############### 원래 0.45
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
    if xyz is not None:
        t += f" | XYZ=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}) m"
    if info_text:
        t += " | " + info_text
    cv2.putText(out, t, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return out



# --------------- Main -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['failure_bg','failure'], default='failure')
    ap.add_argument('--json', action='store_true', help='마지막 줄에 JSON 출력하고 종료')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--intr_json', required=True)
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--zmax', type=float, default=0.5)
    ap.add_argument('--thr', type=float, default=0.6) ########## 원래 0.45
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--diff_mode', choices=['bg','bilateral','min','blend'], default='blend')
    ap.add_argument('--show', action='store_true', help='시각화창 표시')
    ap.add_argument('--save_dir', default='/home/dyros/mcy_ws/Laundry_decision/run_failure/_dbg',
                help='디버그 이미지 저장 폴더')
    ap.add_argument('--save_all', action='store_true',
                    help='프레임마다 마스크/오버레이 저장')
    ap.add_argument('--save_best_only', action='store_true',
                    help='최종 선택(투표 결과) 프레임만 저장')
    ap.add_argument('--save_on_success_only', action='store_true',
                    help='검출 성공시에만 저장')
    ap.add_argument('--prefix', default='failure',
                help='파일 이름 prefix')

    # BG capture / load
    ap.add_argument('--bg_frames', type=int, default=20)
    ap.add_argument('--bg_path', default='/home/dyros/mcy_ws/Laundry_decision/run_bf/laundry_bg/bg_failure.npy')

    # Voting & gating
    ap.add_argument('--vote', action='store_true')
    ap.add_argument('--vote_k', type=int, default=10)
    ap.add_argument('--vote_min_area', type=int, default=150)
    ap.add_argument('--vote_min_dt', type=float, default=3.0)
    ap.add_argument('--vote_require_curr', action='store_true')
    ap.add_argument('--stale_sec', type=float, default=1.0)
    ap.add_argument('--max_sec', type=float, default=30.0)
    # cell occupied decision
    ap.add_argument('--grid', type=int, default=3, help='정사각 그리드(nx=ny=grid)')
    ap.add_argument('--grid_w', type=int, default=4, help='가로 칸 수(미지정 시 grid 사용)')
    ap.add_argument('--grid_h', type=int, default=1, help='세로 칸 수(미지정 시 grid 사용)')
    ap.add_argument('--cell_px', type=int, default=80, help='셀 점유 판정 최소 픽셀 수')
    ap.add_argument('--draw_grid', action='store_true', help='미리보기 창에 그리드/섹션 표시')
    
    # --- [DEBUG RGB_Gate] ---
    ap.add_argument("--rgb_ckpt", type=str, default=None,
                    help="RGB empty/cloth 분류 ckpt (선택 사항)")
    ap.add_argument("--rgb_thr", type=float, default=0.60,
                    help="cloth 확률 상한 임계값")
    ap.add_argument("--rgb_low_thr", type=float, default=0.45,
                    help="cloth 확률 하한 임계값(히스테리시스)")
    # ------

    args = ap.parse_args()
    dev = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # load net
    ckpt = torch.load(args.ckpt, map_location='cpu')
    meta = ckpt
    c_in = int(meta.get('c_in', 5))   # RGB 포함 학습이면 5
    base = int(meta.get('base', 32))
    use_se = bool(meta.get('use_se', True))
    net = TinyUNet(c_in=c_in, base=base, use_se=use_se).to(dev)
    net.load_state_dict(meta['model'])
    net.eval()

    # --- [debug RGB_Gate] ---
    rgb_gate = None
    if args.rgb_ckpt:
        try:
            rgb_gate = _RGBGate(args.rgb_ckpt, device_str=dev)
        except Exception as e:
            print(f"[rgb-gate][WARN] 모델 로드 실패: {e}")
    # ------


    fx, fy, ppx, ppy = load_intrinsics(args.intr_json)

    # RealSense
    try:
        import pyrealsense2 as rs
    except Exception as e:
        print("[ERR] pyrealsense2 not found:", e)
        if args.json:
            print(json.dumps({"success": False, "error": "pyrealsense2 not found"}))
        return

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # meters per unit

    # ---------------- mode: failure_bg (capture & save background) ----------------
    if args.mode == 'failure_bg':
        os.makedirs(os.path.dirname(args.bg_path), exist_ok=True)
        frames = []
        print(f"[INFO] capturing {args.bg_frames} frames for background...")
        for _ in range(args.bg_frames):
            fs = pipeline.wait_for_frames()
            fs = align.process(fs)
            d = np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32) * depth_scale
            frames.append(d)
            if args.show:
                # Show RGB for background capture visualization
                c = np.asanyarray(fs.get_color_frame().get_data())
                cv2.imshow("failure_bg_capture", c)
                cv2.waitKey(1)
        if args.show:
             print("[INFO] Background capture done. Waiting 2s...")
             cv2.waitKey(2000)

        bg_m = np.median(np.stack(frames,0), axis=0).astype(np.float32)
        np.save(args.bg_path, bg_m)
        pipeline.stop()
        if args.json:
            print(json.dumps({"success": True, "bg_path": args.bg_path}))
        else:
            print(f"[OK] background saved to {args.bg_path}")
        return

    # ---------------- mode: failure (detect with saved background) ----------------
    # try load bg
    bg_m = None
    if os.path.exists(args.bg_path):
        try:
            bg_m = np.load(args.bg_path).astype(np.float32)
            print(f"[INFO] loaded background: {args.bg_path}")
        except Exception as e:
            print(f"[WARN] failed to load bg ({e}), falling back to diff={args.diff_mode}")

    start_t = time.time()
    last_valid_t = -1e9
    votes = collections.deque(maxlen=max(1, args.vote_k))

    # UI
    overlay_on = True
    thr = float(args.thr)

    success_out = None  # will be JSON object when success
    best = {
    "color": None, "mask": None, "center": None, "theta": None, "xyz": None,
    "score": -1.0
    }

    def candidate_score(mask_u8, dt=None):
        if mask_u8 is None: return -1.0
        m = mask_u8
        if m.dtype != np.uint8: m = m.astype(np.uint8)
        if m.max() == 1: m = (m * 255).astype(np.uint8)
        area = int(cv2.countNonZero(m))
        s = float(area)
        if dt is not None:
            s += float(dt.max()) * 50.0   # 중심성이 높을수록 가산점
        return s
    cnt =0
    while True:
        # 1. Capture Frame FIRST
        fs = pipeline.wait_for_frames()
        # auto-exposure stabilization
        if cnt < 30:
            cnt += 1
            if cnt % 10 == 0: 
                 print(f"[INFO] Camera Warmup: {cnt}/30")
            continue

        now = time.time()
        if now - start_t > args.max_sec and success_out is None:
            # timeout
            if args.json:
                print(json.dumps({"success": False, "reason": "timeout"}))
            else:
                print("[TIMEOUT] detection timeout.")
            break

        # fs = pipeline.wait_for_frames() # Moved up
        fs = align.process(fs)
        dfrm = fs.get_depth_frame()
        cfrm = fs.get_color_frame()
        if not dfrm or not cfrm:
            continue

        depth = np.asanyarray(dfrm.get_data()).astype(np.float32) * depth_scale  # meters
        color_bgr = np.asanyarray(cfrm.get_data())  # BGR
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        
        H, W = depth.shape
        
        # # --- Gamma Correction for Visualization ---
        # # Dark environment fix
        # val = 1.5 # gamma value ( >1 brightens )
        # invGamma = 1.0 / val
        # table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        # color_bgr_vis = cv2.LUT(color_bgr, table)
        # ------------------------------------------

        # --- [debug RGB_Gate ] ---
        if rgb_gate is not None:
            # [Fix] Use FULL IMAGE to match training data
            # Old Code: img_crop = color_bgr[:,:crop_w+250] 
            img_crop = color_bgr 
            
            # [DEBUG] Save Input Image
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                debug_path = os.path.join(args.save_dir, f"rgb_gate_input_{time.time():.3f}.jpg")
                cv2.imwrite(debug_path, img_crop)
                print(f"[DEBUG] Saved RGB Gate input to: {debug_path}")
                
                # [Internal Verify] Reload and Predict
                try:
                    pil_img = Image.open(debug_path).convert("RGB")
                    # Manual transform
                    xx = rgb_gate.tfm(pil_img).unsqueeze(0).to(rgb_gate.device)
                    with torch.no_grad():
                         ll = rgb_gate.model(xx)
                         pp = torch.softmax(ll, dim=1)[0]
                         p_c = pp[rgb_gate.cloth_idx].item()
                    print(f"[DEBUG] Prediction on RELOADED file: p_cloth={p_c:.4f}")
                except Exception as e:
                    print(f"[DEBUG] Reload check failed: {e}")

            img_crop_for_vis = img_crop.copy()

            g = rgb_gate.decide(img_crop, hi_thr=args.rgb_thr, lo_thr=args.rgb_low_thr)
            
            # 매우 EMPTY에 가깝다고 확신되면 TinyUNet 분석을 건너뜀
            if g["decision"] == "EMPTY":
                log = f"rgb-empty-skip (full) p={g['cloth_prob']:.2f} ema={g['ema_prob']:.2f} thr={args.rgb_thr:.2f}"
                print("[failure] " + log)
                
                # JSON 모드일 경우 "비어있음"을 출력하고 종료
                if args.json:
                    print(json.dumps({"success": False, "reason": "rgb_gate_empty", "object_count": 0}))
                    break # while True 루프 탈출
                
                # 시각화 모드일 경우 다음 프레임으로 넘어감
                # (단, 시각화 로직이 뒤에 있으므로 여기서는 그냥 continue)
                # ... (또는 UI에 'EMPTY' 표시) ...
                
                # (TinyUNet 분석을 건너뛰고 싶다면)
                # continue # -> 이렇게 하면 TinyUNet 분석을 건너뛰고 다음 프레임으로 감
                
                # (일단은 로그만 찍고, TinyUNet이 2차 검증하도록 함)
                # (만약 'EMPTY'일 때 완전히 멈추고 싶다면 아래 3줄을 활성화)
                if args.json:
                    success_out = {"success": False, "reason": "rgb_gate_empty", "object_count": 0}
                    break # while True 종료
                continue # 시각화 모드에서 다음 프레임으로

            else:
                 log = f"rgb-cloth (crop) p={g['cloth_prob']:.2f} ema={g['ema_prob']:.2f} thr={args.rgb_thr:.2f}"
                 print("[failure] " + log)
        # --- [RGB Gate 검사 끝] ---

        x_np = preprocess_inputs(depth, color_rgb, bg_m, args.size, args.zmax, diff_mode=args.diff_mode)
        x = torch.from_numpy(x_np).unsqueeze(0).to(dev)

        with torch.no_grad():
            z = net(x.float())

        mask, prob = postprocess_mask(z, thr, (H, W))
        center, theta, dt_peak, area = grasp_from_mask(mask)
        nx = int(args.grid_w or args.grid)
        ny = int(args.grid_h or args.grid)

        section_idx = None
        section_row = None
        section_col = None
        occupied_cells = []

        if mask is not None:
            occupied_cells = _mask_to_occupied_cells(mask, W, H, nx, ny, min_px=args.cell_px)

        if center is not None:
            cx, cy = int(center[0]), int(center[1])
            section_idx, section_row, section_col = _compute_grid_index(cx, cy, W, H, nx, ny)

        # JSON/저장에 함께 넣을 공통 페이로드
        _grid_payload = {
            "grid": {"nx": nx, "ny": ny},
            "section_idx": (int(section_idx) if section_idx is not None else None),
            "section_row": (int(section_row) if section_row is not None else None),
            "section_col": (int(section_col) if section_col is not None else None),
            "occupied_cells": occupied_cells,
            "center_uv": [int(center[0]), int(center[1])] if center is not None else None
        }
        # gating
        accepted = False
        xyz = None
        if center is not None and area >= args.vote_min_area and dt_peak >= args.vote_min_dt:
            cx, cy = center
            # local depth median
            y0, y1 = max(0, cy-2), min(H, cy+3)
            x0, x1 = max(0, cx-2), min(W, cx+3)
            local = depth[y0:y1, x0:x1]
            val = local[local > 0]
            if val.size == 0:
                raise ValueError("VAL SIZE IS 0!!!")
            if val.size > 0:
                z_m = float(np.median(val))
                X,Y,Z = uvz_to_xyz(cx, cy, z_m, fx, fy, ppx, ppy)
                xyz = (X,Y,Z)
                last_valid_t = now
                accepted = True

        # voting
        if args.vote:
            if args.vote_require_curr:
                # 현재 프레임이 유효할 때만 표 추가
                if accepted and xyz is not None:
                    votes.append(xyz)
            else:
                # 이전 유효 검출이 최근(stale_sec) 내면 유지
                if (now - last_valid_t) <= args.stale_sec:
                    if xyz is not None:
                        votes.append(xyz)
            # 승리 조건
            if len(votes) >= args.vote_k and (now - last_valid_t) <= args.stale_sec:
                arr = np.array(votes, dtype=np.float32)
                med = np.median(arr, axis=0).tolist()
                med[2] += 0.01 # 1cm offset
                success_out = {"success": True, "centers": [med], **_grid_payload}
        else:
            # 단발성: 현재 프레임 유효하면 즉시 성공
            if accepted and xyz is not None:
                l_xyz = list(xyz)
                l_xyz[2] += 0.01 # 1cm offset
                success_out = {"success": True, "centers": [l_xyz], **_grid_payload}

        # UI
        if args.show:
            # print("visualize")
            info = f"area={area} dt={dt_peak:.1f} votes={len(votes)}"
            # Use color_bgr_vis instead of color_bgr
            view = draw_overlay(color_bgr, mask, center, theta if center else 0.0, xyz, thr, info_text=info)
            # view = draw_overlay(color_bgr_vis, mask, center, theta if center else 0.0, xyz, thr, info_text=info)

            if args.draw_grid:
                cell_w = max(1, W // nx)
                cell_h = max(1, H // ny)
                # 격자선
                for i in range(1, nx):
                    x = i * cell_w
                    cv2.line(view, (x, 0), (x, H - 1), (0, 255, 255), 1)
                for j in range(1, ny):
                    y = j * cell_h
                    cv2.line(view, (0, y), (W - 1, y), (0, 255, 255), 1)
                # 셀 인덱스 라벨
                idx = 0
                for r in range(ny):
                    for c in range(nx):
                        x = c * cell_w + cell_w // 2
                        y = r * cell_h + cell_h // 2
                        cv2.putText(view, str(idx), (x - 10, y + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                        idx += 1
                # 현재 center가 속한 셀 강조
                if section_idx is not None:
                    x0 = (section_idx % nx) * cell_w
                    y0 = (section_idx // nx) * cell_h
                    cv2.rectangle(view, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), (0, 255, 255), 2)
            cv2.imshow("failure_detect", view)
            # debug rgb_gate
            if img_crop_for_vis is not None:
                cv2.imshow("RGB gate input (Crop)", img_crop_for_vis)
            ##
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('o'):
                overlay_on = not overlay_on
            elif key == ord(']'):
                thr = min(0.99, thr+0.02); print(f"[thr] {thr:.2f}")
            elif key == ord('['):
                thr = max(0.01, thr-0.02); print(f"[thr] {thr:.2f}")
        
        
        if center is not None and mask is not None:
            # 점수 계산 (거리변환 기반 가중 포함)
            try:
                dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
            except:
                dt = None
            score = candidate_score(mask, dt=dt)

            if score > best["score"]:
                best["score"]  = score
                best["color"]  = color_bgr.copy()
                best["mask"]   = mask.copy()
                best["center"] = (float(center[0]), float(center[1]))
                best["theta"]  = float(theta) if theta is not None else 0.0
                best["xyz"]    = tuple(map(float, xyz)) if xyz is not None else None
            # success, centers(리스트), ... 가 결정된 시점
            if args.save_best_only:
                ok = (not args.save_on_success_only)
                if ok and best["color"] is not None:
                    save_debug_bundle(
                        color_bgr=best["color"],
                        mask_u8=best["mask"],
                        center=best["center"],
                        theta=best["theta"],
                        xyz=best["xyz"],
                        out_dir=args.save_dir,
                        prefix=args.prefix,
                        tag="best"
                    )

        # 종료 조건: 성공하면 바로 리턴
        if success_out is not None:
            if args.json:
                print(json.dumps(success_out))
                # break # debug rgb_gate
            else:
                print("[OK] detection:", success_out)

                # debug rgb_gate
                # success_out = None # debug rgb_gate
                # if args.vote:
                #     votes.clear()
            if args.show:
                print("[INFO] Showing result for 3 seconds...")
                cv2.waitKey(3000)
            break # debug rgb_gate

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
