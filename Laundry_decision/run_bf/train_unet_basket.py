
"""

python /home/cheon/Documents/train_all/train_unet_multi.py \
  --root_base /home/cheon/Documents/train_all/dataset_rgb_sam2 \
  --domains failure,basket \
  --device cuda --size 384 --batch 6 --epochs 60 \
  --zmax 0.6 --patience 10 --seed 42 --base 32 --use_se \
  --use_rgb \
  --bce_w 0.7 --dice_w 0.3 --thr 0.5 --scheduler plateau \
  --ckpt unet_best_multi.pt

데이터 구조(도메인별):
<root_base>/<domain>/samples/<sample_id>/{diff.npy, cur.npy, mask.png[, rgb.png]}

- diff.npy : float32, 미터 단위 차(절대값), 학습 시 0~0.1 클립 후 /0.1 정규화
- cur.npy  : float32, 현재 depth(m), 학습 시 0~zmax 클립 후 /zmax 정규화
- mask.png : 0/255 바이너리 마스크
- rgb.png  : (선택) BGR/PNG; 없으면 학습 시 검은 RGB로 대체
"""

import os, argparse, csv, json, time, random
from glob import glob
import numpy as np
import cv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------- Repro --------------------
def get_domain_id(sample_dir: str) -> int:
    # 경로에 'basket' / 'failure' 포함 여부로 간단 판별
    p = sample_dir.lower()
    if 'basket' in p: return 0
    if 'failure' in p: return 1
    
    return 0


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed); random.seed(worker_seed)

# -------------------- Blocks --------------------
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

# -------------------- Augment --------------------
def random_channel_shuffle(img_rgb, p=0.3):
    """RGB 채널 순서를 랜덤하게 섞어 색상 불변성 강화."""
    if random.random() > p: return img_rgb
    idx = list(range(3))
    random.shuffle(idx)
    return img_rgb[:, :, idx].copy()

def color_jitter_rgb(img_rgb, b=(0.92,1.08), c=(0.92,1.08), s=(0.9,1.1), h=(-4,4), p=0.5):
    if random.random() > p: return img_rgb
    out = img_rgb.copy()
    alpha = random.uniform(*c); beta = (random.uniform(*b) - 1.0)
    out = np.clip(alpha * out + beta, 0, 1)
    hsv = cv2.cvtColor((out*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[...,1] = np.clip(hsv[...,1] * random.uniform(*s), 0, 255)
    hsv[...,0] = (hsv[...,0] + random.uniform(*h)) % 180
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return out

def random_gamma(img_rgb, g=(0.92,1.08), p=0.3):
    if random.random() > p: return img_rgb
    gamma = random.uniform(*g)
    lut = ((np.linspace(0,1,256)**gamma)*255).astype(np.uint8)
    return cv2.LUT((img_rgb*255).astype(np.uint8), lut).astype(np.float32)/255.0

def random_gaussian_blur(img_rgb, k=(3,5), p=0.2):
    if random.random() > p: return img_rgb
    ks = random.choice(k); ks = ks if ks%2==1 else ks+1
    return cv2.GaussianBlur(img_rgb, (ks, ks), 0)

def add_gaussian_noise(x_np, sigma=0.01, p=0.3):
    if random.random() > p: return x_np
    noise = np.random.normal(0, sigma, size=x_np.shape).astype(np.float32)
    x = np.clip(x_np + noise, 0.0, 1.0)
    return x

def random_depth_holes(x_np, max_rect=1, p=0.2):
    if random.random() > p: return x_np
    C,H,W = x_np.shape
    for _ in range(random.randint(1, max_rect)):
        rh = random.randint(max(2,H//30), max(3,H//16))
        rw = random.randint(max(2,W//30), max(3,W//16))
        y = random.randint(0, H-rh); x = random.randint(0, W-rw)
        x_np[0, y:y+rh, x:x+rw] = 0.0  # diff 채널에 홀
        x_np[1, y:y+rh, x:x+rw] = 0.0  # z 채널에 홀
    return x_np

def random_bilateral_blur_on_z(x_np, p=0.15):
    if random.random() > p: return x_np
    cur = (x_np[1]*255).astype(np.uint8)
    cur = cv2.bilateralFilter(cur, d=5, sigmaColor=20, sigmaSpace=10)
    x_np[1] = (cur.astype(np.float32)/255.0)
    return x_np

def intensity_jitter_on_z(x_np, mult=(0.97,1.03), add=(-0.01,0.01), p=0.3):
    if random.random() > p: return x_np
    a = random.uniform(*mult); b = random.uniform(*add)
    x_np[1] = np.clip(a * x_np[1] + b, 0.0, 1.0)
    return x_np

def z_bias_jitter(x_np, diff_weight=0.5, max_bias=0.01, p=0.15):
    if random.random() > p: return x_np
    dz = random.uniform(-max_bias, max_bias)
    x_np[1] = np.clip(x_np[1] + dz, 0.0, 1.0)
    x_np[0] = np.clip(x_np[0] + dz * diff_weight, 0.0, 1.0)
    return x_np

def mask_boundary_jitter(m_np, k=1, p=0.0):
    if random.random() > p: return m_np
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
    if random.random() < 0.5:
        m_aug = cv2.erode(m_np.astype(np.uint8), kernel, 1)
    else:
        m_aug = cv2.dilate(m_np.astype(np.uint8), kernel, 1)
    return m_aug.astype(np.float32)

def sample_affine_params(H, W, max_deg=3, scale_low=0.96, scale_high=1.04, translate=0.02):
    deg = random.uniform(-max_deg, max_deg)
    sc  = random.uniform(scale_low, scale_high)
    tx  = random.uniform(-translate, translate) * W
    ty  = random.uniform(-translate, translate) * H
    M = cv2.getRotationMatrix2D((W/2, H/2), deg, sc)
    M[:,2] += [tx, ty]
    return M

def warp_multi_channel(x_np, M, inter_linear=True):
    C,H,W = x_np.shape
    flags = cv2.INTER_LINEAR if inter_linear else cv2.INTER_NEAREST
    out = np.stack(
        [cv2.warpAffine(x_np[c], M, (W,H), flags=flags, borderMode=cv2.BORDER_REFLECT_101)
         for c in range(C)], axis=0
    )
    return out

# -------------------- Data --------------------
def imread_mask(p):
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None: raise FileNotFoundError(p)
    return (m>0).astype(np.float32)

def discover_samples(root_base, domains):
    sample_dirs = []
    for dom in domains:
        d = os.path.join(root_base, dom, "samples")
        if not os.path.isdir(d):
            raise SystemExit(f"[ERR] samples 폴더가 없습니다: {d}")
        for s in sorted(os.listdir(d)):
            p = os.path.join(d, s)
            if os.path.isdir(p) and all(os.path.exists(os.path.join(p, f)) for f in ["diff.npy","cur.npy","mask.png"]):
                sample_dirs.append(p)
    return sample_dirs

class LaundrySet(Dataset):
    def __init__(self, sample_dirs, zmax=0.5, train=True, size=None, use_rgb=True, strong_color_aug=False):
        self.items = list(sample_dirs)
        self.zmax = zmax; self.train=train; self.size=size
        self.use_rgb = use_rgb
        self.strong_color_aug = strong_color_aug
        if strong_color_aug:
            # 미학습 색상 일반화: hue 전범위 회전, saturation/brightness 넓은 범위
            self.aug_cfg = {
                'affine_p':0.4, 'noise_p':0.3, 'holes_p':0.2, 'bilateral_p':0.15,
                'intjit_p':0.3, 'zbias_p':0.15, 'boundary_p':0.0,
                'color_p':0.95, 'gamma_p':0.5, 'blur_p':0.2,
                'color_b':(0.6,1.4), 'color_c':(0.7,1.3), 'color_s':(0.3,1.7), 'color_h':(-90,90),
                'channel_shuffle_p':0.3,
            }
        else:
            self.aug_cfg = {
                'affine_p':0.4, 'noise_p':0.3, 'holes_p':0.2, 'bilateral_p':0.15,
                'intjit_p':0.3, 'zbias_p':0.15, 'boundary_p':0.0,
                'color_p':0.5, 'gamma_p':0.3, 'blur_p':0.2,
                'color_b':(0.92,1.08), 'color_c':(0.92,1.08), 'color_s':(0.9,1.1), 'color_h':(-4,4),
                'channel_shuffle_p':0.0,
            }

    def _load(self, d):
        D = np.load(os.path.join(d,'diff.npy')).astype(np.float32)
        C = np.load(os.path.join(d,'cur.npy')).astype(np.float32)
        M = imread_mask(os.path.join(d,'mask.png')).astype(np.float32)
        D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
        RGB = None
        if self.use_rgb:
            p_rgb = os.path.join(d, 'rgb.png')
            if os.path.exists(p_rgb):
                rgb = cv2.imread(p_rgb, cv2.IMREAD_COLOR)
                RGB = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return D, C, M, RGB

    def _resize_or_crop(self, x, m, rgb):
        if not self.size: return x, m, rgb
        H,W = x.shape[1:]; th, tw = self.size
        if self.train:
            if H<th or W<tw:
                x = np.stack([cv2.resize(x[c], (tw,th), interpolation=cv2.INTER_NEAREST) for c in range(x.shape[0])],0)
                m = cv2.resize(m, (tw,th), interpolation=cv2.INTER_NEAREST)
                if rgb is not None: rgb = cv2.resize(rgb, (tw,th), interpolation=cv2.INTER_LINEAR)
            else:
                y = random.randint(0, H-th); x0 = random.randint(0, W-tw)
                x = x[:, y:y+th, x0:x0+tw]; m = m[y:y+th, x0:x0+tw]
                if rgb is not None: rgb = rgb[y:y+th, x0:x0+tw]
        else:
            if H!=th or W!=tw:
                x = np.stack([cv2.resize(x[c], (tw,th), interpolation=cv2.INTER_NEAREST) for c in range(x.shape[0])],0)
                m = cv2.resize(m, (tw,th), interpolation=cv2.INTER_NEAREST)
                if rgb is not None: rgb = cv2.resize(rgb, (tw,th), interpolation=cv2.INTER_LINEAR)
        return x, m, rgb

    def _aug(self, x, m, rgb):
        if random.random()<0.5:
            x = x[:,:,::-1].copy(); m = m[:,::-1].copy()
            if rgb is not None: rgb = rgb[:,::-1,:].copy()

        if random.random() < self.aug_cfg['affine_p']:
            H,W = x.shape[1:]
            M = sample_affine_params(H,W, max_deg=3, scale_low=0.96, scale_high=1.04, translate=0.02)
            x = warp_multi_channel(x, M, inter_linear=True)
            m = cv2.warpAffine(m, M, (W,H), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
            if rgb is not None:
                rgb = cv2.warpAffine(rgb, M, (W,H), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)

        x = add_gaussian_noise(x, sigma=0.01, p=self.aug_cfg['noise_p'])
        x = random_depth_holes(x, max_rect=1, p=self.aug_cfg['holes_p'])
        x = random_bilateral_blur_on_z(x, p=self.aug_cfg['bilateral_p'])
        x = intensity_jitter_on_z(x, mult=(0.97,1.03), add=(-0.01,0.01), p=self.aug_cfg['intjit_p'])
        x = z_bias_jitter(x, diff_weight=0.5, max_bias=0.01, p=self.aug_cfg['zbias_p'])

        if self.aug_cfg['boundary_p']>0:
            m = mask_boundary_jitter(m, k=1, p=self.aug_cfg['boundary_p'])

        if rgb is not None:
            rgb = color_jitter_rgb(rgb,
                                   b=self.aug_cfg['color_b'], c=self.aug_cfg['color_c'],
                                   s=self.aug_cfg['color_s'], h=self.aug_cfg['color_h'],
                                   p=self.aug_cfg['color_p'])
            rgb = random_gamma(rgb, g=(0.92,1.08), p=self.aug_cfg['gamma_p'])
            rgb = random_gaussian_blur(rgb, k=(3,5), p=self.aug_cfg['blur_p'])
            rgb = random_channel_shuffle(rgb, p=self.aug_cfg['channel_shuffle_p'])

        x = np.clip(x, 0.0, 1.0)
        return x, m, rgb

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        dpath = self.items[i]  # 샘플 폴더 경로
        D, C, M, RGB = self._load(dpath)

        # Normalize (그대로)
        D = np.clip(D, 0, 0.1)/0.1
        C = np.clip(C, 0, self.zmax)/max(1e-6, self.zmax)
        X = np.stack([D, C], axis=0)  # (2,H,W)

        X, M, RGB = self._resize_or_crop(X, M, RGB)
        if self.train:
            X, M, RGB = self._aug(X, M, RGB)

        if RGB is None:
            RGB = np.zeros((X.shape[1], X.shape[2], 3), dtype=np.float32)
        X = np.concatenate([X, np.transpose(RGB, (2,0,1))], axis=0)  # (5,H,W)

        # --- 도메인 채널 (0=basket, 1=failure) ---
        dom_id = get_domain_id(dpath)
        dom_plane = np.full((1, X.shape[1], X.shape[2]), float(dom_id), dtype=np.float32)
        X = np.concatenate([X, dom_plane], axis=0)  # (6,H,W)

        return torch.from_numpy(X), torch.from_numpy(M[None,...])

# -------------------- Loss / Metric --------------------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6): super().__init__(); self.eps=eps
    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        num = 2*(p*target).sum(dim=(1,2,3)) + self.eps
        den = p.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + self.eps
        return 1 - (num/den).mean()

@torch.no_grad()
def compute_iou(logits, target, thr=0.5):
    p = (torch.sigmoid(logits)>thr).float()
    inter = (p*target).sum(dim=(1,2,3))
    union = p.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - inter + 1e-6
    return (inter/union).mean().item()

# -------------------- Train --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_base', required=True, help='dataset_rgb_sam2 루트 (도메인 하위에 samples 폴더)')
    ap.add_argument('--domains', default='failure,basket', help='쉼표로 구분')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--zmax', type=float, default=0.5)
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--device', default=None)
    ap.add_argument('--ckpt', default='unet_best.pt')
    ap.add_argument('--logdir', default='_logs')
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--val_ratio', type=float, default=0.2)

    ap.add_argument('--use_rgb', action='store_true')
    ap.add_argument('--base', type=int, default=32)
    ap.add_argument('--use_se', action='store_true')
    ap.add_argument('--bce_w', type=float, default=0.7)
    ap.add_argument('--dice_w', type=float, default=0.3)
    ap.add_argument('--pos_weight', type=float, default=0.0, help='>0 to enable BCE pos_weight')
    ap.add_argument('--thr', type=float, default=0.5, help='IoU threshold')
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--clip', type=float, default=1.0, help='grad clip max-norm (<=0 to disable)')
    ap.add_argument('--scheduler', choices=['cosine','plateau'], default='plateau')
    ap.add_argument('--strong_color_aug', action='store_true',
                    help='hue 전범위+채널셔플 강한 색상 aug (미학습 색상 일반화)')

    args = ap.parse_args()
    seed_everything(args.seed)

    dev = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    size = (args.size, args.size)
    pin_mem = (dev.type == 'cuda')

    domains = [d.strip() for d in args.domains.split(',') if d.strip()]
    all_samples = discover_samples(args.root_base, domains)
    if len(all_samples) < 10:
        raise SystemExit(f"[ERR] 샘플이 너무 적습니다: {len(all_samples)}")

    random.Random(args.seed).shuffle(all_samples)
    n_val = max(1, int(len(all_samples) * args.val_ratio))
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    tr = LaundrySet(train_samples, zmax=args.zmax, train=True,  size=size, use_rgb=args.use_rgb, strong_color_aug=args.strong_color_aug)
    va = LaundrySet(val_samples,   zmax=args.zmax, train=False, size=size, use_rgb=args.use_rgb)

    g = torch.Generator(); g.manual_seed(args.seed)
    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True,
                       num_workers=4, pin_memory=pin_mem,
                       worker_init_fn=seed_worker, generator=g, drop_last=False)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False,
                       num_workers=2, pin_memory=pin_mem,
                       worker_init_fn=seed_worker, generator=g, drop_last=False)

    c_in = (5 if args.use_rgb else 2) + 1
    net = TinyUNet(c_in=c_in, base=args.base, use_se=args.use_se).to(dev)

    pos_w = None
    if args.pos_weight and args.pos_weight > 0:
        pos_w = torch.tensor(args.pos_weight, device=dev)

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    dice = DiceLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cosine':
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        plateau = False
    else:
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='max', factor=0.5, patience=4
        )
        plateau = True

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    # logging
    log_root = os.path.join(args.root_base, args.logdir)
    os.makedirs(log_root, exist_ok=True)
    csv_path = os.path.join(log_root, 'train_log.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch','train_loss','val_loss','val_iou','lr',
                'tr_bce','tr_dice','va_bce','va_dice'
            ])
    val_iou_curve = []

    best_iou, best_ep, noimp = 0.0, 0, 0
    t0 = time.time()

    for ep in range(1, args.epochs+1):
        # curriculum
        if hasattr(tr, 'aug_cfg'):
            if ep <= 10:
                tr.aug_cfg.update({'affine_p':0.2,'noise_p':0.2,'holes_p':0.1,'bilateral_p':0.1,
                                   'intjit_p':0.2,'zbias_p':0.1,'boundary_p':0.0,
                                   'color_p':0.3,'gamma_p':0.2,'blur_p':0.1})
            elif ep <= 30:
                tr.aug_cfg.update({'affine_p':0.4,'noise_p':0.3,'holes_p':0.2,'bilateral_p':0.15,
                                   'intjit_p':0.3,'zbias_p':0.15,'boundary_p':0.1,
                                   'color_p':0.5,'gamma_p':0.3,'blur_p':0.2})
            else:
                tr.aug_cfg.update({'affine_p':0.5,'noise_p':0.35,'holes_p':0.25,'bilateral_p':0.2,
                                   'intjit_p':0.35,'zbias_p':0.2,'boundary_p':0.15,
                                   'color_p':0.6,'gamma_p':0.35,'blur_p':0.25})

        # train
        net.train(); loss_tr=0.0; tr_bce=0.0; tr_dice=0.0
        for x, y in dl_tr:
            x=x.to(dev, non_blocking=True).float(); y=y.to(dev, non_blocking=True).float()
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                print("[WARN] NaN/Inf in batch. Skip."); continue

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                z = net(x)
                bce_l  = bce(z, y)
                dice_l = dice(z, y)
                loss = args.bce_w * bce_l + args.dice_w * dice_l

            if not torch.isfinite(z).all():
                print("[WARN] NaN in logits. Skip."); continue

            scaler.scale(loss).backward()
            if args.clip and args.clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            scaler.step(opt); scaler.update()

            bs = x.size(0)
            loss_tr += loss.item()*bs
            tr_bce  += bce_l.item()*bs
            tr_dice += dice_l.item()*bs

        ntr = max(1,len(tr))
        loss_tr /= ntr; tr_bce/=ntr; tr_dice/=ntr

        # val
        net.eval(); loss_va=0.0; iou=0.0; va_bce=0.0; va_dice=0.0
        with torch.no_grad():
            for x, y in dl_va:
                x=x.to(dev).float(); y=y.to(dev).float()
                z = net(x)
                bce_l  = bce(z, y)
                dice_l = dice(z, y)
                loss = args.bce_w * bce_l + args.dice_w * dice_l
                bs = x.size(0)
                loss_va += loss.item()*bs
                va_bce  += bce_l.item()*bs
                va_dice += dice_l.item()*bs
                iou     += compute_iou(z, y, thr=args.thr)*bs

        nva = max(1,len(va))
        loss_va/=nva; va_bce/=nva; va_dice/=nva; iou/=nva
        val_iou_curve.append(iou)

        # scheduler
        if plateau:
            sch.step(iou); cur_lr = opt.param_groups[0]['lr']
        else:
            sch.step(); cur_lr = sch.get_last_lr()[0]

        print(f"[{ep:03d}] train {loss_tr:.4f} (bce {tr_bce:.4f}, dice {tr_dice:.4f}) | "
              f"val {loss_va:.4f} (bce {va_bce:.4f}, dice {va_dice:.4f}) | IoU {iou:.4f} | lr {cur_lr:.2e}")

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                ep, f"{loss_tr:.6f}", f"{loss_va:.6f}", f"{iou:.6f}", f"{cur_lr:.6e}",
                f"{tr_bce:.6f}", f"{tr_dice:.6f}", f"{va_bce:.6f}", f"{va_dice:.6f}"
            ])

        # save best
        if iou > best_iou:
            best_iou, best_ep, noimp = iou, ep, 0
            save_path = os.path.join(args.root_base, args.ckpt)
            torch.save({'model':net.state_dict(),'iou':best_iou,'ep':best_ep,
                        'c_in':c_in,'base':args.base,'use_se':args.use_se},
                       save_path)
            json.dump({'best_ep':best_ep,'best_iou':float(best_iou)},
                      open(os.path.join(log_root,'best.json'),'w'), indent=2)
            print(f"  ↳ saved best: IoU={best_iou:.4f} → {save_path}")
        else:
            noimp += 1
            if args.patience>0 and noimp >= args.patience:
                print(f"Early stop @ epoch {ep} (best IoU {best_iou:.4f} @ {best_ep})")
                break

    # save curves & summary
    np.save(os.path.join(log_root,'val_iou.npy'), np.array(val_iou_curve, dtype=np.float32))
    json.dump({
        'elapsed_sec': round(time.time()-t0,2),
        'last_epoch': ep, 'best_ep': best_ep, 'best_iou': float(best_iou),
        'args': vars(args)
    }, open(os.path.join(log_root,'summary.json'),'w'), indent=2)

if __name__ == '__main__':
    main()