"""
TinyUNet 논문용 평가 스크립트
==============================
기존 train_unet_basket.py와 동일한 아키텍처/하이퍼파라미터를 사용하되,
데이터를 train / val / test 3-way로 분리하여 최종 test 성능을 측정.

- train: 모델 학습
- val  : best model 선택 (early stopping)
- test : val로 선택된 best model로 최종 1회 평가 (학습에 미사용)
- 도메인별(basket / failure) 개별 IoU / Dice 리포트

Usage:
  python train_unet_paper_eval.py \
    --root_base dataset_rgb_sam2 \
    --domains failure,basket \
    --device cuda --size 384 --batch 6 --epochs 60 \
    --zmax 0.5 --patience 10 --seed 42 --base 32 --use_se --use_rgb \
    --bce_w 0.7 --dice_w 0.3 --thr 0.5 --scheduler plateau \
    --val_ratio 0.15 --test_ratio 0.15 \
    --ckpt unet_paper.pt --logdir _logs_paper
"""

import os, argparse, csv, json, time, random
from glob import glob
from collections import defaultdict
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ────────────────────────────────────────────
#  재현성
# ────────────────────────────────────────────
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

# ────────────────────────────────────────────
#  모델 (train_unet_basket.py와 동일)
# ────────────────────────────────────────────
class SE(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, max(1, ch // r), bias=False), nn.ReLU(True),
            nn.Linear(max(1, ch // r), ch, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(self.avg(x).view(b, c)).view(b, c, 1, 1)
        return x * w

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
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
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, use_se)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dh = x2.size(2) - x1.size(2); dw = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, [dw//2, dw-dw//2, dh//2, dh-dh//2])
        return self.conv(torch.cat([x2, x1], dim=1))

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

# ────────────────────────────────────────────
#  Augmentation (train_unet_basket.py와 동일)
# ────────────────────────────────────────────
def color_jitter_rgb(img, b=(0.92,1.08), c=(0.92,1.08), s=(0.9,1.1), h=(-4,4), p=0.5):
    if random.random() > p: return img
    out = img.copy()
    alpha = random.uniform(*c); beta = random.uniform(*b) - 1.0
    out = np.clip(alpha * out + beta, 0, 1)
    hsv = cv2.cvtColor((out*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[...,1] = np.clip(hsv[...,1]*random.uniform(*s), 0, 255)
    hsv[...,0] = (hsv[...,0] + random.uniform(*h)) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

def random_gamma(img, g=(0.92,1.08), p=0.3):
    if random.random() > p: return img
    lut = ((np.linspace(0,1,256)**random.uniform(*g))*255).astype(np.uint8)
    return cv2.LUT((img*255).astype(np.uint8), lut).astype(np.float32) / 255.0

def random_gaussian_blur(img, k=(3,5), p=0.2):
    if random.random() > p: return img
    ks = random.choice(k); ks = ks if ks%2==1 else ks+1
    return cv2.GaussianBlur(img, (ks, ks), 0)

def add_gaussian_noise(x, sigma=0.01, p=0.3):
    if random.random() > p: return x
    return np.clip(x + np.random.normal(0, sigma, x.shape).astype(np.float32), 0, 1)

def random_depth_holes(x, max_rect=1, p=0.2):
    if random.random() > p: return x
    C, H, W = x.shape
    for _ in range(random.randint(1, max_rect)):
        rh = random.randint(max(2,H//30), max(3,H//16))
        rw = random.randint(max(2,W//30), max(3,W//16))
        y0 = random.randint(0, H-rh); x0 = random.randint(0, W-rw)
        x[0, y0:y0+rh, x0:x0+rw] = 0.0
        x[1, y0:y0+rh, x0:x0+rw] = 0.0
    return x

def random_bilateral_blur_on_z(x, p=0.15):
    if random.random() > p: return x
    cur = (x[1]*255).astype(np.uint8)
    x[1] = cv2.bilateralFilter(cur, 5, 20, 10).astype(np.float32) / 255.0
    return x

def intensity_jitter_on_z(x, mult=(0.97,1.03), add=(-0.01,0.01), p=0.3):
    if random.random() > p: return x
    x[1] = np.clip(random.uniform(*mult)*x[1] + random.uniform(*add), 0, 1)
    return x

def z_bias_jitter(x, diff_weight=0.5, max_bias=0.01, p=0.15):
    if random.random() > p: return x
    dz = random.uniform(-max_bias, max_bias)
    x[1] = np.clip(x[1]+dz, 0, 1)
    x[0] = np.clip(x[0]+dz*diff_weight, 0, 1)
    return x

def sample_affine_params(H, W, max_deg=3, scale_low=0.96, scale_high=1.04, translate=0.02):
    deg = random.uniform(-max_deg, max_deg); sc = random.uniform(scale_low, scale_high)
    tx = random.uniform(-translate, translate)*W; ty = random.uniform(-translate, translate)*H
    M = cv2.getRotationMatrix2D((W/2, H/2), deg, sc); M[:,2] += [tx, ty]
    return M

def warp_multi(x, M):
    C, H, W = x.shape
    return np.stack([cv2.warpAffine(x[c], M, (W,H), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT_101) for c in range(C)], 0)

# ────────────────────────────────────────────
#  데이터셋
# ────────────────────────────────────────────
def get_domain(path: str) -> str:
    p = path.lower()
    if 'basket' in p: return 'basket'
    if 'failure' in p: return 'failure'
    return 'unknown'

def discover_samples(root_base, domains):
    samples = []
    for dom in domains:
        d = os.path.join(root_base, dom, "samples")
        if not os.path.isdir(d):
            raise SystemExit(f"[ERR] 폴더 없음: {d}")
        for s in sorted(os.listdir(d)):
            p = os.path.join(d, s)
            if os.path.isdir(p) and all(os.path.exists(os.path.join(p, f))
                                         for f in ["diff.npy", "cur.npy", "mask.png"]):
                samples.append(p)
    return samples

def stratified_split(samples, val_ratio, test_ratio, seed):
    """도메인별 비율을 유지하며 train/val/test 분리."""
    by_domain = defaultdict(list)
    for s in samples:
        by_domain[get_domain(s)].append(s)

    train_s, val_s, test_s = [], [], []
    rng = random.Random(seed)
    for dom, slist in sorted(by_domain.items()):
        rng.shuffle(slist)
        n = len(slist)
        n_test = max(1, round(n * test_ratio))
        n_val  = max(1, round(n * val_ratio))
        n_train = n - n_val - n_test
        if n_train < 1:
            raise SystemExit(f"[ERR] {dom} 샘플 수가 너무 적습니다: {n}")
        test_s  += slist[:n_test]
        val_s   += slist[n_test:n_test+n_val]
        train_s += slist[n_test+n_val:]
        print(f"  [{dom}] total={n}  train={n_train}  val={n_val}  test={n_test}")
    return train_s, val_s, test_s

class LaundrySet(Dataset):
    def __init__(self, sample_dirs, zmax=0.5, train=True, size=None, use_rgb=True):
        self.items = list(sample_dirs)
        self.zmax = zmax; self.train = train; self.size = size; self.use_rgb = use_rgb
        self.aug_cfg = dict(affine_p=0.4, noise_p=0.3, holes_p=0.2, bilateral_p=0.15,
                            intjit_p=0.3, zbias_p=0.15, color_p=0.5, gamma_p=0.3, blur_p=0.2)

    def _load(self, d):
        D = np.nan_to_num(np.load(os.path.join(d,'diff.npy')).astype(np.float32))
        C = np.nan_to_num(np.load(os.path.join(d,'cur.npy')).astype(np.float32))
        M = (cv2.imread(os.path.join(d,'mask.png'), cv2.IMREAD_GRAYSCALE) > 0).astype(np.float32)
        RGB = None
        if self.use_rgb:
            p_rgb = os.path.join(d, 'rgb.png')
            if os.path.exists(p_rgb):
                RGB = cv2.cvtColor(cv2.imread(p_rgb), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return D, C, M, RGB

    def _resize(self, x, m, rgb, H, W):
        th, tw = self.size
        if self.train and (H >= th and W >= tw):
            y0 = random.randint(0, H-th); x0 = random.randint(0, W-tw)
            x = x[:, y0:y0+th, x0:x0+tw]; m = m[y0:y0+th, x0:x0+tw]
            if rgb is not None: rgb = rgb[y0:y0+th, x0:x0+tw]
        else:
            x = np.stack([cv2.resize(x[c],(tw,th),interpolation=cv2.INTER_NEAREST) for c in range(x.shape[0])],0)
            m = cv2.resize(m, (tw,th), interpolation=cv2.INTER_NEAREST)
            if rgb is not None: rgb = cv2.resize(rgb, (tw,th), interpolation=cv2.INTER_LINEAR)
        return x, m, rgb

    def _aug(self, x, m, rgb):
        if random.random() < 0.5:
            x = x[:,:,::-1].copy(); m = m[:,::-1].copy()
            if rgb is not None: rgb = rgb[:,::-1,:].copy()
        if random.random() < self.aug_cfg['affine_p']:
            H, W = x.shape[1:]
            M = sample_affine_params(H, W)
            x = warp_multi(x, M)
            m = cv2.warpAffine(m, M, (W,H), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
            if rgb is not None:
                rgb = cv2.warpAffine(rgb, M, (W,H), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)
        x = add_gaussian_noise(x, p=self.aug_cfg['noise_p'])
        x = random_depth_holes(x, p=self.aug_cfg['holes_p'])
        x = random_bilateral_blur_on_z(x, p=self.aug_cfg['bilateral_p'])
        x = intensity_jitter_on_z(x, p=self.aug_cfg['intjit_p'])
        x = z_bias_jitter(x, p=self.aug_cfg['zbias_p'])
        if rgb is not None:
            rgb = color_jitter_rgb(rgb, p=self.aug_cfg['color_p'])
            rgb = random_gamma(rgb, p=self.aug_cfg['gamma_p'])
            rgb = random_gaussian_blur(rgb, p=self.aug_cfg['blur_p'])
        return np.clip(x, 0, 1), m, rgb

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        d = self.items[i]
        D, C, M, RGB = self._load(d)
        D = np.clip(D, 0, 0.1) / 0.1
        C = np.clip(C, 0, self.zmax) / max(1e-6, self.zmax)
        x = np.stack([D, C], axis=0)

        if self.size:
            x, M, RGB = self._resize(x, M, RGB, x.shape[1], x.shape[2])
        if self.train:
            x, M, RGB = self._aug(x, M, RGB)

        if RGB is None:
            RGB = np.zeros((x.shape[1], x.shape[2], 3), dtype=np.float32)
        x = np.concatenate([x, np.transpose(RGB, (2,0,1))], axis=0)  # (5,H,W)

        dom_id = 0 if get_domain(d) == 'basket' else 1
        dom_plane = np.full((1, x.shape[1], x.shape[2]), float(dom_id), dtype=np.float32)
        x = np.concatenate([x, dom_plane], axis=0)  # (6,H,W)

        return torch.from_numpy(x), torch.from_numpy(M[None,...]), get_domain(d)

def collate_fn(batch):
    xs, ys, doms = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(doms)

# ────────────────────────────────────────────
#  Loss / Metric
# ────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6): super().__init__(); self.eps = eps
    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        num = 2*(p*target).sum(dim=(1,2,3)) + self.eps
        den = p.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + self.eps
        return 1 - (num/den).mean()

@torch.no_grad()
def compute_metrics(logits, target, thr=0.5):
    """IoU와 Dice를 샘플별로 계산하여 평균 반환."""
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * target).sum(dim=(1,2,3))
    union = p.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - inter + 1e-6
    iou  = (inter / union).mean().item()
    dice = (2 * inter / (p.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + 1e-6)).mean().item()
    return iou, dice

# ────────────────────────────────────────────
#  평가 루프 (domain별 분리)
# ────────────────────────────────────────────
@torch.no_grad()
def evaluate(net, loader, bce_fn, dice_fn, thr, dev):
    net.eval()
    metrics = defaultdict(lambda: dict(iou=0.0, dice=0.0, n=0, loss=0.0))

    for x, y, doms in loader:
        x = x.to(dev).float(); y = y.to(dev).float()
        z = net(x)
        loss = (0.7 * bce_fn(z, y) + 0.3 * dice_fn(z, y)).item()

        for idx, dom in enumerate(doms):
            iou, dice = compute_metrics(z[idx:idx+1], y[idx:idx+1], thr)
            metrics[dom]['iou']  += iou
            metrics[dom]['dice'] += dice
            metrics[dom]['n']    += 1
            metrics[dom]['loss'] += loss

    result = {}
    all_iou, all_dice, all_n = 0.0, 0.0, 0
    for dom, m in metrics.items():
        n = max(1, m['n'])
        result[dom] = dict(iou=m['iou']/n, dice=m['dice']/n, n=m['n'])
        all_iou += m['iou']; all_dice += m['dice']; all_n += m['n']
    result['_all'] = dict(iou=all_iou/max(1,all_n), dice=all_dice/max(1,all_n), n=all_n)
    return result

# ────────────────────────────────────────────
#  main
# ────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_base', default='dataset_rgb_sam2')
    ap.add_argument('--domains',   default='failure,basket')
    ap.add_argument('--epochs',    type=int,   default=60)
    ap.add_argument('--batch',     type=int,   default=6)
    ap.add_argument('--lr',        type=float, default=1e-3)
    ap.add_argument('--zmax',      type=float, default=0.5)
    ap.add_argument('--size',      type=int,   default=384)
    ap.add_argument('--device',    default=None)
    ap.add_argument('--ckpt',      default='unet_paper.pt')
    ap.add_argument('--logdir',    default='_logs_paper')
    ap.add_argument('--patience',  type=int,   default=10)
    ap.add_argument('--seed',      type=int,   default=42)
    ap.add_argument('--val_ratio', type=float, default=0.15)
    ap.add_argument('--test_ratio',type=float, default=0.15)
    ap.add_argument('--use_rgb',   action='store_true')
    ap.add_argument('--base',      type=int,   default=32)
    ap.add_argument('--use_se',    action='store_true')
    ap.add_argument('--bce_w',     type=float, default=0.7)
    ap.add_argument('--dice_w',    type=float, default=0.3)
    ap.add_argument('--thr',       type=float, default=0.5)
    ap.add_argument('--scheduler', choices=['cosine','plateau'], default='plateau')
    ap.add_argument('--clip',      type=float, default=1.0)
    args = ap.parse_args()

    seed_everything(args.seed)
    dev = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    size = (args.size, args.size)
    pin_mem = (dev.type == 'cuda')

    # ── 데이터 분할
    domains = [d.strip() for d in args.domains.split(',') if d.strip()]
    all_samples = discover_samples(args.root_base, domains)
    print(f"\n[DATA] 전체 샘플: {len(all_samples)}개")
    print(f"[SPLIT] val={args.val_ratio:.0%}  test={args.test_ratio:.0%}  "
          f"train={1-args.val_ratio-args.test_ratio:.0%}")
    train_s, val_s, test_s = stratified_split(all_samples, args.val_ratio, args.test_ratio, args.seed)
    print(f"[SPLIT] train={len(train_s)}  val={len(val_s)}  test={len(test_s)}\n")

    tr = LaundrySet(train_s, zmax=args.zmax, train=True,  size=size, use_rgb=args.use_rgb)
    va = LaundrySet(val_s,   zmax=args.zmax, train=False, size=size, use_rgb=args.use_rgb)
    te = LaundrySet(test_s,  zmax=args.zmax, train=False, size=size, use_rgb=args.use_rgb)

    g = torch.Generator(); g.manual_seed(args.seed)
    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=collate_fn,
                       num_workers=4, pin_memory=pin_mem, worker_init_fn=seed_worker, generator=g)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False, collate_fn=collate_fn,
                       num_workers=2, pin_memory=pin_mem)
    dl_te = DataLoader(te, batch_size=args.batch, shuffle=False, collate_fn=collate_fn,
                       num_workers=2, pin_memory=pin_mem)

    # ── 모델
    c_in = (5 if args.use_rgb else 2) + 1
    net = TinyUNet(c_in=c_in, base=args.base, use_se=args.use_se).to(dev)
    bce_fn = nn.BCEWithLogitsLoss()
    dice_fn = DiceLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'plateau':
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=4)
        plateau = True
    else:
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        plateau = False

    # ── 로깅
    log_root = os.path.join(args.root_base, args.logdir)
    os.makedirs(log_root, exist_ok=True)
    csv_path = os.path.join(log_root, 'train_log.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','train_loss','val_iou','val_dice','lr'])

    best_iou, best_ep, noimp = 0.0, 0, 0
    t0 = time.time()

    # ── 학습
    for ep in range(1, args.epochs+1):
        # curriculum augmentation
        if ep <= 10:
            tr.aug_cfg.update(dict(affine_p=0.2, noise_p=0.2, holes_p=0.1, bilateral_p=0.1,
                                   intjit_p=0.2, zbias_p=0.1, color_p=0.3, gamma_p=0.2, blur_p=0.1))
        elif ep <= 30:
            tr.aug_cfg.update(dict(affine_p=0.4, noise_p=0.3, holes_p=0.2, bilateral_p=0.15,
                                   intjit_p=0.3, zbias_p=0.15, color_p=0.5, gamma_p=0.3, blur_p=0.2))
        else:
            tr.aug_cfg.update(dict(affine_p=0.5, noise_p=0.35, holes_p=0.25, bilateral_p=0.2,
                                   intjit_p=0.35, zbias_p=0.2, color_p=0.6, gamma_p=0.35, blur_p=0.25))

        net.train(); loss_tr = 0.0
        for x, y, _ in dl_tr:
            x = x.to(dev).float(); y = y.to(dev).float()
            if not (torch.isfinite(x).all() and torch.isfinite(y).all()): continue
            opt.zero_grad(set_to_none=True)
            z = net(x)
            loss = args.bce_w * bce_fn(z, y) + args.dice_w * dice_fn(z, y)
            if not torch.isfinite(z).all(): continue
            loss.backward()
            if args.clip > 0:
                nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            opt.step()
            loss_tr += loss.item() * x.size(0)
        loss_tr /= max(1, len(tr))

        # validation
        val_m = evaluate(net, dl_va, bce_fn, dice_fn, args.thr, dev)
        val_iou  = val_m['_all']['iou']
        val_dice = val_m['_all']['dice']
        cur_lr = opt.param_groups[0]['lr']

        if plateau: sch.step(val_iou)
        else:       sch.step()

        print(f"[{ep:03d}] train={loss_tr:.4f} | val IoU={val_iou:.4f} Dice={val_dice:.4f} | lr={cur_lr:.2e}")
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep, f"{loss_tr:.6f}", f"{val_iou:.6f}", f"{val_dice:.6f}", f"{cur_lr:.6e}"])

        if val_iou > best_iou:
            best_iou, best_ep, noimp = val_iou, ep, 0
            ckpt_path = os.path.join(args.root_base, args.ckpt)
            torch.save({'model': net.state_dict(), 'iou': best_iou, 'ep': best_ep,
                        'c_in': c_in, 'base': args.base, 'use_se': args.use_se}, ckpt_path)
            print(f"  ↳ best val IoU={best_iou:.4f} @ epoch {best_ep} → {ckpt_path}")
        else:
            noimp += 1
            if args.patience > 0 and noimp >= args.patience:
                print(f"  Early stop @ epoch {ep}"); break

    # ── 최종 Test 평가 (best model 로드)
    print(f"\n{'='*60}")
    print(f"[TEST] Best val IoU={best_iou:.4f} @ epoch {best_ep}")
    print(f"[TEST] Best model 로드 후 test set 평가...")
    ckpt = torch.load(os.path.join(args.root_base, args.ckpt), map_location=dev)
    net.load_state_dict(ckpt['model'])

    test_m = evaluate(net, dl_te, bce_fn, dice_fn, args.thr, dev)

    print(f"\n{'='*60}")
    print(f"  {'Domain':<12} {'N':>4}  {'IoU':>8}  {'Dice':>8}")
    print(f"  {'-'*40}")
    for dom in sorted(k for k in test_m if k != '_all'):
        m = test_m[dom]
        print(f"  {dom:<12} {m['n']:>4}  {m['iou']*100:>7.2f}%  {m['dice']*100:>7.2f}%")
    m = test_m['_all']
    print(f"  {'─'*40}")
    print(f"  {'Overall':<12} {m['n']:>4}  {m['iou']*100:>7.2f}%  {m['dice']*100:>7.2f}%")
    print(f"{'='*60}\n")

    # ── 결과 저장
    result = {
        'split': {'train': len(train_s), 'val': len(val_s), 'test': len(test_s)},
        'best_val': {'epoch': best_ep, 'iou': float(best_iou)},
        'test': {dom: {k: float(v) if isinstance(v, float) else v
                       for k, v in m.items()}
                 for dom, m in test_m.items()},
        'elapsed_sec': round(time.time()-t0, 2),
        'args': vars(args),
    }
    result_path = os.path.join(log_root, 'paper_results.json')
    json.dump(result, open(result_path, 'w'), indent=2)
    print(f"[SAVE] 결과 저장: {result_path}")

if __name__ == '__main__':
    main()
