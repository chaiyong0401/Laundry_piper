'''
python /home/cheon/Documents/run_failure/train_unet_rgb.py \
  --root /home/cheon/Documents/run_failure/dataset_rgb_sam2_1 \
  --device cuda --size 384 --batch 6 --epochs 60 \
  --zmax 0.5 \
  --patience 10 --seed 42 --base 32 --use_se \
  --bce_w 0.7 --dice_w 0.3 --thr 0.5 --scheduler plateau \
  --feat2 z --use_rgb \
  --rgb_dir /home/cheon/Documents/garment_decision/init/roi/dataset/failure/failure/rgb \
  --rgb_suffix _rgb.png

'''

import os, argparse, csv, json, time, random, re
import numpy as np
import cv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------- Repro --------------------
def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# -------------------- Small Blocks --------------------
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
    def __init__(self, c_in=2, base=32, use_se=True):
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

# -------------------- Helpers --------------------
def imread_mask(p):
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None: raise FileNotFoundError(p)
    return (m>0).astype(np.float32)

def load_intrinsics(path):
    J = json.load(open(path, 'r'))
    fx = float(J.get('fx') or J.get('intrinsics',{}).get('fx'))
    fy = float(J.get('fy') or J.get('intrinsics',{}).get('fy'))
    ppx = float(J.get('ppx') or J.get('cx') or J.get('intrinsics',{}).get('ppx'))
    ppy = float(J.get('ppy') or J.get('cy') or J.get('intrinsics',{}).get('ppy'))
    return fx, fy, ppx, ppy

def depth_to_height(depth_m, fx, fy, ppx, ppy, plane):
    H, W = depth_m.shape
    us = np.arange(W)[None,:].repeat(H,0).astype(np.float32)
    vs = np.arange(H)[:,None].repeat(W,1).astype(np.float32)
    Z = depth_m.astype(np.float32)
    X = (us - ppx) * Z / fx
    Y = (vs - ppy) * Z / fy
    n = np.array(plane['n'], dtype=np.float32).reshape(1,1,3)
    d = float(plane['d'])
    XYZ = np.stack([X,Y,Z], axis=-1)
    h = -(XYZ * n).sum(axis=-1) - d
    return h.astype(np.float32)

def extract_id_from_path(path):
    b = os.path.basename(os.path.normpath(path))
    nums = re.findall(r'\d+', b)
    if nums: return nums[-1]
    return None

# -------------------- Dataset --------------------
class LaundrySet(Dataset):
    def __init__(self, list_txt, root_dir=None, zmax=0.5, train=True, size=None,
                 feat2='z', hmax=0.30, q_lo=5.0, q_hi=95.0,
                 use_rgb=False, rgb_name='rgb.png',
                 rgb_dir=None, rgb_suffix='_rgb.png', rgb_index_mode='basename',
                 intr_json='/home/cheon/Documents/run_failure/intrinsics_depth_640x480.json',
                 plane_json='/home/cheon/Documents/run_failure/plane.json'):
        self.items = [ln.strip() for ln in open(list_txt).read().splitlines() if ln.strip()]
        self.root_dir = root_dir
        self.zmax = float(zmax); self.train=train; self.size=size
        self.use_rgb = use_rgb or (rgb_dir is not None)
        self.rgb_name = rgb_name
        self.rgb_dir = rgb_dir
        self.rgb_suffix = rgb_suffix
        self.rgb_index_mode = rgb_index_mode

        self.feat2 = feat2; self.hmax=float(hmax); self.q_lo=float(q_lo); self.q_hi=float(q_hi)

        self.fx=self.fy=self.ppx=self.ppy=None
        self.plane=None
        if self.feat2.startswith('plane'):
            assert intr_json is not None and plane_json is not None, "feat2=plane_* requires intr/plane"
            self.fx, self.fy, self.ppx, self.ppy = load_intrinsics(intr_json)
            self.plane = json.load(open(plane_json,'r'))

        self.aug_cfg = {
            'affine_p':0.4, 'noise_p':0.3, 'holes_p':0.2, 'bilateral_p':0.15,
            'intjit_p':0.3, 'zbias_p':0.15, 'boundary_p':0.0,
            'color_p':0.5, 'gamma_p':0.3, 'blur_p':0.2
        }

    def _resolve(self, path):
        if self.root_dir and not os.path.isabs(path):
            return os.path.normpath(os.path.join(self.root_dir, path))
        return path

    def _load_rgb_external(self, idx, item_dir):
        if self.rgb_dir is None: return None
        if self.rgb_index_mode == 'basename':
            sid = extract_id_from_path(item_dir)
            if sid is None: return None
        else:
            sid = str(idx)
        fn = os.path.join(self.rgb_dir, f"{sid}{self.rgb_suffix}")
        rgb = cv2.imread(fn, cv2.IMREAD_COLOR)
        if rgb is None: return None
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

    def _load(self, idx):
        d = self._resolve(self.items[idx])
        D = np.load(os.path.join(d,'diff.npy')).astype(np.float32)
        C = np.load(os.path.join(d,'cur.npy')).astype(np.float32)
        M = imread_mask(os.path.join(d,'mask.png')).astype(np.float32)
        D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

        RGB = None
        if self.use_rgb:
            RGB = self._load_rgb_external(idx, d)
            if RGB is None:
                p_rgb = os.path.join(d, self.rgb_name)
                rgb = cv2.imread(p_rgb, cv2.IMREAD_COLOR)
                if rgb is not None:
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
            M2 = cv2.getRotationMatrix2D((W/2, H/2), random.uniform(-3,3), random.uniform(0.96,1.04))
            M2[:,2] += [random.uniform(-0.02,0.02)*W, random.uniform(-0.02,0.02)*H]
            x = np.stack([cv2.warpAffine(x[c], M2, (W,H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) for c in range(x.shape[0])], axis=0)
            m = cv2.warpAffine(m, M2, (W,H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
            if rgb is not None:
                rgb = cv2.warpAffine(rgb, M2, (W,H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        if random.random()<0.3:
            x = np.clip(x + np.random.normal(0, 0.01, size=x.shape).astype(np.float32), 0.0, 1.0)
        if random.random()<0.2:
            C,H,W = x.shape
            rh, rw = random.randint(max(2,H//30), max(3,H//16)), random.randint(max(2,W//30), max(3,W//16))
            y = random.randint(0, H-rh); x0 = random.randint(0, W-rw)
            x[1, y:y+rh, x0:x0+rw] = 0.0
        if random.random()<0.15:
            cur = (x[1]*255).astype(np.uint8)
            cur = cv2.bilateralFilter(cur, d=5, sigmaColor=20, sigmaSpace=10)
            x[1] = (cur.astype(np.float32)/255.0)
        if random.random()<0.3:
            a = random.uniform(0.97,1.03); b = random.uniform(-0.01,0.01)
            x[1] = np.clip(a * x[1] + b, 0.0, 1.0)

        if rgb is not None:
            if random.random()<0.5:
                out = rgb.copy()
                alpha = random.uniform(0.92,1.08); beta = (random.uniform(0.92,1.08) - 1.0)
                out = np.clip(alpha * out + beta, 0, 1)
                hsv = cv2.cvtColor((out*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[...,1] = np.clip(hsv[...,1] * random.uniform(0.9,1.1), 0, 255)
                hsv[...,0] = (hsv[...,0] + random.uniform(-4,4)) % 180
                rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
            if random.random()<0.3:
                gamma = random.uniform(0.92,1.08)
                lut = ((np.linspace(0,1,256)**gamma)*255).astype(np.uint8)
                rgb = cv2.LUT((rgb*255).astype(np.uint8), lut).astype(np.float32)/255.0
            if random.random()<0.2:
                ks = random.choice([3,5]); ks = ks if ks%2==1 else ks+1
                rgb = cv2.GaussianBlur(rgb, (ks, ks), 0)

        x = np.clip(x, 0.0, 1.0)
        return x, m, rgb

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        D, C, M, RGB = self._load(idx)
        D = np.clip(D, 0, 0.1)/0.1
        Cn = np.clip(C, 0, self.zmax)/max(1e-6, self.zmax)
        X = np.stack([D, Cn], axis=0)

        X, M, RGB = self._resize_or_crop(X, M, RGB)
        if self.train:
            X, M, RGB = self._aug(X, M, RGB)

        if self.feat2 == 'z':
            pass
        else:
            C_m = X[1] * self.zmax
            H = depth_to_height(C_m, self.fx, self.fy, self.ppx, self.ppy, self.plane)
            if self.feat2 == 'plane_fixed':
                Hf = np.clip(H, 0.0, self.hmax) / max(1e-6, self.hmax)
                X[1] = Hf.astype(np.float32)
            else:
                v = (C_m > 0)
                if v.sum() > 0:
                    hm = H[v]
                    p_lo, p_hi = np.percentile(hm, self.q_lo), np.percentile(hm, self.q_hi)
                else:
                    p_lo, p_hi = -0.01, 0.01
                Hn = np.clip((H - p_lo) / max(1e-6, (p_hi-p_lo)), 0, 1).astype(np.float32)
                X[1] = Hn

        if self.use_rgb and RGB is not None:
            if RGB.ndim == 3:
                X = np.concatenate([X, np.transpose(RGB, (2,0,1))], axis=0)

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
    ap.add_argument('--root', required=True, help='dataset root (contains train.txt/val.txt)')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--zmax', type=float, default=0.5)
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--device', default=None)
    ap.add_argument('--ckpt', default='unet_best_min_wo.pt')
    ap.add_argument('--logdir', default='_logs')
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--use_rgb', action='store_true', help='use rgb (auto true if --rgb_dir given)')
    ap.add_argument('--rgb_name', default='rgb.png', help='filename if rgb is inside each sample dir')
    ap.add_argument('--rgb_dir', default=None, help='external RGB directory containing e.g. 0_rgb.png')
    ap.add_argument('--rgb_suffix', default='_rgb.png', help='suffix after numeric id, e.g. _rgb.png')
    ap.add_argument('--rgb_index_mode', choices=['basename','enumerate'], default='basename',
                    help='how to infer numeric id: from sample dir basename or dataset order')

    ap.add_argument('--base', type=int, default=32)
    ap.add_argument('--use_se', action='store_true')
    ap.add_argument('--bce_w', type=float, default=0.7)
    ap.add_argument('--dice_w', type=float, default=0.3)
    ap.add_argument('--pos_weight', type=float, default=0.0, help='>0 to enable BCE pos_weight')
    ap.add_argument('--thr', type=float, default=0.5, help='IoU threshold')
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--clip', type=float, default=1.0, help='grad clip max-norm (<=0 to disable)')
    ap.add_argument('--scheduler', choices=['cosine','plateau'], default='cosine')

    ap.add_argument('--feat2', choices=['z','plane_fixed','plane_q'], default='z')
    ap.add_argument('--hmax', type=float, default=0.30)
    ap.add_argument('--q_lo', type=float, default=5.0)
    ap.add_argument('--q_hi', type=float, default=95.0)

    ap.add_argument('--intr_json', default='/home/cheon/Documents/run_failure/intrinsics_depth_640x480.json')
    ap.add_argument('--plane_json', default='/home/cheon/Documents/run_failure/plane.json')

    args = ap.parse_args()
    seed_everything(args.seed)

    if args.rgb_dir is not None:
        args.use_rgb = True

    if args.feat2.startswith('plane'):
        assert args.intr_json and args.plane_json, "--intr_json/--plane_json required when feat2 uses plane"

    dev = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    size = (args.size, args.size)
    pin_mem = (dev.type == 'cuda')

    tr = LaundrySet(os.path.join(args.root,'train.txt'), root_dir=args.root,
                    zmax=args.zmax, train=True,  size=size,
                    feat2=args.feat2, hmax=args.hmax, q_lo=args.q_lo, q_hi=args.q_hi,
                    use_rgb=args.use_rgb, rgb_name=args.rgb_name,
                    rgb_dir=args.rgb_dir, rgb_suffix=args.rgb_suffix, rgb_index_mode=args.rgb_index_mode,
                    intr_json=args.intr_json, plane_json=args.plane_json)
    va = LaundrySet(os.path.join(args.root,'val.txt'),   root_dir=args.root,
                    zmax=args.zmax, train=False, size=size,
                    feat2=args.feat2, hmax=args.hmax, q_lo=args.q_lo, q_hi=args.q_hi,
                    use_rgb=args.use_rgb, rgb_name=args.rgb_name,
                    rgb_dir=args.rgb_dir, rgb_suffix=args.rgb_suffix, rgb_index_mode=args.rgb_index_mode,
                    intr_json=args.intr_json, plane_json=args.plane_json)

    g = torch.Generator(); g.manual_seed(args.seed)
    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True,
                       num_workers=4, pin_memory=pin_mem,
                       worker_init_fn=seed_worker, generator=g, drop_last=False)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False,
                       num_workers=2, pin_memory=pin_mem,
                       worker_init_fn=seed_worker, generator=g, drop_last=False)

    c_in = 5 if args.use_rgb else 2
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

    log_root = os.path.join(args.root, args.logdir)
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

        net.train(); loss_tr=0.0; tr_bce=0.0; tr_dice=0.0
        for x, y in dl_tr:
            x=x.to(dev, non_blocking=True).float(); y=y.to(dev, non_blocking=True).float()
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                print("[WARN] Found NaN/Inf in input/label batch. Skipped.")
                continue

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                z = net(x)
                bce_l  = bce(z, y)
                dice_l = dice(z, y)
                loss = args.bce_w * bce_l + args.dice_w * dice_l

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

        if plateau:
            sch.step(iou); cur_lr = opt.param_groups[0]['lr']
        else:
            sch.step(); cur_lr = sch.get_last_lr()[0]

        print(f"[{ep:03d}] train {loss_tr:.4f} (bce {tr_bce:.4f}, dice {tr_dice:.4f}) | "
              f"val {loss_va:.4f} (bce {va_bce:.4f}, dice {va_dice:.4f}) | "
              f"IoU {iou:.4f} | lr {cur_lr:.2e}")

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                ep, f"{loss_tr:.6f}", f"{loss_va:.6f}", f"{iou:.6f}", f"{cur_lr:.6e}",
                f"{tr_bce:.6f}", f"{tr_dice:.6f}", f"{va_bce:.6f}", f"{va_dice:.6f}"
            ])

        c_in = 5 if args.use_rgb else 2
        if iou > best_iou:
            best_iou, best_ep, noimp = iou, ep, 0
            torch.save({'model':net.state_dict(),'iou':best_iou,'ep':best_ep,
                        'c_in':c_in,'base':args.base,'use_se':args.use_se,
                        'feat2':args.feat2,'zmax':args.zmax,'hmax':args.hmax,
                        'q_lo':args.q_lo,'q_hi':args.q_hi,'use_rgb':args.use_rgb},
                       os.path.join(args.root, args.ckpt))
            json.dump({'best_ep':best_ep,'best_iou':float(best_iou)},
                      open(os.path.join(log_root,'best.json'),'w'), indent=2)
            print(f"  ↳ saved best: IoU={best_iou:.4f}")
        else:
            noimp += 1
            if args.patience>0 and noimp >= args.patience:
                print(f"Early stop @ epoch {ep} (best IoU {best_iou:.4f} @ {best_ep})")
                break

    np.save(os.path.join(log_root,'val_iou.npy'), np.array(val_iou_curve, dtype=np.float32))
    json.dump({
        'elapsed_sec': round(time.time()-t0,2),
        'last_epoch': ep, 'best_ep': best_ep, 'best_iou': float(best_iou),
        'args': vars(args)
    }, open(os.path.join(log_root,'summary.json'),'w'), indent=2)

if __name__ == '__main__':
    main()
