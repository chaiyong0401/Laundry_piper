"""
ablation_unet.py  — TinyUNet vision ablation study
====================================================
Row 1: Single TinyUNet without domain channel  (c_in=5, basket+failure 공동 학습)
Row 2: Two separate TinyUNets                  (basket 전용 / failure 전용, c_in=5)
Row 3: Domain-conditioned TinyUNet             (paper_results.json 에서 로드)

동일한 seed=42, val=15%, test=15% 분할로 모든 행을 평가.

Usage:
  python ablation_unet.py --root_base dataset_rgb_sam2 --device cuda
"""

import os, json, random, time
from collections import defaultdict
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── 재현성 ────────────────────────────────────────────────────────────────────
SEED       = 42
VAL_RATIO  = 0.15
TEST_RATIO = 0.15
EPOCHS     = 60
BATCH      = 6
LR         = 1e-3
PATIENCE   = 10
ZMAX       = 0.5
SIZE       = 384
BCE_W      = 0.7
DICE_W     = 0.3
THR        = 0.5
BASE       = 32

def seed_everything(s):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(wid):
    np.random.seed((torch.initial_seed() + wid) % 2**32)
    random.seed((torch.initial_seed() + wid) % 2**32)

# ── 모델 ──────────────────────────────────────────────────────────────────────
class SE(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, max(1, ch//r), bias=False), nn.ReLU(True),
            nn.Linear(max(1, ch//r), ch, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.shape
        return x * self.fc(self.avg(x).view(b, c)).view(b, c, 1, 1)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.se = SE(out_ch)
    def forward(self, x): return self.se(self.block(x))

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2); self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dh = x2.size(2)-x1.size(2); dw = x2.size(3)-x1.size(3)
        x1 = nn.functional.pad(x1, [dw//2, dw-dw//2, dh//2, dh-dh//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class TinyUNet(nn.Module):
    def __init__(self, c_in=6):
        super().__init__()
        b = BASE
        self.inc = DoubleConv(c_in, b)
        self.d1  = Down(b,   b*2)
        self.d2  = Down(b*2, b*4)
        self.d3  = Down(b*4, b*8)
        self.u1  = Up(b*8, b*4)
        self.u2  = Up(b*4, b*2)
        self.u3  = Up(b*2, b)
        self.out = nn.Conv2d(b, 1, 1)
    def forward(self, x):
        x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3)
        x=self.u1(x4,x3); x=self.u2(x,x2); x=self.u3(x,x1)
        return self.out(x)

# ── 데이터 유틸 ───────────────────────────────────────────────────────────────
def get_domain(path):
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
                                         for f in ["diff.npy","cur.npy","mask.png"]):
                samples.append(p)
    return samples

def stratified_split(samples, val_ratio, test_ratio, seed):
    by_dom = defaultdict(list)
    for s in samples: by_dom[get_domain(s)].append(s)
    train_s, val_s, test_s = [], [], []
    rng = random.Random(seed)
    for dom, sl in sorted(by_dom.items()):
        rng.shuffle(sl)
        n = len(sl)
        nt = max(1, round(n*test_ratio)); nv = max(1, round(n*val_ratio))
        test_s += sl[:nt]; val_s += sl[nt:nt+nv]; train_s += sl[nt+nv:]
        print(f"  [{dom}] total={n}  train={n-nt-nv}  val={nv}  test={nt}")
    return train_s, val_s, test_s

# ── Augmentation (간소화) ─────────────────────────────────────────────────────
def augment(x, m, rgb):
    if random.random() < 0.5:
        x=x[:,:,::-1].copy(); m=m[:,::-1].copy()
        if rgb is not None: rgb=rgb[:,::-1,:].copy()
    if random.random() < 0.3:
        noise = np.random.normal(0, 0.01, x.shape).astype(np.float32)
        x = np.clip(x+noise, 0, 1)
    if rgb is not None and random.random() < 0.5:
        alpha = random.uniform(0.92,1.08); beta = random.uniform(-0.08,0.08)
        rgb = np.clip(alpha*rgb+beta, 0, 1)
    return x, m, rgb

# ── Dataset ───────────────────────────────────────────────────────────────────
class LaundrySet(Dataset):
    """use_domain=True → c_in=6 (domain channel 포함), False → c_in=5"""
    def __init__(self, samples, train=True, use_domain=True):
        self.items = list(samples)
        self.train = train
        self.use_domain = use_domain

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        d = self.items[i]
        D = np.nan_to_num(np.load(os.path.join(d,'diff.npy')).astype(np.float32))
        C = np.nan_to_num(np.load(os.path.join(d,'cur.npy')).astype(np.float32))
        M = (cv2.imread(os.path.join(d,'mask.png'), cv2.IMREAD_GRAYSCALE) > 0).astype(np.float32)
        D = np.clip(D, 0, 0.1) / 0.1
        C = np.clip(C, 0, ZMAX) / ZMAX

        rp = os.path.join(d, 'rgb.png')
        RGB = cv2.cvtColor(cv2.imread(rp), cv2.COLOR_BGR2RGB).astype(np.float32)/255. \
              if os.path.exists(rp) else None

        x = np.stack([D, C], 0)
        H, W = x.shape[1:]
        x, M, RGB = self._resize(x, M, RGB, H, W)
        if self.train:
            x, M, RGB = augment(x, M, RGB)
        if RGB is None:
            RGB = np.zeros((x.shape[1], x.shape[2], 3), np.float32)
        x = np.concatenate([x, np.transpose(RGB,(2,0,1))], 0)  # (5,H,W)

        if self.use_domain:
            dom_id = 0.0 if get_domain(d)=='basket' else 1.0
            x = np.concatenate([x, np.full((1,x.shape[1],x.shape[2]),dom_id,np.float32)], 0)  # (6,H,W)

        return torch.from_numpy(x), torch.from_numpy(M[None]), get_domain(d)

    def _resize(self, x, m, rgb, H, W):
        th = tw = SIZE
        if self.train and H>=th and W>=tw:
            y0=random.randint(0,H-th); x0=random.randint(0,W-tw)
            x=x[:,y0:y0+th,x0:x0+tw]; m=m[y0:y0+th,x0:x0+tw]
            if rgb is not None: rgb=rgb[y0:y0+th,x0:x0+tw]
        else:
            x=np.stack([cv2.resize(x[c],(tw,th),interpolation=cv2.INTER_LINEAR) for c in range(x.shape[0])],0)
            m=cv2.resize(m,(tw,th),interpolation=cv2.INTER_NEAREST)
            if rgb is not None: rgb=cv2.resize(rgb,(tw,th),interpolation=cv2.INTER_LINEAR)
        return x, m, rgb

def collate_fn(batch):
    xs, ys, ds = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(ds)

# ── Loss / Metric ─────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        num = 2*(p*target).sum(dim=(1,2,3)) + 1e-6
        den = p.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + 1e-6
        return 1 - (num/den).mean()

@torch.no_grad()
def compute_metrics(logits, target):
    p = (torch.sigmoid(logits) > THR).float()
    inter = (p*target).sum(dim=(1,2,3))
    union = p.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - inter + 1e-6
    iou  = (inter/union).mean().item()
    dice = (2*inter/(p.sum(dim=(1,2,3))+target.sum(dim=(1,2,3))+1e-6)).mean().item()
    return iou, dice

@torch.no_grad()
def evaluate(net, loader, bce_fn, dice_fn, dev):
    net.eval()
    agg = defaultdict(lambda: dict(iou=0., dice=0., n=0))
    for x, y, doms in loader:
        x=x.to(dev).float(); y=y.to(dev).float()
        z = net(x)
        for i, dom in enumerate(doms):
            iou, dice = compute_metrics(z[i:i+1], y[i:i+1])
            agg[dom]['iou'] += iou; agg[dom]['dice'] += dice; agg[dom]['n'] += 1
    result = {}
    for dom, m in sorted(agg.items()):
        n = max(1, m['n'])
        result[dom] = dict(iou=m['iou']/n, dice=m['dice']/n, n=n)
    return result

# ── 학습 함수 ─────────────────────────────────────────────────────────────────
def train_model(label, train_s, val_s, test_s, dev, use_domain, ckpt_path):
    print(f"\n{'='*60}")
    print(f"  Training: {label}  (c_in={'6' if use_domain else '5'})")
    print(f"  train={len(train_s)}  val={len(val_s)}  test={len(test_s)}")
    print('='*60)

    pin = (dev.type == 'cuda')
    g = torch.Generator(); g.manual_seed(SEED)
    dl_tr = DataLoader(LaundrySet(train_s, train=True,  use_domain=use_domain),
                       BATCH, shuffle=True, collate_fn=collate_fn,
                       num_workers=4, pin_memory=pin, worker_init_fn=seed_worker, generator=g)
    dl_va = DataLoader(LaundrySet(val_s,   train=False, use_domain=use_domain),
                       BATCH, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=pin)
    dl_te = DataLoader(LaundrySet(test_s,  train=False, use_domain=use_domain),
                       BATCH, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=pin)

    c_in = 6 if use_domain else 5
    net = TinyUNet(c_in=c_in).to(dev)
    bce_fn = nn.BCEWithLogitsLoss()
    dice_fn = DiceLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=4)

    best_iou, best_ep, no_imp = 0., 0, 0
    t0 = time.time()

    for ep in range(1, EPOCHS+1):
        net.train(); loss_tr = 0.
        for x, y, _ in dl_tr:
            x=x.to(dev).float(); y=y.to(dev).float()
            opt.zero_grad(set_to_none=True)
            z = net(x)
            loss = BCE_W*bce_fn(z,y) + DICE_W*dice_fn(z,y)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            loss_tr += loss.item()*x.size(0)

        val_res = evaluate(net, dl_va, bce_fn, dice_fn, dev)
        val_iou_all = np.mean([v['iou'] for v in val_res.values()])
        sch.step(val_iou_all)

        if val_iou_all > best_iou:
            best_iou = val_iou_all; best_ep = ep; no_imp = 0
            torch.save({'model': net.state_dict(), 'iou': best_iou, 'ep': ep}, ckpt_path)
        else:
            no_imp += 1

        if ep % 10 == 0 or ep == EPOCHS:
            print(f"  ep={ep:3d}  val_iou={val_iou_all:.4f}  best={best_iou:.4f}(ep{best_ep})"
                  f"  lr={opt.param_groups[0]['lr']:.2e}  [{time.time()-t0:.0f}s]")

        if no_imp >= PATIENCE:
            print(f"  Early stop at ep={ep}")
            break

    # 최종 test 평가
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    net.load_state_dict(ckpt['model'])
    test_res = evaluate(net, dl_te, bce_fn, dice_fn, dev)
    return test_res

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_base', default='dataset_rgb_sam2')
    ap.add_argument('--device',    default=None)
    ap.add_argument('--paper_json', default='dataset_rgb_sam2/_logs_paper/paper_results.json')
    args = ap.parse_args()

    seed_everything(SEED)
    dev = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {dev}")

    # ── 공통 데이터 분할 (basket + failure) ──────────────────────────────────
    print("\n[DATA] basket+failure 분할")
    all_samples = discover_samples(args.root_base, ['basket', 'failure'])
    train_s, val_s, test_s = stratified_split(all_samples, VAL_RATIO, TEST_RATIO, SEED)

    # 도메인별로 분리 (Row 2 용)
    basket_all  = [s for s in all_samples if get_domain(s)=='basket']
    failure_all = [s for s in all_samples if get_domain(s)=='failure']

    def single_domain_split(samples, seed):
        rng = random.Random(seed)
        sl = list(samples); rng.shuffle(sl)
        n = len(sl)
        nt = max(1, round(n*TEST_RATIO)); nv = max(1, round(n*VAL_RATIO))
        return sl[nt+nv:], sl[nt:nt+nv], sl[:nt]  # train, val, test

    bk_tr, bk_va, bk_te = single_domain_split(basket_all,  SEED)
    fa_tr, fa_va, fa_te = single_domain_split(failure_all, SEED)
    print(f"  [basket-only]  train={len(bk_tr)}  val={len(bk_va)}  test={len(bk_te)}")
    print(f"  [failure-only] train={len(fa_tr)}  val={len(fa_va)}  test={len(fa_te)}")

    os.makedirs('dataset_rgb_sam2/_logs_ablation', exist_ok=True)

    # ── Row 1: No domain channel ──────────────────────────────────────────────
    r1 = train_model(
        label      = "Row1: No domain channel",
        train_s    = train_s, val_s=val_s, test_s=test_s,
        dev        = dev,
        use_domain = False,
        ckpt_path  = 'dataset_rgb_sam2/_logs_ablation/row1_no_domain.pt'
    )

    # ── Row 2a: Basket-only model ─────────────────────────────────────────────
    r2_basket = train_model(
        label      = "Row2a: Basket-only TinyUNet",
        train_s    = bk_tr, val_s=bk_va, test_s=bk_te,
        dev        = dev,
        use_domain = False,
        ckpt_path  = 'dataset_rgb_sam2/_logs_ablation/row2_basket.pt'
    )

    # ── Row 2b: Failure-only model ────────────────────────────────────────────
    r2_failure = train_model(
        label      = "Row2b: Failure-only TinyUNet",
        train_s    = fa_tr, val_s=fa_va, test_s=fa_te,
        dev        = dev,
        use_domain = False,
        ckpt_path  = 'dataset_rgb_sam2/_logs_ablation/row2_failure.pt'
    )

    # Row 2 결합
    r2 = {
        'basket':  r2_basket.get('basket', {'iou':0.,'dice':0.,'n':0}),
        'failure': r2_failure.get('failure', {'iou':0.,'dice':0.,'n':0}),
    }

    # ── Row 3: Domain-conditioned (paper_results.json) ────────────────────────
    with open(args.paper_json) as f:
        paper = json.load(f)
    r3 = paper['test']

    # ── 최종 출력 ─────────────────────────────────────────────────────────────
    def fmt(res, dom):
        m = res.get(dom, {})
        iou  = m.get('iou',  0.) * 100
        dice = m.get('dice', 0.) * 100
        n    = m.get('n', 0)
        return f"{iou:5.2f}%", f"{dice:5.2f}%", n

    print("\n" + "="*72)
    print(f"{'Method':<38} {'Basket IoU':>11} {'Recovery IoU':>13} {'Basket Dice':>12} {'Recovery Dice':>14}")
    print("-"*72)
    rows = [
        ("Single TinyUNet (no domain ch.)", r1),
        ("Two separate TinyUNets",           r2),
        ("Domain-conditioned TinyUNet",       r3),
    ]
    for name, res in rows:
        bi, bd, bn = fmt(res, 'basket')
        fi, fd, fn = fmt(res, 'failure')
        print(f"{name:<38} {bi:>11} {fi:>13} {bd:>12} {fd:>14}  (n={bn}/{fn})")
    print("="*72)

    ablation = {
        'row1_no_domain':     r1,
        'row2_two_separate': r2,
        'row3_domain_cond':  r3,
    }
    out = 'dataset_rgb_sam2/_logs_ablation/ablation_results.json'
    with open(out, 'w') as f:
        json.dump(ablation, f, indent=2)
    print(f"\n[SAVE] {out}")

if __name__ == '__main__':
    main()
