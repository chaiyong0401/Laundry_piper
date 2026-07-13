"""
TinyUNet eval-only script
평가 대상: basket_new / failure_new (unet_paper.pt 학습에 미포함된 unseen 데이터)

Usage:
  python eval_unet_paper.py \
    --root_base dataset_rgb_sam2 \
    --domains basket_new,failure_new \
    --ckpt dataset_rgb_sam2/unet_paper.pt \
    --device cuda --size 384 --zmax 0.5 --thr 0.5
"""
import os, argparse, json
from collections import defaultdict
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ── 모델 (train_unet_paper_eval.py와 동일) ─────────────────────────────────
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
        return x * self.fc(self.avg(x).view(b, c)).view(b, c, 1, 1)

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


# ── 데이터셋 ────────────────────────────────────────────────────────────────
def get_domain_label(sample_dir: str) -> str:
    p = sample_dir.lower()
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

class LaundrySet(Dataset):
    def __init__(self, sample_dirs, zmax=0.5, size=None, use_rgb=True):
        self.items = list(sample_dirs)
        self.zmax = zmax
        self.size = size
        self.use_rgb = use_rgb

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        p = self.items[idx]
        diff = np.load(os.path.join(p, "diff.npy")).astype(np.float32)
        cur  = np.load(os.path.join(p, "cur.npy")).astype(np.float32)
        mask = cv2.imread(os.path.join(p, "mask.png"), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        diff = np.clip(diff, 0, 0.1) / 0.1
        cur  = np.clip(cur,  0, self.zmax) / self.zmax

        dom_id = 0.0 if 'basket' in p.lower() else 1.0
        dom_ch = np.full_like(diff, dom_id)

        if self.use_rgb:
            rgb_path = os.path.join(p, "rgb.png")
            if os.path.exists(rgb_path):
                rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            else:
                rgb = np.zeros((*diff.shape, 3), dtype=np.float32)
            x = np.stack([diff, cur, rgb[...,0], rgb[...,1], rgb[...,2], dom_ch], axis=0)
        else:
            x = np.stack([diff, cur, dom_ch], axis=0)

        if self.size is not None:
            H, W = self.size
            x    = np.stack([cv2.resize(x[c], (W, H), interpolation=cv2.INTER_LINEAR)
                             for c in range(x.shape[0])], axis=0)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        return (torch.from_numpy(x),
                torch.from_numpy(mask[None]),
                get_domain_label(p))


# ── 메트릭 ──────────────────────────────────────────────────────────────────
def compute_metrics(logits, target, thr=0.5):
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = (pred + target).clamp(0,1).sum(dim=(1,2,3))
    iou  = (inter / (union + 1e-6)).mean().item()
    dice = (2 * inter / (pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + 1e-6)).mean().item()
    return iou, dice


# ── 평가 루프 ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(net, loader, thr, dev):
    net.eval()
    agg = defaultdict(lambda: dict(iou=0.0, dice=0.0, n=0))
    for x, y, domains in loader:
        x = x.to(dev).float(); y = y.to(dev).float()
        logits = net(x)
        for i, dom in enumerate(domains):
            iou, dice = compute_metrics(logits[i:i+1], y[i:i+1], thr)
            agg[dom]['iou']  += iou
            agg[dom]['dice'] += dice
            agg[dom]['n']    += 1

    result = {}
    all_iou, all_dice, all_n = 0.0, 0.0, 0
    for dom, m in sorted(agg.items()):
        n = m['n']
        result[dom] = dict(iou=m['iou']/n, dice=m['dice']/n, n=n)
        all_iou += m['iou']; all_dice += m['dice']; all_n += n
    result['_all'] = dict(iou=all_iou/max(1,all_n), dice=all_dice/max(1,all_n), n=all_n)
    return result


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_base', default='dataset_rgb_sam2')
    ap.add_argument('--domains',   default='basket_new,failure_new')
    ap.add_argument('--ckpt',      default='dataset_rgb_sam2/unet_paper.pt')
    ap.add_argument('--device',    default=None)
    ap.add_argument('--size',      type=int, default=384)
    ap.add_argument('--zmax',      type=float, default=0.5)
    ap.add_argument('--thr',       type=float, default=0.5)
    ap.add_argument('--base',      type=int, default=32)
    ap.add_argument('--batch',     type=int, default=8)
    ap.add_argument('--no_rgb',    action='store_true')
    args = ap.parse_args()

    dev = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    domains = [d.strip() for d in args.domains.split(',') if d.strip()]
    samples = discover_samples(args.root_base, domains)

    # 도메인별 샘플 수 출력
    from collections import Counter
    dom_counts = Counter(get_domain_label(s) for s in samples)
    print(f"\n[DATA] 전체 샘플: {len(samples)}개")
    for dom, cnt in sorted(dom_counts.items()):
        print(f"  {dom}: {cnt}개")

    dataset = LaundrySet(samples, zmax=args.zmax, size=(args.size, args.size),
                         use_rgb=not args.no_rgb)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=4)

    use_rgb = not args.no_rgb
    c_in = 6 if use_rgb else 3
    net = TinyUNet(c_in=c_in, base=args.base, use_se=True).to(dev)
    state = torch.load(args.ckpt, map_location=dev)
    # 체크포인트가 {"model": state_dict, ...} 형태로 저장된 경우 처리
    if isinstance(state, dict) and 'model' in state:
        ckpt_meta = {k: v for k, v in state.items() if k != 'model'}
        print(f"[CKPT] meta: {ckpt_meta}")
        state = state['model']
    net.load_state_dict(state)
    print(f"\n[CKPT] {args.ckpt} 로드 완료")

    print("[EVAL] 평가 중...")
    result = evaluate(net, loader, args.thr, dev)

    print("\n" + "="*52)
    print(f"{'Mode':<20} {'IoU (%)':>10} {'Dice (%)':>10} {'n':>6}")
    print("-"*52)
    label_map = {'basket': 'Basket (d_dom=0.0)', 'failure': 'Failure (d_dom=1.0)'}
    for dom in sorted(k for k in result if not k.startswith('_')):
        m = result[dom]
        label = label_map.get(dom, dom)
        print(f"{label:<20} {m['iou']*100:>10.2f} {m['dice']*100:>10.2f} {m['n']:>6}")
    print("-"*52)
    m = result['_all']
    print(f"{'Overall':<20} {m['iou']*100:>10.2f} {m['dice']*100:>10.2f} {m['n']:>6}")
    print("="*52)

    out_path = os.path.join(os.path.dirname(args.ckpt), 'eval_results_new.json')
    with open(out_path, 'w') as f:
        json.dump({'ckpt': args.ckpt, 'domains': domains, 'results': result}, f, indent=2)
    print(f"\n[SAVE] {out_path}")


if __name__ == '__main__':
    main()
