"""
논문용 정성적 결과 시각화 스크립트
=====================================
train_unet_paper_eval.py와 동일한 split(seed=42)으로 test set을 재현하고,
best model 추론 결과를 그림으로 저장.

출력:
  _vis_paper/
    grid_basket.png   ← basket 도메인 전체 test 샘플 그리드
    grid_failure.png  ← failure 도메인 전체 test 샘플 그리드
    grid_all.png      ← 각 도메인에서 N개씩 뽑은 혼합 그리드 (논문용)

Usage:
  python visualize_paper.py \
    --root_base dataset_rgb_sam2 \
    --domains failure,basket \
    --ckpt dataset_rgb_sam2/unet_paper.pt \
    --seed 42 --size 384 --zmax 0.5 --thr 0.5 \
    --use_rgb --base 32 --use_se \
    --val_ratio 0.15 --test_ratio 0.15 \
    --outdir _vis_paper
"""

import os, argparse, random
from collections import defaultdict
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# train_unet_paper_eval.py에서 동일 코드 복사
# ────────────────────────────────────────────
#  모델
# ────────────────────────────────────────────
class SE(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, max(1, ch//r), bias=False), nn.ReLU(True),
            nn.Linear(max(1, ch//r), ch, bias=False), nn.Sigmoid())
    def forward(self, x):
        b,c,_,_ = x.shape
        return x * self.fc(self.avg(x).view(b,c)).view(b,c,1,1)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.se = SE(out_ch) if use_se else nn.Identity()
    def forward(self, x): return self.se(self.block(x))

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=True):
        super().__init__()
        self.pool = nn.MaxPool2d(2); self.conv = DoubleConv(in_ch, out_ch, use_se)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, use_se)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dh = x2.size(2)-x1.size(2); dw = x2.size(3)-x1.size(3)
        x1 = nn.functional.pad(x1, [dw//2,dw-dw//2,dh//2,dh-dh//2])
        return self.conv(torch.cat([x2,x1],dim=1))

class TinyUNet(nn.Module):
    def __init__(self, c_in=6, base=32, use_se=True):
        super().__init__()
        self.inc=DoubleConv(c_in,base,use_se); self.d1=Down(base,base*2,use_se)
        self.d2=Down(base*2,base*4,use_se);   self.d3=Down(base*4,base*8,use_se)
        self.u1=Up(base*8,base*4,use_se);     self.u2=Up(base*4,base*2,use_se)
        self.u3=Up(base*2,base,use_se);        self.outc=nn.Conv2d(base,1,1)
    def forward(self, x):
        x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3)
        x=self.u1(x4,x3); x=self.u2(x,x2); x=self.u3(x,x1)
        return self.outc(x)

# ────────────────────────────────────────────
#  Split 재현 (train_unet_paper_eval.py와 동일)
# ────────────────────────────────────────────
def get_domain(path):
    p = path.lower()
    if 'basket' in p: return 'basket'
    if 'failure' in p: return 'failure'
    return 'unknown'

def discover_samples(root_base, domains):
    samples = []
    for dom in domains:
        d = os.path.join(root_base, dom, "samples")
        for s in sorted(os.listdir(d)):
            p = os.path.join(d, s)
            if os.path.isdir(p) and all(os.path.exists(os.path.join(p,f))
                                         for f in ["diff.npy","cur.npy","mask.png"]):
                samples.append(p)
    return samples

def stratified_split(samples, val_ratio, test_ratio, seed):
    by_domain = defaultdict(list)
    for s in samples: by_domain[get_domain(s)].append(s)
    train_s, val_s, test_s = [], [], []
    rng = random.Random(seed)
    for dom, slist in sorted(by_domain.items()):
        rng.shuffle(slist)
        n = len(slist)
        n_test = max(1, round(n*test_ratio))
        n_val  = max(1, round(n*val_ratio))
        test_s  += slist[:n_test]
        val_s   += slist[n_test:n_test+n_val]
        train_s += slist[n_test+n_val:]
    return train_s, val_s, test_s

# ────────────────────────────────────────────
#  단일 샘플 추론
# ────────────────────────────────────────────
def infer_sample(net, sample_dir, size, zmax, use_rgb, dev, thr=0.5):
    """sample_dir → (rgb_hw3, gt_mask, pred_mask, prob_map, iou)"""
    # 로드
    D = np.nan_to_num(np.load(os.path.join(sample_dir,'diff.npy')).astype(np.float32))
    C = np.nan_to_num(np.load(os.path.join(sample_dir,'cur.npy')).astype(np.float32))
    M = (cv2.imread(os.path.join(sample_dir,'mask.png'), cv2.IMREAD_GRAYSCALE) > 0).astype(np.float32)

    p_rgb = os.path.join(sample_dir, 'rgb.png')
    if use_rgb and os.path.exists(p_rgb):
        rgb_orig = cv2.cvtColor(cv2.imread(p_rgb), cv2.COLOR_BGR2RGB)
    else:
        rgb_orig = np.zeros((*D.shape, 3), dtype=np.uint8)

    # 전처리
    th, tw = size, size
    D_n = np.clip(D, 0, 0.1)/0.1
    C_n = np.clip(C, 0, zmax)/max(1e-6, zmax)
    x = np.stack([D_n, C_n], axis=0)

    def resize2d(arr, h, w, inter=cv2.INTER_NEAREST):
        return cv2.resize(arr, (w, h), interpolation=inter)
    def resize3d(arr, h, w):
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)

    H, W = x.shape[1], x.shape[2]
    if H != th or W != tw:
        x = np.stack([resize2d(x[c], th, tw) for c in range(x.shape[0])], 0)
        M_r = resize2d(M, th, tw)
        rgb_r = resize3d(rgb_orig, th, tw)
    else:
        M_r = M; rgb_r = rgb_orig

    if use_rgb and os.path.exists(p_rgb):
        RGB = rgb_r.astype(np.float32) / 255.0
        x = np.concatenate([x, np.transpose(RGB, (2,0,1))], axis=0)
    else:
        x = np.concatenate([x, np.zeros((3,th,tw),dtype=np.float32)], axis=0)

    dom_id = 0 if get_domain(sample_dir) == 'basket' else 1
    dom_plane = np.full((1, th, tw), float(dom_id), dtype=np.float32)
    x = np.concatenate([x, dom_plane], axis=0)

    # 추론
    x_t = torch.tensor(x).unsqueeze(0).to(dev).float()
    with torch.no_grad():
        logit = net(x_t)
    prob = torch.sigmoid(logit).squeeze().cpu().numpy()
    pred = (prob > thr).astype(np.float32)

    # IoU 계산
    inter = (pred * M_r).sum()
    union = pred.sum() + M_r.sum() - inter + 1e-6
    iou = inter / union

    return rgb_r, M_r, pred, prob, float(iou)

# ────────────────────────────────────────────
#  시각화 그리드 생성
# ────────────────────────────────────────────
OVERLAY_ALPHA = 0.45

def make_overlay(rgb, mask, color):
    """RGB 이미지에 마스크를 반투명 색으로 오버레이."""
    out = rgb.copy().astype(np.float32) / 255.0
    for c, v in enumerate(color):
        out[..., c] = np.where(mask > 0.5,
                               out[..., c] * (1-OVERLAY_ALPHA) + v * OVERLAY_ALPHA,
                               out[..., c])
    return np.clip(out, 0, 1)

def draw_grid(samples_data, title, outpath, cols=4):
    """
    samples_data: list of (rgb, gt, pred, prob, iou, sample_name)
    각 샘플을 한 행으로: [RGB+GT overlay | RGB+Pred overlay | Probability map]
    """
    n = len(samples_data)
    n_cols = min(cols, n)          # 샘플 수
    n_rows = n                     # 샘플 하나당 1행
    fig_w = n_cols * 3.2           # 열당 너비
    fig_h = n_rows * 3.2           # 행당 높이

    fig, axes = plt.subplots(n_rows, 3, figsize=(9.6, n_rows*3.2),
                             squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    col_titles = ['Input + GT mask', 'Input + Prediction', 'Confidence map']
    for c, ct in enumerate(col_titles):
        axes[0][c].set_title(ct, fontsize=10, pad=6)

    green = (0.2, 0.9, 0.2)
    red   = (0.9, 0.2, 0.2)

    for row, (rgb, gt, pred, prob, iou, name) in enumerate(samples_data):
        gt_ov   = make_overlay(rgb, gt,   green)
        pred_ov = make_overlay(rgb, pred, red)

        axes[row][0].imshow(gt_ov);   axes[row][0].axis('off')
        axes[row][1].imshow(pred_ov); axes[row][1].axis('off')
        im = axes[row][2].imshow(prob, cmap='hot', vmin=0, vmax=1)
        axes[row][2].axis('off')
        plt.colorbar(im, ax=axes[row][2], fraction=0.046, pad=0.04)

        axes[row][0].set_ylabel(f"{name}\nIoU={iou*100:.1f}%",
                                 fontsize=8, rotation=0,
                                 labelpad=80, va='center')

    # 범례
    patch_gt   = mpatches.Patch(color=green, alpha=0.8, label='Ground truth')
    patch_pred = mpatches.Patch(color=red,   alpha=0.8, label='Prediction')
    fig.legend(handles=[patch_gt, patch_pred],
               loc='lower center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] {outpath}")

def draw_compact_grid(samples_basket, samples_failure, outpath, n_each=3):
    """
    논문용 compact 그리드:
    basket N개 + failure N개를 2행으로 배치.
    열: 샘플별 [GT overlay | Pred overlay]
    """
    sel_b = samples_basket[:n_each]
    sel_f = samples_failure[:n_each]

    n_col = n_each * 2   # GT+Pred 쌍
    fig, axes = plt.subplots(2, n_col, figsize=(n_col*2.4, 5.5))

    row_labels = ['Basket mode', 'Failure mode']
    green = (0.2, 0.9, 0.2)
    red   = (0.9, 0.2, 0.2)

    for row_idx, samples in enumerate([sel_b, sel_f]):
        for col_idx, (rgb, gt, pred, prob, iou, name) in enumerate(samples):
            gt_ov   = make_overlay(rgb, gt,   green)
            pred_ov = make_overlay(rgb, pred, red)

            ax_gt   = axes[row_idx][col_idx*2]
            ax_pred = axes[row_idx][col_idx*2+1]

            ax_gt.imshow(gt_ov);     ax_gt.axis('off')
            ax_pred.imshow(pred_ov); ax_pred.axis('off')
            ax_pred.set_title(f"IoU={iou*100:.1f}%", fontsize=8, pad=3)

            if col_idx == 0:
                ax_gt.set_ylabel(row_labels[row_idx], fontsize=10,
                                 fontweight='bold', rotation=90, labelpad=8)

    # 상단 column 헤더 (GT / Pred 반복)
    for ci in range(n_each):
        axes[0][ci*2].set_title('GT',   fontsize=8, pad=3, color='green')
        axes[0][ci*2+1].set_title('Pred', fontsize=8, pad=3, color='red')

    patch_gt   = mpatches.Patch(color=green, alpha=0.8, label='Ground truth')
    patch_pred = mpatches.Patch(color=red,   alpha=0.8, label='Prediction')
    fig.legend(handles=[patch_gt, patch_pred],
               loc='lower center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    plt.suptitle('TinyUNet Segmentation Results', fontsize=12,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] {outpath}")

# ────────────────────────────────────────────
#  개별 이미지 저장 (논문용 18장)
# ────────────────────────────────────────────
def save_best_individual(results, dom, outdir, n_best=3):
    """
    IoU 상위 n_best 샘플에 대해 각각 3장(gt, pred, conf)을 개별 PNG로 저장.
    파일명: {dom}_{rank}_{type}.png  (rank: 1-based)
    """
    sorted_res = sorted(results, key=lambda x: x[4], reverse=True)[:n_best]
    green = (0.2, 0.9, 0.2)
    red   = (0.9, 0.2, 0.2)

    for rank, (rgb, gt, pred, prob, iou, name) in enumerate(sorted_res, 1):
        base = os.path.join(outdir, f"{dom}_{rank}")

        # GT overlay
        gt_ov = make_overlay(rgb, gt, green)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(gt_ov); ax.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(f"{base}_gt.png", dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Prediction overlay
        pred_ov = make_overlay(rgb, pred, red)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(pred_ov); ax.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(f"{base}_pred.png", dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Confidence map
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        im = ax.imshow(prob, cmap='hot', vmin=0, vmax=1); ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout(pad=0)
        plt.savefig(f"{base}_conf.png", dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"  [{dom}] rank{rank} ({name}) IoU={iou*100:.1f}%  → {base}_{{gt,pred,conf}}.png")


# ────────────────────────────────────────────
#  main
# ────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_base',  default='dataset_rgb_sam2')
    ap.add_argument('--domains',    default='failure,basket')
    ap.add_argument('--ckpt',       default='dataset_rgb_sam2/unet_paper.pt')
    ap.add_argument('--seed',       type=int,   default=42)
    ap.add_argument('--size',       type=int,   default=384)
    ap.add_argument('--zmax',       type=float, default=0.5)
    ap.add_argument('--thr',        type=float, default=0.5)
    ap.add_argument('--val_ratio',  type=float, default=0.15)
    ap.add_argument('--test_ratio', type=float, default=0.15)
    ap.add_argument('--use_rgb',    action='store_true')
    ap.add_argument('--base',       type=int,   default=32)
    ap.add_argument('--use_se',     action='store_true')
    ap.add_argument('--device',     default=None)
    ap.add_argument('--outdir',     default='_vis_paper')
    ap.add_argument('--n_each',     type=int, default=3,
                    help='compact grid에 도메인별 샘플 수 (default: 3)')
    args = ap.parse_args()

    dev = torch.device(args.device if args.device
                       else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # ── split 재현
    domains = [d.strip() for d in args.domains.split(',') if d.strip()]
    all_samples = discover_samples(args.root_base, domains)
    random.seed(args.seed)
    _, _, test_s = stratified_split(all_samples, args.val_ratio, args.test_ratio, args.seed)

    by_dom = defaultdict(list)
    for s in test_s: by_dom[get_domain(s)].append(s)
    print(f"[TEST] basket={len(by_dom['basket'])}  failure={len(by_dom['failure'])}")

    # ── 모델 로드
    c_in = (5 if args.use_rgb else 2) + 1
    net = TinyUNet(c_in=c_in, base=args.base, use_se=args.use_se).to(dev)
    ckpt = torch.load(args.ckpt, map_location=dev)
    net.load_state_dict(ckpt['model'])
    net.eval()
    print(f"[MODEL] loaded: {args.ckpt}  (val IoU={ckpt.get('iou',0):.4f} @ ep {ckpt.get('ep',0)})")

    os.makedirs(args.outdir, exist_ok=True)

    # ── 추론
    def infer_domain(sample_list, dom):
        results = []
        for sp in sample_list:
            name = os.path.basename(sp)
            rgb, gt, pred, prob, iou = infer_sample(
                net, sp, args.size, args.zmax, args.use_rgb, dev, args.thr)
            print(f"  [{dom}] {name}: IoU={iou*100:.2f}%")
            results.append((rgb, gt, pred, prob, iou, name))
        return results

    res_basket  = infer_domain(by_dom['basket'],  'basket')
    res_failure = infer_domain(by_dom['failure'], 'failure')

    # ── 도메인별 전체 그리드
    if res_basket:
        draw_grid(res_basket,
                  f'Basket mode — test set (N={len(res_basket)})',
                  os.path.join(args.outdir, 'grid_basket.png'))

    if res_failure:
        draw_grid(res_failure,
                  f'Failure recovery mode — test set (N={len(res_failure)})',
                  os.path.join(args.outdir, 'grid_failure.png'))

    # ── 논문용 compact 그리드
    n_each = min(args.n_each, len(res_basket), len(res_failure))
    if n_each > 0:
        draw_compact_grid(res_basket, res_failure,
                          os.path.join(args.outdir, 'grid_paper.png'),
                          n_each=n_each)

    # ── 개별 이미지 저장 (IoU 상위 3개씩, 총 18장)
    ind_dir = os.path.join(args.outdir, 'individual')
    os.makedirs(ind_dir, exist_ok=True)
    print(f"\n[Individual] IoU 상위 {args.n_each}개 × 2도메인 × 3타입 = {args.n_each*2*3}장 저장 → {ind_dir}/")
    if res_basket:
        save_best_individual(res_basket,  'basket',  ind_dir, n_best=args.n_each)
    if res_failure:
        save_best_individual(res_failure, 'failure', ind_dir, n_best=args.n_each)

    # ── IoU 통계 출력
    print(f"\n{'='*50}")
    for dom, res in [('basket', res_basket), ('failure', res_failure)]:
        ious = [r[4] for r in res]
        if ious:
            print(f"  {dom:10s} IoU: mean={np.mean(ious)*100:.2f}%  "
                  f"std={np.std(ious)*100:.2f}%  "
                  f"min={np.min(ious)*100:.2f}%  max={np.max(ious)*100:.2f}%")
    all_ious = [r[4] for r in res_basket+res_failure]
    print(f"  {'overall':10s} IoU: mean={np.mean(all_ious)*100:.2f}%  "
          f"std={np.std(all_ious)*100:.2f}%")
    print(f"{'='*50}")
    print(f"\n[DONE] 결과 저장 위치: {os.path.abspath(args.outdir)}/")

if __name__ == '__main__':
    main()
