"""
benchmark_models.py
경량화(TinyUNet) vs 비경량화(DeepLabV3+, UNet-ResNet50) 성능 비교 측정

실행:
  python benchmark_models.py
  python benchmark_models.py --device cuda   # GPU 측정 시
"""

import os, time, argparse, statistics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil

# ── 측정 설정 ──────────────────────────────────────────────────────────────────
CKPT_BASKET  = "/home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/unet_best_basket_new.pt"
CKPT_FAILURE = "/home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/unet_best_failure_new.pt"
INPUT_HW     = (480, 640)   # 실제 카메라 해상도
INFER_SIZE   = 384          # TinyUNet 입력 크기
N_WARMUP     = 10
N_REPEAT     = 50

# ── TinyUNet 정의 (node2_backup_in_process.py와 동일) ─────────────────────────
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

# ── 헬퍼 ──────────────────────────────────────────────────────────────────────
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def model_file_mb(path):
    return os.path.getsize(path) / 1024 / 1024

def measure_inference(model, dummy_input, device, n_warmup=N_WARMUP, n_repeat=N_REPEAT):
    model.eval()
    x = dummy_input.to(device)
    # warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)  # ms

    return {
        "mean_ms":   round(statistics.mean(times), 2),
        "median_ms": round(statistics.median(times), 2),
        "std_ms":    round(statistics.stdev(times), 2),
        "min_ms":    round(min(times), 2),
        "fps":       round(1000 / statistics.mean(times), 1),
    }

def model_param_mb(model):
    """파라미터 메모리 이론값 (float32 기준)"""
    return round(sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024, 1)

def print_result(name, params, file_mb, timing, param_mb, input_shape):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  입력 크기         : {input_shape}")
    print(f"  총 파라미터       : {params[0]:,}  ({params[0]/1e6:.2f}M)")
    if file_mb:
        print(f"  모델 파일 크기    : {file_mb:.1f} MB")
    print(f"  파라미터 메모리   : {param_mb} MB  (float32 이론값)")
    print(f"  호출 1회 지연     : {timing['mean_ms']} ms  (median {timing['median_ms']} ms)")
    print(f"  편차              : ±{timing['std_ms']} ms")
    print(f"  FPS (참고)        : {timing['fps']}")

# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"\n[설정] device={device}, warmup={N_WARMUP}, repeat={N_REPEAT}")
    print(f"[CPU] {psutil.cpu_count()} cores, RAM {psutil.virtual_memory().total/1024**3:.1f} GB")
    if device.type == 'cuda':
        print(f"[GPU] {torch.cuda.get_device_name(device)}, VRAM {torch.cuda.get_device_properties(device).total_memory/1024**3:.1f} GB")

    results = {}

    # ── 1. TinyUNet basket (경량화) ────────────────────────────────────────────
    print("\n[1/4] TinyUNet basket 로드 중...")
    ckpt = torch.load(CKPT_BASKET, map_location='cpu')
    net_b = TinyUNet(c_in=int(ckpt.get('c_in',6)), base=int(ckpt.get('base',32)),
                     use_se=bool(ckpt.get('use_se',True))).to(device)
    net_b.load_state_dict(ckpt['model']); net_b.eval()
    dummy_b = torch.randn(1, int(ckpt.get('c_in',6)), INFER_SIZE, INFER_SIZE)
    timing_b  = measure_inference(net_b, dummy_b, device)
    pmb_b     = model_param_mb(net_b)
    results['TinyUNet-basket'] = {
        'params': count_params(net_b), 'file_mb': model_file_mb(CKPT_BASKET),
        'timing': timing_b, 'mem_mb': pmb_b,
        'input': f"(1,{int(ckpt.get('c_in',6))},{INFER_SIZE},{INFER_SIZE})"
    }
    print_result("경량화 ① TinyUNet-basket (c_in=6, base=32)",
                 results['TinyUNet-basket']['params'], results['TinyUNet-basket']['file_mb'],
                 timing_b, pmb_b, results['TinyUNet-basket']['input'])

    # ── 2. TinyUNet failure (경량화) ───────────────────────────────────────────
    print("\n[2/5] TinyUNet failure 로드 중...")
    ckpt_f = torch.load(CKPT_FAILURE, map_location='cpu')
    net_f  = TinyUNet(c_in=int(ckpt_f.get('c_in',6)), base=int(ckpt_f.get('base',32)),
                      use_se=bool(ckpt_f.get('use_se',True))).to(device)
    net_f.load_state_dict(ckpt_f['model']); net_f.eval()
    dummy_f  = torch.randn(1, int(ckpt_f.get('c_in',6)), INFER_SIZE, INFER_SIZE)
    timing_f = measure_inference(net_f, dummy_f, device)
    pmb_f    = model_param_mb(net_f)
    results['TinyUNet-failure'] = {
        'params': count_params(net_f), 'file_mb': model_file_mb(CKPT_FAILURE),
        'timing': timing_f, 'mem_mb': pmb_f,
        'input': f"(1,{int(ckpt_f.get('c_in',6))},{INFER_SIZE},{INFER_SIZE})"
    }
    print_result("경량화 ② TinyUNet-failure (c_in=6, base=32)",
                 results['TinyUNet-failure']['params'], results['TinyUNet-failure']['file_mb'],
                 timing_f, pmb_f, results['TinyUNet-failure']['input'])

    # ── 3. DeepLabV3+ ResNet50 (비경량화) ─────────────────────────────────────
    print("\n[3/5] DeepLabV3-ResNet50 로드 중...")
    try:
        from torchvision.models.segmentation import deeplabv3_resnet50
        net_dl   = deeplabv3_resnet50(weights=None, num_classes=2).to(device).eval()
        dummy_dl = torch.randn(1, 3, *INPUT_HW)
        timing_dl = measure_inference(net_dl, dummy_dl, device)
        pmb_dl    = model_param_mb(net_dl)
        results['DeepLabV3-R50'] = {
            'params': count_params(net_dl), 'file_mb': None,
            'timing': timing_dl, 'mem_mb': pmb_dl,
            'input': f"(1,3,{INPUT_HW[0]},{INPUT_HW[1]})"
        }
        print_result("비경량화 ① DeepLabV3+ ResNet50",
                     results['DeepLabV3-R50']['params'], None,
                     timing_dl, pmb_dl, results['DeepLabV3-R50']['input'])
    except Exception as e:
        print(f"  [SKIP] DeepLabV3 로드 실패: {e}")

    # ── 4. FCN ResNet101 (비경량화) ────────────────────────────────────────────
    print("\n[4/5] FCN-ResNet101 로드 중...")
    try:
        from torchvision.models.segmentation import fcn_resnet101
        net_fcn   = fcn_resnet101(weights=None, num_classes=2).to(device).eval()
        dummy_fcn = torch.randn(1, 3, *INPUT_HW)
        timing_fcn = measure_inference(net_fcn, dummy_fcn, device)
        pmb_fcn    = model_param_mb(net_fcn)
        results['FCN-R101'] = {
            'params': count_params(net_fcn), 'file_mb': None,
            'timing': timing_fcn, 'mem_mb': pmb_fcn,
            'input': f"(1,3,{INPUT_HW[0]},{INPUT_HW[1]})"
        }
        print_result("비경량화 ② FCN ResNet101",
                     results['FCN-R101']['params'], None,
                     timing_fcn, pmb_fcn, results['FCN-R101']['input'])
    except Exception as e:
        print(f"  [SKIP] FCN-ResNet101 로드 실패: {e}")

    # ── 5. SegFormer-B5 (대형 Transformer) ────────────────────────────────────
    print("\n[5/5] SegFormer-B5 로드 중...")
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        cfg = SegformerConfig(
            num_encoder_blocks=4,
            depths=[3, 6, 40, 3],
            hidden_sizes=[64, 128, 320, 512],
            num_attention_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            num_labels=2,
        )
        net_sf   = SegformerForSemanticSegmentation(cfg).to(device).eval()
        dummy_sf = torch.randn(1, 3, *INPUT_HW)
        timing_sf = measure_inference(net_sf, dummy_sf, device)
        pmb_sf    = model_param_mb(net_sf)
        results['SegFormer-B5'] = {
            'params': count_params(net_sf), 'file_mb': None,
            'timing': timing_sf, 'mem_mb': pmb_sf,
            'input': f"(1,3,{INPUT_HW[0]},{INPUT_HW[1]})"
        }
        print_result("비경량화 ③ SegFormer-B5 (Transformer)",
                     results['SegFormer-B5']['params'], None,
                     timing_sf, pmb_sf, results['SegFormer-B5']['input'])
    except Exception as e:
        print(f"  [SKIP] SegFormer-B5 로드 실패: {e}")

    # ── 6. SAM ViT-B / ViT-H (SOTA 대형 모델) ────────────────────────────────
    print("\n[6/6] SAM 로드 중...")
    try:
        from segment_anything import sam_model_registry
        from segment_anything.modeling import Sam

        for sam_type, vit_type in [("SAM-ViT-B", "vit_b"), ("SAM-ViT-H", "vit_h")]:
            # 체크포인트 없이 아키텍처만 생성 (추론 속도/파라미터 측정용)
            if vit_type == "vit_b":
                from segment_anything.modeling import (
                    ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer)
                image_encoder = ImageEncoderViT(
                    depth=12, embed_dim=768, img_size=1024,
                    mlp_ratio=4, num_heads=12, patch_size=16,
                    qkv_bias=True, use_rel_pos=True,
                    window_size=14, out_chans=256,
                    global_attn_indexes=[2,5,8,11])
            else:  # vit_h
                from segment_anything.modeling import (
                    ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer)
                image_encoder = ImageEncoderViT(
                    depth=32, embed_dim=1280, img_size=1024,
                    mlp_ratio=4, num_heads=16, patch_size=16,
                    qkv_bias=True, use_rel_pos=True,
                    window_size=14, out_chans=256,
                    global_attn_indexes=[7,15,23,31])

            image_encoder = image_encoder.to(device).eval()
            # SAM은 1024x1024 입력 사용
            dummy_sam = torch.randn(1, 3, 1024, 1024)
            timing_sam = measure_inference(image_encoder, dummy_sam, device,
                                           n_warmup=3, n_repeat=10)
            pmb_sam = model_param_mb(image_encoder)
            p_sam   = count_params(image_encoder)
            results[sam_type] = {
                'params': p_sam, 'file_mb': None,
                'timing': timing_sam, 'mem_mb': pmb_sam,
                'input': "(1,3,1024,1024)"
            }
            print_result(f"SOTA ④ {sam_type} Image Encoder",
                         p_sam, None, timing_sam, pmb_sam, "(1,3,1024,1024)")
    except Exception as e:
        print(f"  [SKIP] SAM 로드 실패: {e}")

    # ── 최종 비교 테이블 ───────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("  최종 비교 테이블")
    print(f"{'='*80}")
    tiny_ms = results.get('TinyUNet-basket', {}).get('timing', {}).get('mean_ms', 1)
    header = f"{'모델':<22} {'파라미터(M)':>11} {'파일(MB)':>9} {'파라미터메모리':>13} {'호출지연(ms)':>12} {'FPS':>6} {'배율(vs TinyUNet)':>17}"
    print(header)
    print("-"*95)
    for name, r in results.items():
        p_m   = f"{r['params'][0]/1e6:.2f}"
        f_mb  = f"{r['file_mb']:.1f}" if r['file_mb'] else "N/A"
        pmb   = f"{r['mem_mb']}"
        ms    = f"{r['timing']['mean_ms']}"
        fps   = f"{r['timing']['fps']}"
        ratio = f"x{r['timing']['mean_ms']/tiny_ms:.1f}" if tiny_ms else "-"
        print(f"  {name:<20} {p_m:>11} {f_mb:>9} {pmb:>13} {ms:>12} {fps:>6} {ratio:>17}")
    print(f"{'='*95}")
    print(f"\n[측정 조건] device={device}, TinyUNet 입력={INFER_SIZE}x{INFER_SIZE} / 비경량화 입력={INPUT_HW}, repeat={N_REPEAT}회 평균")
    print(f"[참고] 로봇 운용 환경 = CPU. 세탁물 검출은 호출 1회 지연시간이 핵심 지표.")

if __name__ == '__main__':
    main()
