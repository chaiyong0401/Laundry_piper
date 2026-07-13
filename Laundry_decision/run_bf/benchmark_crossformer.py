"""
Crossformer (ICLR 2023) vs GRU-AE  CPU/GPU 벤치마크
Cross-Time + Cross-Dimension 2단계 attention
"""
import time, math, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from collections import deque
sys.path.insert(0, '/home/dyros/mcy_ws')
from train_gru_ae import apply_relative_transform


# ══════════════════════════════════════════════════════════════════
#  Crossformer (Zhang & Yan, ICLR 2023)
#  핵심: 2단계 attention
#   Stage 1 — Cross-Time  : 시간 세그먼트 간 attention
#   Stage 2 — Cross-Dim   : 채널(변수) 간 router-based attention
# ══════════════════════════════════════════════════════════════════
class TwoStageAttention(nn.Module):
    """Cross-Time → Cross-Dimension 순차 적용."""
    def __init__(self, d_model, nhead, num_routers, dropout=0.1):
        super().__init__()
        # Stage 1: Cross-Time attention (시간축)
        self.time_attn = nn.MultiheadAttention(d_model, nhead,
                                               dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Stage 2: Cross-Dim attention (변수축, router 사용)
        self.routers   = nn.Parameter(torch.randn(num_routers, d_model))
        self.dim_attn  = nn.MultiheadAttention(d_model, nhead,
                                               dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff   = nn.Sequential(nn.Linear(d_model, d_model * 4),
                                  nn.GELU(),
                                  nn.Linear(d_model * 4, d_model))
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, D, num_segs, d_model)
        B, D, S, E = x.shape

        # ── Stage 1: Cross-Time (각 채널별로 세그먼트 간 attention)
        x_t = x.reshape(B * D, S, E)
        xt2, _ = self.time_attn(x_t, x_t, x_t)
        x_t    = self.norm1(x_t + self.drop(xt2))
        x = x_t.reshape(B, D, S, E)

        # ── Stage 2: Cross-Dim (각 세그먼트별로 채널 간 router attention)
        x_d   = x.permute(0, 2, 1, 3).reshape(B * S, D, E)   # (B*S, D, E)
        r     = self.routers.unsqueeze(0).expand(B * S, -1, -1)  # (B*S, R, E)
        # router가 각 채널에서 정보 수집 → 다시 채널로 배포
        r2, _ = self.dim_attn(r, x_d, x_d)                    # (B*S, R, E)
        xd2, _= self.dim_attn(x_d, r2, r2)                    # (B*S, D, E)
        x_d   = self.norm2(x_d + self.drop(xd2))
        x_d   = self.norm3(x_d + self.drop(self.ff(x_d)))
        x     = x_d.reshape(B, S, D, E).permute(0, 2, 1, 3)  # (B, D, S, E)
        return x


class CrossformerAE(nn.Module):
    """
    Crossformer Autoencoder.
    인코더: TwoStageAttention 스택 (Cross-Time + Cross-Dim)
    디코더: Linear projection → 원래 시퀀스 복원
    """
    def __init__(self, D=6, T=50, d_model=256, nhead=4,
                 num_layers=3, seg_len=5, num_routers=10, dropout=0.1):
        super().__init__()
        self.D       = D
        self.T       = T
        self.seg_len = seg_len
        self.num_segs = math.ceil(T / seg_len)
        pad_len = self.num_segs * seg_len - T
        self.pad_len  = pad_len

        # 세그먼트 임베딩: seg_len × 1 → d_model  (채널 독립)
        self.seg_embed = nn.Linear(seg_len, d_model)

        # 인코더: TwoStageAttention 스택
        self.layers = nn.ModuleList([
            TwoStageAttention(d_model, nhead, num_routers, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # 디코더: (D, num_segs, d_model) → (T, D)
        self.dec = nn.Linear(d_model, seg_len)

    def forward(self, x):                        # x: (B, T, D)
        B, T, D = x.shape

        # 패딩 (세그먼트 경계 맞춤)
        if self.pad_len > 0:
            x = F.pad(x.transpose(1, 2), (0, self.pad_len),
                      mode='replicate').transpose(1, 2)  # (B, T+pad, D)

        # 세그먼트 분할: (B, D, num_segs, seg_len)
        xp = x.transpose(1, 2).reshape(B, D, self.num_segs, self.seg_len)

        # 임베딩: → (B, D, num_segs, d_model)
        z = self.seg_embed(xp)

        # 인코더
        for layer in self.layers:
            z = layer(z)
        z = self.norm(z)                         # (B, D, num_segs, d_model)

        # 디코더: (B, D, num_segs, seg_len)
        out = self.dec(z)
        # → (B, T+pad, D) → (B, T, D)
        out = out.reshape(B, D, -1).transpose(1, 2)[:, :T, :]
        return out

    def reconstruction_error(self, x):
        return ((x - self(x)) ** 2).mean(dim=(1, 2))


# ── GRU-AE (기준 모델) ─────────────────────────────────────────────
class GRUAE(nn.Module):
    class Enc(nn.Module):
        def __init__(self, i, h, l, d):
            super().__init__()
            self.gru = nn.GRU(i, h, l, batch_first=True, dropout=d if l>1 else 0.)
        def forward(self, x): return self.gru(x)

    class Dec(nn.Module):
        def __init__(self, h, o, l, d, T):
            super().__init__()
            self.T = T
            self.gru = nn.GRU(h, h, l, batch_first=True, dropout=d if l>1 else 0.)
            self.fc  = nn.Linear(h, o)
        def forward(self, h_n):
            inp = h_n[-1].unsqueeze(1).expand(-1, self.T, -1)
            out, _ = self.gru(inp, h_n)
            return self.fc(out)

    def __init__(self, D=6, H=128, L=2, drop=0.2, T=50):
        super().__init__()
        self.enc = self.Enc(D, H, L, drop)
        self.dec = self.Dec(H, D, L, drop, T)

    def forward(self, x):
        _, h = self.enc(x); return self.dec(h)

    def reconstruction_error(self, x):
        return ((x - self(x)) ** 2).mean(dim=(1, 2))


# ── 측정 함수 ──────────────────────────────────────────────────────
def count_params(m): return sum(p.numel() for p in m.parameters())

def measure_cpu(model, D=6, T=50, repeat=200, warmup=30):
    model.eval().cpu()
    scaler = StandardScaler()
    scaler.fit(np.random.randn(500, D).astype(np.float32))
    buf = deque(maxlen=T)
    stream = [np.random.randn(D).astype(np.float32) for _ in range(T + repeat + warmup)]
    for f in stream[:T]: buf.append(f)
    total = 0.
    for i, frame in enumerate(stream[T:]):
        buf.append(frame)
        arr    = np.array(buf, dtype=np.float32)
        scaled = scaler.transform(arr).astype(np.float32)
        x_np   = apply_relative_transform(scaled, 'none')
        x_t    = torch.tensor(x_np).unsqueeze(0)
        t0 = time.perf_counter()
        with torch.no_grad(): model.reconstruction_error(x_t).item()
        if i >= warmup:
            total += (time.perf_counter() - t0) * 1000
    return total / repeat

def measure_gpu(model, D=6, T=50, repeat=200, warmup=30):
    if not torch.cuda.is_available(): return None
    model.eval().cuda()
    x = torch.randn(1, T, D, device='cuda')
    with torch.no_grad():
        for _ in range(warmup): model.reconstruction_error(x)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    with torch.no_grad():
        for _ in range(repeat): model.reconstruction_error(x)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / repeat


# ── 메인 ──────────────────────────────────────────────────────────
D, T = 6, 50
FRAME_MS = 10.0
has_gpu = torch.cuda.is_available()

configs = [
    ("GRU-AE (현재 시스템)",
     GRUAE(D, T=T)),
    ("Crossformer-Small",
     CrossformerAE(D, T, d_model=64,  nhead=4, num_layers=2, seg_len=5, num_routers=5)),
    ("Crossformer-Base",
     CrossformerAE(D, T, d_model=256, nhead=4, num_layers=3, seg_len=5, num_routers=10)),
    ("Crossformer-Large",
     CrossformerAE(D, T, d_model=512, nhead=8, num_layers=4, seg_len=5, num_routers=10)),
]

print(f"GPU: {torch.cuda.get_device_name(0) if has_gpu else 'N/A'}")
print(f"측정 중... (CPU/GPU repeat=200)\n")

rows = []
for name, model in configs:
    n_p = count_params(model)
    mb  = n_p * 4 / 1024 / 1024
    print(f"  [{name}] {n_p/1e3:.1f}K params ...", end=' ', flush=True)
    cpu_ms = measure_cpu(model, D, T)
    gpu_ms = measure_gpu(model, D, T) if has_gpu else None
    rows.append((name, n_p, mb, cpu_ms, gpu_ms))
    g = f"{gpu_ms:.2f}ms" if gpu_ms else "N/A"
    print(f"CPU={cpu_ms:.2f}ms  GPU={g}")

W = 98
print(f"\n{'═'*W}")
print(f"  Crossformer (ICLR 2023) vs GRU-AE  —  파라미터 / 메모리 / 지연시간")
print(f"{'═'*W}")
print(f"  {'모델':<28} {'파라미터':>9} {'크기':>7}  {'CPU':>9}  {'CPU 부하':>8}  {'GPU':>9}  {'실시간':>6}")
print(f"{'─'*W}")
for name, n_p, mb, cpu_ms, gpu_ms in rows:
    p_str = f"{n_p/1e6:.2f}M" if n_p >= 1e6 else f"{n_p/1e3:.0f}K"
    load  = f"{cpu_ms/FRAME_MS*100:.0f}%"
    g     = f"{gpu_ms:.2f}ms" if gpu_ms else "N/A"
    rt    = "✓" if cpu_ms < FRAME_MS else "✗"
    print(f"  {name:<28} {p_str:>9} {mb:>5.1f}MB  {cpu_ms:>7.2f}ms  {load:>8}  {g:>9}  {rt:>6}")
print(f"{'═'*W}")
print(f"\n  ※ CPU 부하: 100Hz(10ms/frame) 기준.")
print(f"  ※ Crossformer 설정: seg_len=5 → {math.ceil(T/5)}세그먼트 / seq_len={T}")
print(f"  ※ Cross-Time(시간축) + Cross-Dim(채널축) 2단계 attention, 비인과(전체 시퀀스 필요).\n")
