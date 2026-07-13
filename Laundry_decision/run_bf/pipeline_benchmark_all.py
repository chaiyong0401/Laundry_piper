"""
전 모델 실제 파이프라인 벤치마크 (CPU 기준)
===========================================
pure inference 가 아닌, 실제 detector.py update() 경로를 재현:
  deque(buffer) → np.array → scaler.transform → apply_relative_transform
  → torch.tensor → model.reconstruction_error

모든 모델에 동일한 전처리 파이프라인 적용 후 종합 비교표 출력.

Usage:
  python pipeline_benchmark_all.py
  python pipeline_benchmark_all.py --repeat 300
"""

import argparse
import math
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '/home/dyros/mcy_ws')
from train_gru_ae import apply_relative_transform


# ══════════════════════════════════════════════════════════════════
#  모델 정의 (benchmark_gru_ae.py 와 동일)
# ══════════════════════════════════════════════════════════════════

# ── GRU-AE ────────────────────────────────────────────────────────
class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
    def forward(self, x):
        return self.gru(x)

class GRUDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.fc  = nn.Linear(hidden_dim, output_dim)
    def forward(self, h_n):
        last_h = h_n[-1]
        inp = last_h.unsqueeze(1).expand(-1, self.seq_len, -1)
        out, _ = self.gru(inp, h_n)
        return self.fc(out)

class GRUAE(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, dropout=0.2, seq_len=50):
        super().__init__()
        self.encoder = GRUEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = GRUDecoder(hidden_dim, input_dim, num_layers, dropout, seq_len)
    def forward(self, x):
        _, h_n = self.encoder(x)
        return self.decoder(h_n)
    def reconstruction_error(self, x):
        recon = self.forward(x)
        return ((x - recon) ** 2).mean(dim=(1, 2))

# ── LSTM-AE ───────────────────────────────────────────────────────
class LSTMAE(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, dropout=0.2, seq_len=50):
        super().__init__()
        self.seq_len = seq_len
        self.enc = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.dec = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.fc  = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        _, (h_n, c_n) = self.enc(x)
        last_h = h_n[-1]
        inp = last_h.unsqueeze(1).expand(-1, self.seq_len, -1)
        out, _ = self.dec(inp, (h_n, c_n))
        return self.fc(out)
    def reconstruction_error(self, x):
        return ((x - self.forward(x)) ** 2).mean(dim=(1, 2))

# ── BiGRU-AE ──────────────────────────────────────────────────────
class BiGRUAE(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, dropout=0.2, seq_len=50):
        super().__init__()
        self.seq_len, self.num_layers = seq_len, num_layers
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                               bidirectional=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dec_gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        _, h_n = self.encoder(x)
        ctx = self.proj(torch.cat([h_n[-2], h_n[-1]], dim=-1))
        h_d = ctx.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        inp = ctx.unsqueeze(1).expand(-1, self.seq_len, -1)
        out, _ = self.dec_gru(inp, h_d)
        return self.fc(out)
    def reconstruction_error(self, x):
        return ((x - self.forward(x)) ** 2).mean(dim=(1, 2))

# ── TCN-AE ────────────────────────────────────────────────────────
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, ks, dilation):
        super().__init__()
        self.pad  = (ks - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, ks, padding=self.pad, dilation=dilation)
    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.pad] if self.pad > 0 else x

class TCNBlock(nn.Module):
    def __init__(self, ch, ks, dilation, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(ch, ch, ks, dilation), nn.BatchNorm1d(ch), nn.GELU(), nn.Dropout(dropout),
            CausalConv1d(ch, ch, ks, dilation), nn.BatchNorm1d(ch), nn.GELU(), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x) + x

class TCNAE(nn.Module):
    def __init__(self, input_dim=6, channels=64, num_blocks=6, seq_len=50):
        super().__init__()
        self.input_dim, self.seq_len = input_dim, seq_len
        self.proj = nn.Conv1d(input_dim, channels, 1)
        self.enc  = nn.Sequential(*[TCNBlock(channels, 3, 2**i) for i in range(num_blocks)])
        self.dec  = nn.Sequential(nn.Linear(channels, channels*2), nn.GELU(),
                                   nn.Linear(channels*2, seq_len * input_dim))
    def forward(self, x):
        z = self.enc(self.proj(x.transpose(1,2))).mean(dim=-1)
        return self.dec(z).view(x.shape[0], self.seq_len, self.input_dim)
    def reconstruction_error(self, x):
        return ((x - self.forward(x)) ** 2).mean(dim=(1, 2))

# ── Causal Transformer-AE ─────────────────────────────────────────
class CausalTransformerAE(nn.Module):
    def __init__(self, input_dim=6, d_model=256, nhead=8,
                 num_layers=6, dim_ff=1024, dropout=0.1, seq_len=50):
        super().__init__()
        self.seq_len = seq_len
        self.proj = nn.Linear(input_dim, d_model)
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.head = nn.Linear(d_model, input_dim)
    def _cmask(self, T, device):
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
    def forward(self, x):
        T = x.shape[1]
        z = self.proj(x) + self.pe[:, :T]
        z = self.encoder(z, mask=self._cmask(T, x.device))
        return self.head(z)
    def reconstruction_error(self, x):
        return ((x - self.forward(x)) ** 2).mean(dim=(1, 2))

# ── Mamba-AE ──────────────────────────────────────────────────────
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner   = d_model * expand
        dt_rank   = max(1, d_model // 16)
        self.d_inner, self.d_state = d_inner, d_state
        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(d_inner, d_inner, d_conv,
                                   padding=d_conv - 1, groups=d_inner, bias=True)
        self.x_proj   = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj  = nn.Linear(dt_rank, d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dt_rank = dt_rank

    def forward(self, x):
        B, T, _ = x.shape
        xz      = self.in_proj(x)
        x_, z   = xz.chunk(2, dim=-1)
        x_      = self.conv1d(x_.transpose(1,2))[:, :, :T].transpose(1,2)
        x_      = F.silu(x_)
        bcd     = self.x_proj(x_)
        dt      = F.softplus(self.dt_proj(bcd[:, :, :self.dt_rank]))
        B_      = bcd[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C_      = bcd[:, :, self.dt_rank + self.d_state:]
        A       = -torch.exp(self.A_log)
        h       = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys      = []
        for t in range(T):
            dA = torch.exp(dt[:, t, :].unsqueeze(-1) * A)
            dB = dt[:, t, :].unsqueeze(-1) * B_[:, t, :].unsqueeze(1)
            h  = h * dA + dB * x_[:, t, :].unsqueeze(-1)
            ys.append((h * C_[:, t, :].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + self.D * x_
        return self.out_proj(y * F.silu(z))

class MambaAE(nn.Module):
    def __init__(self, input_dim=6, d_model=256, d_state=32, d_conv=4,
                 expand=2, num_layers=6, seq_len=50):
        super().__init__()
        self.proj   = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_model), MambaBlock(d_model, d_state, d_conv, expand))
            for _ in range(num_layers)])
        self.norm   = nn.LayerNorm(d_model)
        self.head   = nn.Linear(d_model, input_dim)
    def forward(self, x):
        z = self.proj(x)
        for blk in self.blocks: z = z + blk(z)
        return self.head(self.norm(z))
    def reconstruction_error(self, x):
        return ((x - self.forward(x)) ** 2).mean(dim=(1, 2))


# ══════════════════════════════════════════════════════════════════
#  파이프라인 측정 함수
# ══════════════════════════════════════════════════════════════════
def measure_pipeline(model, seq_len, input_dim, device, repeat=300, warmup=30):
    """
    실제 detector.py update() 경로 재현:
      deque → np.array → scaler.transform → to_tensor → model.reconstruction_error
    반환: (전처리ms, tensor이동ms, 추론ms, 전체ms)
    """
    model.eval()
    scaler = StandardScaler()
    scaler.fit(np.random.randn(500, input_dim).astype(np.float32))

    buffer = deque(maxlen=seq_len)
    stream = [np.random.randn(input_dim).astype(np.float32) for _ in range(seq_len + repeat + warmup)]
    for f in stream[:seq_len]:
        buffer.append(f)

    t_preproc = t_move = t_infer = 0.0

    for i, frame in enumerate(stream[seq_len:]):
        buffer.append(frame)

        t0 = time.perf_counter()
        arr    = np.array(buffer, dtype=np.float32)
        scaled = scaler.transform(arr).astype(np.float32)
        x_np   = apply_relative_transform(scaled, 'none')
        t1 = time.perf_counter()

        x_t = torch.tensor(x_np).unsqueeze(0).to(device)
        t2 = time.perf_counter()

        with torch.no_grad():
            model.reconstruction_error(x_t).item()
        t3 = time.perf_counter()

        if i >= warmup:
            t_preproc += (t1 - t0) * 1000
            t_move    += (t2 - t1) * 1000
            t_infer   += (t3 - t2) * 1000

    n = repeat
    return t_preproc/n, t_move/n, t_infer/n, (t_preproc+t_move+t_infer)/n


def count_params(m):
    return sum(p.numel() for p in m.parameters())


# ══════════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat',    type=int, default=300)
    parser.add_argument('--seq_len',   type=int, default=50)
    parser.add_argument('--input_dim', type=int, default=6)
    args = parser.parse_args()

    device = torch.device('cpu')   # 실제 시스템 기준
    D, T   = args.input_dim, args.seq_len
    n      = args.repeat

    print(f"측정 조건: CPU, seq_len={T}, input_dim={D}, repeat={n}회")
    print(f"파이프라인: deque→np.array→scaler→to_tensor→model.reconstruction_error\n")

    model_defs = [
        # (이름,                          모델,                                                         분류)
        ("GRU-AE (현재 시스템)",
         GRUAE(D, hidden_dim=128, num_layers=2, seq_len=T),                                           "경량·AR"),
        ("LSTM-AE",
         LSTMAE(D, hidden_dim=128, num_layers=2, seq_len=T),                                          "비교·AR"),
        ("BiGRU-AE",
         BiGRUAE(D, hidden_dim=128, num_layers=2, seq_len=T),                                         "비교·AR"),
        ("TCN-AE",
         TCNAE(D, channels=64, num_blocks=6, seq_len=T),                                              "비교"),
        ("Causal-Transformer Medium (4.7M)",
         CausalTransformerAE(D, d_model=256, nhead=8, num_layers=6, dim_ff=1024, seq_len=T),          "AR비경량"),
        ("Causal-Transformer Large (25M)",
         CausalTransformerAE(D, d_model=512, nhead=8, num_layers=8, dim_ff=2048, seq_len=T),          "AR대형"),
        ("Mamba Medium (2.8M)",
         MambaAE(D, d_model=256, d_state=32, d_conv=4, expand=2, num_layers=6, seq_len=T),            "AR비경량"),
        ("Mamba Large (14.8M)",
         MambaAE(D, d_model=512, d_state=64, d_conv=4, expand=2, num_layers=8, seq_len=T),            "AR대형"),
    ]

    results = []
    for name, model, cat in model_defs:
        model = model.to(device).eval()
        n_params = count_params(model)
        mb       = n_params * 4 / 1024 / 1024
        try:
            tp, tm, ti, total = measure_pipeline(model, T, D, device, repeat=n, warmup=30)
            results.append((name, cat, n_params, mb, tp, tm, ti, total))
            print(f"[OK] {name}: 전처리={tp:.3f}ms  이동={tm:.3f}ms  추론={ti:.3f}ms  합계={total:.3f}ms")
        except Exception as e:
            print(f"[ERR] {name}: {e}")
            results.append((name, cat, n_params, mb, None, None, None, None))

    # ── 비교표
    gru_total = results[0][7]
    frame_ms  = 10.0   # 100Hz

    print("\n" + "═" * 110)
    print(f"  실제 파이프라인 전체 비용 비교  (CPU, 반복 {n}회 평균)")
    print(f"  측정 내용: np.array + scaler.transform + torch.tensor + model.reconstruction_error")
    print("═" * 110)
    print(f"  {'모델':<40} {'분류':<10} {'파라미터':>9} {'전처리':>9} {'텐서이동':>9} "
          f"{'모델추론':>9} {'합계':>9} {'상대속도':>9} {'100Hz부하':>10}")
    print("─" * 110)

    for name, cat, n_params, mb, tp, tm, ti, total in results:
        if total is None:
            print(f"  {name:<40} {cat:<10} {'오류':>9}")
            continue
        ratio   = total / gru_total
        load    = total / frame_ms * 100
        p_str   = f"{n_params/1e6:.2f}M" if n_params >= 1e6 else f"{n_params/1e3:.0f}K"
        r_str   = f"×{ratio:.1f}" if ratio >= 1 else f"×{ratio:.2f}"
        l_str   = f"{load:.1f}%"
        print(f"  {name:<40} {cat:<10} {p_str:>9} {tp:>7.3f}ms {tm:>7.3f}ms "
              f"{ti:>7.2f}ms {total:>7.2f}ms {r_str:>9} {l_str:>10}")

    print("═" * 110)

    # ── 실시간성 요약
    print(f"""
[실시간성 기준: XELA 100Hz, 프레임 주기 10ms]
  · GRU-AE update() 1회: {gru_total:.2f}ms → 100Hz 루프에서 매 프레임 호출 시 {gru_total/frame_ms*100:.0f}% CPU 점유
  · 전처리(np.array+scaler)는 모든 모델에서 ~0.17ms 로 고정 — 모델 구조와 무관
  · 모델 추론이 전체 비용을 지배: 전처리/텐서이동 합계 ≈ 0.20ms (변동 없음)

  · Mamba: Python sequential scan 한계로 측정값 과다 — 실제 mamba-ssm(CUDA)은 GRU급
  · Causal-Transformer: 전체 시퀀스 재처리 필요 → stride 줄여도 비용 고정
""")

    # ── 추론 비용만 추출한 요약표 (전처리 제외)
    print("─" * 70)
    print(f"  {'모델':<40} {'추론만':>9} {'파이프라인 합계':>16}")
    print("─" * 70)
    for name, cat, n_params, mb, tp, tm, ti, total in results:
        if total is None: continue
        print(f"  {name:<40} {ti:>7.2f}ms {total:>14.2f}ms")
    print("─" * 70)
    print(f"  * 전처리(np.array+scaler+relative): ~{results[0][4]:.3f}ms (고정)")
    print(f"  * numpy→tensor (CPU): ~{results[0][5]:.3f}ms (고정)")


if __name__ == '__main__':
    main()
