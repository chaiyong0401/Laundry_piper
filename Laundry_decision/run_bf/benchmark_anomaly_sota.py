"""
시계열 이상감지 SOTA 모델 벤치마크 (CPU + GPU)
===============================================
실제 파이프라인 비용: np.array → scaler → to_tensor → model.reconstruction_error

모델 목록:
  [현재 시스템]
  · GRU-AE       — 경량 AR, hidden=128, 2레이어

  [유사 규모 비교]
  · LSTM-AE      — AR, LSTM 게이트 (CPU에서 GRU보다 빠름: fused GEMM)
  · TCN-AE       — 인과 팽창 합성곱, 파라미터 효율 높음 (Conv: O(C²·k) vs RNN: O(H²))

  [SOTA 이상감지 — causal/AR]
  · VRNN         — OmniAnomaly 구조 (KDD 2019), Variational RNN
                   다변량 센서 이상감지 최초 SOTA, 순차적·인과적
  · Anomaly-Transformer — ICLR 2022 SOTA
                   Association Discrepancy로 이상 감지
                   ※ 전체 시퀀스 attention 사용 (비인과)

  [최신 SOTA — 2022~2024]
  · TranAD       — VLDB 2022, Two-branch Transformer + 적대적 자기조건화
                   ※ 비인과 (전체 시퀀스 attention)
  · TimesNet     — ICLR 2023, 1D→2D 변환 후 2D Conv
                   ※ 전체 시퀀스 필요 (비인과)
  · ModernTCN    — 2024, 대형 커널 TCN (receptive field↑, 파라미터↓)
                   인과 합성곱 유지 가능

  [인과 비교군]
  · Causal Transformer-AE — GPT-decoder 스타일, causal mask
                             GRU-AE 성능 상한선 참고용

Usage:
  python benchmark_anomaly_sota.py
  python benchmark_anomaly_sota.py --repeat 300
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
#  1. GRU-AE (현재 시스템)
# ══════════════════════════════════════════════════════════════════
class GRUAE(nn.Module):
    """현재 시스템 — 경량 GRU Autoencoder."""
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
        _, h = self.enc(x)
        return self.dec(h)

    def reconstruction_error(self, x):
        return ((x - self(x)) ** 2).mean(dim=(1, 2))


# ══════════════════════════════════════════════════════════════════
#  2. LSTM-AE
# ══════════════════════════════════════════════════════════════════
class LSTMAE(nn.Module):
    """LSTM-AE — CPU에서 fused GEMM으로 GRU보다 빠름."""
    def __init__(self, D=6, H=128, L=2, drop=0.2, T=50):
        super().__init__()
        self.T = T
        self.enc = nn.LSTM(D, H, L, batch_first=True, dropout=drop if L>1 else 0.)
        self.dec = nn.LSTM(H, H, L, batch_first=True, dropout=drop if L>1 else 0.)
        self.fc  = nn.Linear(H, D)

    def forward(self, x):
        _, (h, c) = self.enc(x)
        inp = h[-1].unsqueeze(1).expand(-1, self.T, -1)
        out, _ = self.dec(inp, (h, c))
        return self.fc(out)

    def reconstruction_error(self, x):
        return ((x - self(x)) ** 2).mean(dim=(1, 2))


# ══════════════════════════════════════════════════════════════════
#  3. TCN-AE
#  ─ 인과 팽창 합성곱 (WaveNet 스타일)
#  ─ 파라미터: Conv O(C²·k) — RNN O(H²) 보다 효율적
#  ─ receptive field = (k-1) × Σ(2^i) = 2 × 63 = 126프레임 (k=3, 6블록)
# ══════════════════════════════════════════════════════════════════
class TCN_AE(nn.Module):
    """
    TCN Autoencoder.
    [왜 파라미터가 적은가]
    RNN은 hidden↔hidden 재귀 연결 O(H²)이 지배.
    Conv는 커널 크기에만 의존: O(C² × kernel_size).
    channels=64, kernel=3, 6블록 → 약 197K params.
    GRU 350K보다 작지만 수용 영역(126프레임)은 충분.
    """
    class CausalConv(nn.Module):
        def __init__(self, c, k, d):
            super().__init__()
            self.pad  = (k - 1) * d
            self.conv = nn.Conv1d(c, c, k, padding=self.pad, dilation=d)
        def forward(self, x):
            o = self.conv(x)
            return o[:, :, :-self.pad] if self.pad else o

    class Block(nn.Module):
        def __init__(self, c, k, d):
            super().__init__()
            self.net = nn.Sequential(
                TCN_AE.CausalConv(c, k, d), nn.BatchNorm1d(c), nn.GELU(),
                TCN_AE.CausalConv(c, k, d), nn.BatchNorm1d(c), nn.GELU(),
            )
        def forward(self, x): return self.net(x) + x

    def __init__(self, D=6, C=64, k=3, nb=6, T=50):
        super().__init__()
        self.D, self.T = D, T
        self.proj = nn.Conv1d(D, C, 1)
        self.enc  = nn.Sequential(*[self.Block(C, k, 2**i) for i in range(nb)])
        self.dec  = nn.Sequential(nn.Linear(C, C*2), nn.GELU(),
                                   nn.Linear(C*2, T*D))

    def forward(self, x):
        z = self.enc(self.proj(x.transpose(1,2))).mean(-1)
        return self.dec(z).view(x.shape[0], self.T, self.D)

    def reconstruction_error(self, x):
        return ((x - self(x)) ** 2).mean(dim=(1, 2))


# ══════════════════════════════════════════════════════════════════
#  4. VRNN — OmniAnomaly 구조 (KDD 2019)
#  ─ Variational RNN: 각 타임스텝에서 잠재 변수 z_t 샘플링
#  ─ Prior: p(z_t | h_{t-1}),  Posterior: q(z_t | x_t, h_{t-1})
#  ─ Decoder: p(x̂_t | z_t, h_{t-1})
#  ─ Anomaly score = 재구성 오차 + KL divergence
#  ─ 완전히 인과적 (과거 h와 현재 x만 사용)
# ══════════════════════════════════════════════════════════════════
class VRNN(nn.Module):
    """
    Variational RNN (OmniAnomaly 구조, KDD 2019).
    다변량 센서 데이터 이상감지 SOTA.
    각 타임스텝: prior → posterior → z 샘플링 → 재구성 → 은닉 갱신
    """
    def __init__(self, D=6, H=128, Z=32, L=1, T=50):
        super().__init__()
        self.D, self.H, self.Z, self.T = D, H, Z, T

        # Feature extractor
        self.phi_x = nn.Sequential(nn.Linear(D, H), nn.GELU())
        self.phi_z = nn.Sequential(nn.Linear(Z, H), nn.GELU())

        # Prior: p(z_t | h_{t-1})
        self.prior_mu  = nn.Linear(H, Z)
        self.prior_log = nn.Linear(H, Z)

        # Encoder: q(z_t | x_t, h_{t-1})
        self.enc_mu  = nn.Linear(H + H, Z)
        self.enc_log = nn.Linear(H + H, Z)

        # Decoder: p(x̂_t | z_t, h_{t-1})
        self.dec_mu  = nn.Linear(H + H, D)

        # RNN: h_t = GRU([phi_x(x_t), phi_z(z_t)], h_{t-1})
        self.rnn = nn.GRUCell(H + H, H)

    def _reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):                      # x: (B, T, D)
        B, T, _ = x.shape
        h = torch.zeros(B, self.H, device=x.device)
        recons = []

        for t in range(T):
            x_t  = x[:, t, :]                                   # (B, D)
            px   = self.phi_x(x_t)                              # (B, H)

            # Prior (inference 시에는 사용 안 함, ELBO 학습 시 사용)
            # Encoder posterior
            enc_in    = torch.cat([px, h], dim=-1)              # (B, 2H)
            z_mu      = self.enc_mu(enc_in)
            z_log     = self.enc_log(enc_in)
            z_t       = self._reparameterize(z_mu, z_log)       # (B, Z)
            pz        = self.phi_z(z_t)                         # (B, H)

            # Decoder
            dec_in    = torch.cat([pz, h], dim=-1)              # (B, 2H)
            x_hat_t   = self.dec_mu(dec_in)                     # (B, D)
            recons.append(x_hat_t)

            # RNN state update
            rnn_in = torch.cat([px, pz], dim=-1)                # (B, 2H)
            h      = self.rnn(rnn_in, h)

        return torch.stack(recons, dim=1)                        # (B, T, D)

    def reconstruction_error(self, x):
        return ((x - self(x)) ** 2).mean(dim=(1, 2))


# ══════════════════════════════════════════════════════════════════
#  5. Anomaly Transformer (ICLR 2022)
#  ─ 가장 인용된 시계열 이상감지 SOTA
#  ─ 핵심: Association Discrepancy
#         Prior-Association (Gaussian kernel) vs
#         Series-Association (learned attention)
#         두 분포의 KL divergence → 이상 점수
#  ─ ※ 전체 시퀀스 attention 사용 (비인과) — 오프라인/배치 탐지에 적합
# ══════════════════════════════════════════════════════════════════
class AnomalyTransformer(nn.Module):
    """
    Anomaly Transformer (Xu et al., ICLR 2022).
    Association Discrepancy 기반 이상감지 SOTA.
    Prior-Association: Gaussian kernel attention (위치 기반)
    Series-Association: learned multi-head attention (패턴 기반)
    이상 점수 = 재구성 오차 + 두 association 간 KL divergence
    """
    class AnomalyAttention(nn.Module):
        def __init__(self, d_model, nhead, seq_len):
            super().__init__()
            self.nhead   = nhead
            self.d_head  = d_model // nhead
            self.seq_len = seq_len
            self.qkv_proj = nn.Linear(d_model, 3 * d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            # σ: Gaussian prior의 폭 (학습 가능)
            self.sigma    = nn.Parameter(torch.ones(1, nhead, 1, 1))

        def _prior_assoc(self, T, device):
            """Gaussian kernel attention: P_{ij} = G(|i-j|, σ)"""
            pos = torch.arange(T, device=device, dtype=torch.float)
            dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # (T, T)
            sigma = torch.abs(self.sigma) + 1e-5                # (1, H, 1, 1)
            prior = torch.exp(-dist**2 / (2 * sigma**2))        # (1, H, T, T)
            prior = prior / (prior.sum(-1, keepdim=True) + 1e-9)
            return prior

        def forward(self, x):
            B, T, D = x.shape
            qkv = self.qkv_proj(x).reshape(B, T, 3, self.nhead, self.d_head)
            q, k, v = qkv.unbind(2)                             # each (B, T, H, d)
            q = q.transpose(1, 2)                               # (B, H, T, d)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Series-Association (learned)
            scale  = math.sqrt(self.d_head)
            series = F.softmax(q @ k.transpose(-2,-1) / scale, dim=-1)  # (B, H, T, T)

            # Prior-Association (Gaussian)
            prior  = self._prior_assoc(T, x.device).expand(B, -1, -1, -1)

            # Output
            out = (series @ v).transpose(1, 2).reshape(B, T, D)
            out = self.out_proj(out)

            # Association discrepancy (KL: series || prior)
            assoc_disc = F.kl_div(
                (series + 1e-9).log(), prior + 1e-9,
                reduction='none').sum(-1).mean()                 # scalar

            return out, assoc_disc

    class EncoderLayer(nn.Module):
        def __init__(self, d_model, nhead, dim_ff, seq_len, dropout=0.1):
            super().__init__()
            self.attn = AnomalyTransformer.AnomalyAttention(d_model, nhead, seq_len)
            self.ff   = nn.Sequential(
                nn.Linear(d_model, dim_ff), nn.GELU(), nn.Linear(dim_ff, d_model))
            self.ln1  = nn.LayerNorm(d_model)
            self.ln2  = nn.LayerNorm(d_model)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            attn_out, disc = self.attn(x)
            x = self.ln1(x + self.drop(attn_out))
            x = self.ln2(x + self.drop(self.ff(x)))
            return x, disc

    def __init__(self, D=6, d_model=128, nhead=4, num_layers=3,
                 dim_ff=256, dropout=0.1, T=50):
        super().__init__()
        self.proj   = nn.Linear(D, d_model)
        pe = torch.zeros(T, d_model)
        pos = torch.arange(T, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.layers = nn.ModuleList([
            self.EncoderLayer(d_model, nhead, dim_ff, T, dropout)
            for _ in range(num_layers)])
        self.head   = nn.Linear(d_model, D)

    def forward(self, x):
        z = self.proj(x) + self.pe
        total_disc = 0.
        for layer in self.layers:
            z, disc = layer(z)
            total_disc = total_disc + disc
        return self.head(z), total_disc

    def reconstruction_error(self, x):
        recon, disc = self(x)
        recon_err = ((x - recon) ** 2).mean(dim=(1, 2))
        # 논문: score = recon_err - λ·disc (disc가 높으면 이상)
        # 벤치마크용으로는 recon_err만 반환 (구조 포함)
        return recon_err


# ══════════════════════════════════════════════════════════════════
#  6. TimesNet (ICLR 2023)
#  ─ 1D 시계열을 2D로 재구성 후 2D Conv 적용
#  ─ 핵심 아이디어: 시계열의 주기성을 2D 공간에서 캡처
#  ─ ※ 전체 시퀀스 필요 (비인과), 배치 처리에 적합
# ══════════════════════════════════════════════════════════════════
class TimesNet(nn.Module):
    """
    TimesNet (Wu et al., ICLR 2023).
    1D 시계열 → 2D reshape → 2D Conv으로 패턴 캡처.
    시계열의 주기 p를 FFT로 추정 → (T/p, p) 2D 텐서로 변환.
    다양한 시간 스케일의 변화를 동시에 모델링.
    ※ 비인과: 전체 시퀀스에서 FFT → 실시간 사용 불가
    """
    class Inception2D(nn.Module):
        """TimesBlock의 2D 합성곱 모듈."""
        def __init__(self, d_model, dim_ff):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(d_model, dim_ff, 3, padding=1), nn.GELU(),
                nn.Conv2d(dim_ff,  d_model, 3, padding=1),
            )
        def forward(self, x): return self.conv(x)  # (B, C, H, W)

    def __init__(self, D=6, d_model=64, dim_ff=128, num_layers=3,
                 top_k=3, T=50, dropout=0.1):
        super().__init__()
        self.T      = T
        self.top_k  = top_k
        self.d_model = d_model
        self.proj   = nn.Linear(D, d_model)
        self.blocks = nn.ModuleList([self.Inception2D(d_model, dim_ff)
                                     for _ in range(num_layers)])
        self.norms  = nn.ModuleList([nn.LayerNorm(d_model)
                                     for _ in range(num_layers)])
        self.head   = nn.Linear(d_model, D)

    def _top_k_periods(self, x):
        """FFT로 상위 top_k 주기 추출."""
        # x: (B, T, d_model)
        xf = torch.fft.rfft(x, dim=1)                          # (B, T//2+1, d)
        freq_amp = xf.abs().mean(dim=-1)                        # (B, T//2+1)
        # DC 제거
        freq_amp[:, 0] = 0.
        _, top_idx = torch.topk(freq_amp, self.top_k, dim=-1)  # (B, top_k)
        periods = self.T // (top_idx + 1)                       # 주기 길이
        periods = periods.clamp(min=2)
        return periods                                           # (B, top_k)

    def forward(self, x):                                       # x: (B, T, D)
        B, T, D = x.shape
        z = self.proj(x)                                        # (B, T, d_model)

        for blk, norm in zip(self.blocks, self.norms):
            periods = self._top_k_periods(z)                    # (B, top_k)
            res = torch.zeros_like(z)

            for k in range(self.top_k):
                p = int(periods[:, k].float().mean().item())
                p = max(2, p)
                # 패딩하여 p의 배수로 맞춤
                pad_len = math.ceil(T / p) * p - T
                z_pad = F.pad(z.transpose(1, 2), (0, pad_len))  # (B, d, T+pad)
                H2 = (T + pad_len) // p
                # 2D reshape: (B, d, H2, p)
                z2d = z_pad.reshape(B, self.d_model, H2, p)
                # 2D conv
                z2d = blk(z2d)
                # 다시 1D로
                z1d = z2d.reshape(B, self.d_model, -1)[:, :, :T]  # (B, d, T)
                res = res + z1d.transpose(1, 2)

            res = res / self.top_k
            z   = norm(z + res)

        return self.head(z)                                     # (B, T, D)

    def reconstruction_error(self, x):
        return ((x - self(x)) ** 2).mean(dim=(1, 2))


# ══════════════════════════════════════════════════════════════════
#  7. ModernTCN (2024)
#  ─ 대형 커널(large kernel) + Depthwise 합성곱
#  ─ 큰 커널로 넓은 receptive field를 적은 파라미터로 달성
#  ─ kernel=51: receptive field ≈ (51-1)×1 = 50프레임 (단일 블록)
#  ─ TCN-AE(dilated) 와 달리 단일 패스로 전체 시퀀스 커버
# ══════════════════════════════════════════════════════════════════
class ModernTCN(nn.Module):
    """
    ModernTCN (Luo & Wang, 2024).
    대형 Depthwise 커널로 넓은 시간 수용 영역을 파라미터 효율적으로 구현.
    Block: DWConv(large k) → LayerNorm → PointwiseConv(FFN)
    인과 패딩 적용 시 real-time 사용 가능.
    """
    class Block(nn.Module):
        def __init__(self, d_model, k, dim_ff, causal=True):
            super().__init__()
            self.causal = causal
            self.pad_size = k - 1 if causal else (k - 1) // 2
            # Depthwise large-kernel conv
            self.dw_conv = nn.Conv1d(d_model, d_model, k,
                                      padding=self.pad_size if not causal else 0,
                                      groups=d_model)
            # Pointwise FFN
            self.pw_conv = nn.Sequential(
                nn.Conv1d(d_model, dim_ff, 1), nn.GELU(),
                nn.Conv1d(dim_ff, d_model, 1),
            )
            self.ln = nn.LayerNorm(d_model)

        def forward(self, x):              # x: (B, C, T)
            if self.causal:
                x_pad = F.pad(x, (self.pad_size, 0))
                z = self.dw_conv(x_pad)
            else:
                z = self.dw_conv(x)
            # LayerNorm on (B, C, T): apply over C dim
            z = self.ln(z.transpose(1, 2)).transpose(1, 2)
            z = self.pw_conv(z)
            return x + z

    def __init__(self, D=6, d_model=64, kernel_size=51,
                 dim_ff=128, num_layers=3, T=50, causal=True):
        super().__init__()
        self.D, self.T = D, T
        self.proj   = nn.Conv1d(D, d_model, 1)
        self.blocks = nn.Sequential(*[
            self.Block(d_model, kernel_size, dim_ff, causal)
            for _ in range(num_layers)])
        self.dec    = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(),
                                     nn.Linear(dim_ff, T * D))

    def forward(self, x):
        z = self.proj(x.transpose(1, 2))    # (B, d, T)
        z = self.blocks(z)
        z_pool = z.mean(-1)                 # (B, d)
        return self.dec(z_pool).view(x.shape[0], self.T, self.D)

    def reconstruction_error(self, x):
        return ((x - self(x)) ** 2).mean(dim=(1, 2))


# ══════════════════════════════════════════════════════════════════
#  8. TranAD (Tuli et al., VLDB 2022)
#  ─ Two-branch Transformer + 자기조건화(self-conditioning)
#  ─ Phase 1: W1이 x를 재구성 → x̂₁
#  ─ Phase 2: W2가 x̂₁을 피드백으로 받아 x를 재구성 → x̂₂ (적대적 정제)
#  ─ Anomaly score = ||x - x̂₁|| + ||x - x̂₂||
#  ─ 비인과: 전체 시퀀스 attention 사용
# ══════════════════════════════════════════════════════════════════
class TranAD(nn.Module):
    """
    TranAD (Tuli et al., VLDB 2022).
    Two-branch Transformer with adversarial self-conditioning.
    W1은 원본 재구성, W2는 W1 출력을 조건으로 적대적 정제.
    학습 시 두 phase를 번갈아 최적화.
    Anomaly score = α·||x−x̂₁|| + (1−α)·||x−x̂₂||
    """
    def _pe(self, T, d, device):
        pe  = torch.zeros(T, d, device=device)
        pos = torch.arange(T, dtype=torch.float, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float, device=device)
                        * (-math.log(10000.) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)                             # (1, T, d)

    def __init__(self, D=6, d_model=128, nhead=4,
                 num_layers=2, dim_ff=256, dropout=0.1, T=50):
        super().__init__()
        self.D, self.T = D, T

        self.input_proj  = nn.Linear(D, d_model)

        # W1: 재구성 branch
        e1 = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        d1 = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.w1_enc = nn.TransformerEncoder(e1, num_layers)
        self.w1_dec = nn.TransformerDecoder(d1, num_layers)
        self.out1   = nn.Linear(d_model, D)

        # W2: 적대적 정제 branch
        e2 = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        d2 = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.w2_enc = nn.TransformerEncoder(e2, num_layers)
        self.w2_dec = nn.TransformerDecoder(d2, num_layers)
        self.out2   = nn.Linear(d_model, D)

    def forward(self, x):                                  # x: (B, T, D)
        pe   = self._pe(x.shape[1], self.input_proj.out_features, x.device)
        z    = self.input_proj(x) + pe                    # (B, T, d_model)

        # Phase 1: W1 인코더-디코더
        mem1  = self.w1_enc(z)
        dec1  = self.w1_dec(z, mem1)
        x_hat1 = self.out1(dec1)                          # (B, T, D)

        # Phase 2: x̂₁을 조건으로 W2 정제
        # 추론 시 epoch→∞ → 자기조건화 입력 = x̂₁
        z2    = self.input_proj(x_hat1.detach()) + pe
        mem2  = self.w2_enc(z2)
        dec2  = self.w2_dec(z, mem2)
        x_hat2 = self.out2(dec2)                          # (B, T, D)

        return x_hat1, x_hat2

    def reconstruction_error(self, x):
        x_hat1, x_hat2 = self(x)
        e1 = ((x - x_hat1) ** 2).mean(dim=(1, 2))
        e2 = ((x - x_hat2) ** 2).mean(dim=(1, 2))
        return (e1 + e2) / 2


# ══════════════════════════════════════════════════════════════════
#  9. Causal Transformer-AE (GPT-decoder 스타일)
#  ─ causal mask로 t번째 출력은 0..t-1 만 참조
#  ─ GRU와 동일한 auto-regressive 인과성, 더 무거운 대안
# ══════════════════════════════════════════════════════════════════
class CausalTransformerAE(nn.Module):
    """
    Causal Transformer AE (GPT-decoder 스타일).
    상삼각 causal mask 적용: t번째 출력이 0..t 정보만 사용.
    GRU-AE와 동일한 인과성 보장, 더 많은 파라미터.
    """
    def __init__(self, D=6, d_model=128, nhead=4,
                 num_layers=4, dim_ff=256, dropout=0.1, T=50):
        super().__init__()
        self.T = T
        self.proj = nn.Linear(D, d_model)
        pe = torch.zeros(T, d_model)
        pos = torch.arange(T, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.head    = nn.Linear(d_model, D)

    def _cmask(self, T, device):
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(self, x):
        T = x.shape[1]
        z = self.proj(x) + self.pe[:, :T]
        z = self.encoder(z, mask=self._cmask(T, x.device))
        return self.head(z)

    def reconstruction_error(self, x):
        return ((x - self(x)) ** 2).mean(dim=(1, 2))


# ══════════════════════════════════════════════════════════════════
#  벤치마크 유틸
# ══════════════════════════════════════════════════════════════════
def count_params(m):
    return sum(p.numel() for p in m.parameters())


def measure_pipeline_cpu(model, D, T, repeat=300, warmup=30):
    """실제 파이프라인 전체: deque→preproc→tensor→inference (CPU)."""
    model.eval().cpu()
    scaler = StandardScaler()
    scaler.fit(np.random.randn(500, D).astype(np.float32))
    buf    = deque(maxlen=T)
    stream = [np.random.randn(D).astype(np.float32) for _ in range(T + repeat + warmup)]
    for f in stream[:T]: buf.append(f)

    t_pre = t_mov = t_inf = 0.

    for i, frame in enumerate(stream[T:]):
        buf.append(frame)
        t0 = time.perf_counter()
        arr    = np.array(buf, dtype=np.float32)
        scaled = scaler.transform(arr).astype(np.float32)
        x_np   = apply_relative_transform(scaled, 'none')
        t1 = time.perf_counter()
        x_t    = torch.tensor(x_np).unsqueeze(0)
        t2 = time.perf_counter()
        with torch.no_grad(): model.reconstruction_error(x_t).item()
        t3 = time.perf_counter()

        if i >= warmup:
            t_pre += (t1 - t0) * 1000
            t_mov += (t2 - t1) * 1000
            t_inf += (t3 - t2) * 1000

    n = repeat
    return t_pre/n, t_mov/n, t_inf/n, (t_pre+t_mov+t_inf)/n


def measure_inference_gpu(model, D, T, repeat=200, warmup=30):
    """GPU pure inference (전처리 제외, tensor 이미 GPU에 있음 가정)."""
    if not torch.cuda.is_available():
        return None
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


def measure_pipeline_gpu(model, D, T, repeat=200, warmup=30):
    """GPU 파이프라인 전체: numpy 전처리(CPU) + tensor이동(CPU→GPU) + inference(GPU)."""
    if not torch.cuda.is_available():
        return None, None, None
    model.eval().cuda()
    scaler = StandardScaler()
    scaler.fit(np.random.randn(500, D).astype(np.float32))
    buf    = deque(maxlen=T)
    stream = [np.random.randn(D).astype(np.float32) for _ in range(T + repeat + warmup)]
    for f in stream[:T]: buf.append(f)

    t_pre = t_mov = t_inf = 0.

    for i, frame in enumerate(stream[T:]):
        buf.append(frame)
        t0 = time.perf_counter()
        arr    = np.array(buf, dtype=np.float32)
        scaled = scaler.transform(arr).astype(np.float32)
        x_np   = apply_relative_transform(scaled, 'none')
        t1 = time.perf_counter()
        x_t = torch.tensor(x_np).unsqueeze(0).cuda()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        with torch.no_grad():
            model.reconstruction_error(x_t).item()
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        if i >= warmup:
            t_pre += (t1 - t0) * 1000
            t_mov += (t2 - t1) * 1000
            t_inf += (t3 - t2) * 1000

    n = repeat
    return t_pre/n, t_mov/n, t_inf/n


# ══════════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', type=int, default=300)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--input_dim', type=int, default=6)
    args = parser.parse_args()

    D, T, N = args.input_dim, args.seq_len, args.repeat
    has_gpu  = torch.cuda.is_available()
    print(f"측정 조건: seq_len={T}, input_dim={D}, repeat={N}회")
    print(f"GPU 사용 가능: {has_gpu}\n")

    FRAME_MS = 10.0  # 100Hz

    model_defs = [
        # (이름, 모델, 분류, causal, 비고)
        ("GRU-AE (현재)",
         GRUAE(D, H=128, L=2, T=T),
         "경량·AR", True,  "현재 시스템 기준"),

        ("LSTM-AE",
         LSTMAE(D, H=128, L=2, T=T),
         "비교·AR", True,  "CPU fused GEMM으로 GRU보다 빠름"),

        ("TCN-AE",
         TCN_AE(D, C=64, nb=6, T=T),
         "비교·인과", True,  "Conv O(C²k) << RNN O(H²), receptive field=126"),

        ("VRNN (OmniAnomaly)",
         VRNN(D, H=128, Z=32, T=T),
         "SOTA·AR", True,  "KDD 2019, 다변량 센서 anomaly detection SOTA"),

        ("Anomaly Transformer",
         AnomalyTransformer(D, d_model=128, nhead=4, num_layers=3, dim_ff=256, T=T),
         "SOTA·비인과", False, "ICLR 2022 최다인용 SOTA, association discrepancy"),

        ("TimesNet",
         TimesNet(D, d_model=64, dim_ff=128, num_layers=3, T=T),
         "SOTA·비인과", False, "ICLR 2023, 1D→2D변환+Conv"),

        ("ModernTCN (causal)",
         ModernTCN(D, d_model=64, kernel_size=51, dim_ff=128, num_layers=3, T=T, causal=True),
         "SOTA·인과", True,  "2024, large-kernel depthwise conv"),

        ("TranAD (VLDB 2022)",
         TranAD(D, d_model=128, nhead=4, num_layers=2, dim_ff=256, T=T),
         "SOTA·비인과", False, "Two-branch Transformer + 적대적 자기조건화"),

        ("Causal Transformer-AE",
         CausalTransformerAE(D, d_model=128, nhead=4, num_layers=4, dim_ff=256, T=T),
         "비교·AR", True,  "GPT-decoder 스타일, causal mask, 인과성 보장"),
    ]

    rows = []
    for name, model, cat, causal, note in model_defs:
        n_p = count_params(model)
        mb  = n_p * 4 / 1024 / 1024
        print(f"측정 중: {name} ({n_p/1e3:.1f}K params)...")

        # CPU 파이프라인
        tp_c, tm_c, ti_c, tot_c = measure_pipeline_cpu(model, D, T, N, warmup=30)

        # GPU
        ti_g = None
        tot_g = None
        if has_gpu:
            tp_g, tm_g, ti_g = measure_pipeline_gpu(model, D, T, min(N, 200), warmup=20)
            if ti_g is not None:
                tot_g = (tp_g or 0) + (tm_g or 0) + ti_g

        rows.append((name, cat, causal, n_p, mb,
                      tp_c, tm_c, ti_c, tot_c,
                      ti_g, tot_g, note))
        gstr = f"  GPU추론={ti_g:.2f}ms, 파이프라인={tot_g:.2f}ms" if ti_g else ""
        print(f"  CPU: 전처리={tp_c:.3f}ms, 이동={tm_c:.3f}ms, 추론={ti_c:.2f}ms, 합계={tot_c:.2f}ms{gstr}")

    # ── 출력
    gru_cpu = rows[0][7]   # GRU CPU 추론
    gru_tot_c = rows[0][8]
    gru_tot_g = rows[0][10]

    W = 130
    print("\n" + "═" * W)
    print(f"  시계열 이상감지 모델 비교표  (파이프라인 전체 비용, repeat={N})")
    print(f"  파이프라인: deque → scaler.transform → torch.tensor → model.reconstruction_error")
    print("═" * W)

    hdr = (f"  {'모델':<30} {'분류':<12} {'인과':<5} {'파라미터':>8} {'크기':>7}"
           f"  {'──── CPU (전체 파이프라인) ────':^36}"
           f"  {'── GPU ──':^20}")
    print(hdr)
    sub = (f"  {'':30} {'':12} {'':5} {'':8} {'':7}"
           f"  {'전처리':>8} {'이동':>7} {'추론':>8} {'합계':>9} {'부하':>7}"
           f"  {'추론':>8} {'합계':>9}")
    print(sub)
    print("─" * W)

    for (name, cat, causal, n_p, mb,
         tp_c, tm_c, ti_c, tot_c,
         ti_g, tot_g, note) in rows:
        p_str  = f"{n_p/1e6:.2f}M" if n_p >= 1e6 else f"{n_p/1e3:.0f}K"
        mb_str = f"{mb:.1f}MB"
        caus   = "✓" if causal else "✗"
        load   = f"{tot_c/FRAME_MS*100:.0f}%"

        g_inf_str = f"{ti_g:>7.2f}ms" if ti_g else f"{'N/A':>8}"
        g_tot_str = f"{tot_g:>7.2f}ms" if tot_g else f"{'N/A':>8}"

        print(f"  {name:<30} {cat:<12} {caus:<5} {p_str:>8} {mb_str:>7}"
              f"  {tp_c:>6.3f}ms {tm_c:>5.3f}ms {ti_c:>6.2f}ms {tot_c:>7.2f}ms {load:>7}"
              f"  {g_inf_str} {g_tot_str}")

    print("═" * W)
    print(f"""
[모델별 설명]
  GRU-AE        : 현재 시스템. 경량·인과. h_n 스트리밍 가능.
  LSTM-AE       : GRU 대비 파라미터 33% 많지만 CPU에서 빠름 (4게이트 fused GEMM).
  TCN-AE        : 인과 팽창 합성곱. Conv O(C²k) vs RNN O(H²) → 파라미터 효율↑.
                  단점: receptive field = (k-1)×Σ(2^i) 이상 과거 정보 활용 불가.
  VRNN          : KDD 2019 SOTA. 잠재 변수 z_t 샘플링 → 불확실성 모델링.
                  인과적, 점진적 slip/변화에 민감. 현재 시스템의 가장 자연스러운 업그레이드 후보.
  Anomaly-Trans : ICLR 2022 SOTA. Association Discrepancy 기반 검출.
                  비인과 → 실시간 불가, 오프라인/사후분석용.
  TimesNet      : ICLR 2023. FFT 주기 추출 + 2D Conv. 주기성 있는 신호에 강함.
                  비인과 (FFT가 전체 시퀀스 필요).
  ModernTCN     : 2024. large-kernel depthwise conv (kernel=51).
                  인과 패딩으로 real-time 사용 가능, TCN-AE보다 receptive field↑.
                  파라미터 효율은 TCN-AE보다 낮지만 성능↑.
  TranAD        : VLDB 2022. Two-branch Transformer + 적대적 자기조건화.
                  W1 재구성 → x̂₁, W2가 x̂₁을 피드백으로 받아 정제 → x̂₂.
                  비인과 (전체 시퀀스 attention) → 오프라인 분석용.
  Causal-Trans  : GPT-decoder 스타일, 상삼각 causal mask 적용.
                  GRU-AE와 동일한 인과성, 파라미터 多. 성능 상한선 참고용.

[실시간 사용 가능 여부 (✓인과 모델만)]
  GRU-AE / LSTM-AE / VRNN / TCN-AE / ModernTCN / Causal-Trans → 실시간 OK
  Anomaly-Transformer / TimesNet / TranAD                       → 배치/오프라인 전용

[현재 시스템 업그레이드 추천 순서]
  1. VRNN          — 구조적으로 가장 유사 (AR + 잠재변수), 불확실성 추정 가능
  2. ModernTCN     — large kernel로 TCN 개선, 인과, 성능↑ 파라미터 효율적
  3. LSTM-AE       — CPU에서 가장 빠른 AR 모델, drop-in 교체 가능
  4. Causal-Trans  — GRU보다 표현력↑, 인과 보장, 무거운 대신 정확도 향상 가능
""")


if __name__ == '__main__':
    main()
