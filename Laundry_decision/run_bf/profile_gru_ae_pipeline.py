"""
GRU-AE 실제 파이프라인 단계별 지연시간 프로파일
================================================
벤치마크(pure inference)가 아닌, 실제 detector.py의 update() 호출 경로를 재현.

측정 단계:
  1. np.array(buffer)          — deque → numpy 변환
  2. build_hybrid_window()     — norm 차원 추가 (hybrid 모드)
  3. scaler.transform()        — StandardScaler (sklearn, CPU numpy)
  4. apply_relative_transform()— window_mean / first_frame 변환 (numpy)
  5. torch.tensor().unsqueeze(0).to(device) — numpy → torch tensor + 장치 이동
  6. model.reconstruction_error()           — forward pass + MSE 계산
  ─────────────────────────────────────────────────
  전체 (3~6): 실제 추론 호출 1회 비용 (buffer 풀 때마다)
  전체 (1~6): update() 1회 비용 (stride 마다)

Usage:
  python profile_gru_ae_pipeline.py
  python profile_gru_ae_pipeline.py --device cpu
  python profile_gru_ae_pipeline.py --repeat 500
"""

import argparse
import time
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '/home/dyros/mcy_ws')

# train_gru_ae.py 의 실제 함수 가져오기
from train_gru_ae import (
    GRUAE, build_hybrid_window, apply_relative_transform
)


# ──────────────────────────────────────────────────────────────
#  단계별 타이밍 헬퍼
# ──────────────────────────────────────────────────────────────
class Stopwatch:
    def __init__(self):
        self.laps = {}

    def lap(self, name: str, fn):
        t0 = time.perf_counter()
        result = fn()
        self.laps[name] = self.laps.get(name, 0.0) + (time.perf_counter() - t0) * 1000
        return result

    def mean(self, name: str, n: int):
        return self.laps.get(name, 0.0) / n


# ──────────────────────────────────────────────────────────────
#  메인
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', help='cpu / cuda')
    parser.add_argument('--repeat', type=int, default=500)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--input_dim', type=int, default=6)
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='실제 시스템 hidden_dim (basket2.py 기본값 128)')
    parser.add_argument('--hybrid', action='store_true',
                        help='build_hybrid_window 사용 (D→D+1)')
    parser.add_argument('--relative', default='none',
                        choices=['none', 'window_mean', 'first_frame'])
    parser.add_argument('--stride', type=int, default=10,
                        help='score를 계산하는 stride (버퍼 풀 때마다는 아님)')
    args = parser.parse_args()

    device = torch.device(args.device)
    D = args.input_dim
    effective_D = D + 1 if args.hybrid else D
    T = args.seq_len
    n = args.repeat

    print(f"장치: {device}  |  input_dim={D}  |  seq_len={T}  |  "
          f"hybrid={args.hybrid}  |  relative={args.relative}")
    print(f"hidden_dim={args.hidden_dim}  |  repeat={n}회\n")

    # ── 모델 (실제 시스템과 동일 구조)
    model = GRUAE(
        input_dim  = effective_D,
        hidden_dim = args.hidden_dim,
        num_layers = 2,
        dropout    = 0.2,
        seq_len    = T,
    ).to(device)
    model.eval()

    # ── 가상 scaler (실제는 joblib.load로 로드된 StandardScaler)
    scaler = StandardScaler()
    dummy_fit = np.random.randn(1000, D).astype(np.float32)
    scaler.fit(dummy_fit)

    # ── 가상 XELA 프레임 스트림 (실제 raw 6D feature)
    fake_stream = [np.random.randn(D).astype(np.float32) for _ in range(T + n)]

    # ──────────────────────────────────────────────
    #  워밍업 (cache, JIT, 등)
    # ──────────────────────────────────────────────
    buf_warmup = deque(maxlen=T)
    for frame in fake_stream[:T]:
        buf_warmup.append(frame)
    w = np.array(buf_warmup, dtype=np.float32)
    if args.hybrid:
        w = build_hybrid_window(w)
    x_np = apply_relative_transform(scaler.transform(w if not args.hybrid else w[:, :D])
                                     if not args.hybrid else scaler.transform(w[:, :D]),
                                     args.relative)
    # simpler warmup
    for _ in range(30):
        xt = torch.tensor(x_np).unsqueeze(0).to(device)
        with torch.no_grad():
            model.reconstruction_error(xt)

    # ──────────────────────────────────────────────
    #  실제 측정: 프레임 스트리밍 시뮬레이션
    # ──────────────────────────────────────────────
    sw = Stopwatch()
    buffer = deque(maxlen=T)

    # 버퍼 초기 채우기
    for frame in fake_stream[:T]:
        buffer.append(frame)

    inference_count = 0

    for i, frame in enumerate(fake_stream[T:T + n]):
        # 실제 update() 와 동일한 흐름
        frame_np = frame  # 실제는 XELA callback에서 옴

        # 1. buffer append (deque O(1))
        sw.lap('1_buffer_append', lambda: buffer.append(frame_np))

        # 2. buffer → numpy (sliding window)
        arr = sw.lap('2_np_array', lambda: np.array(buffer, dtype=np.float32))

        # 3. hybrid window (optional)
        if args.hybrid:
            arr = sw.lap('3_hybrid', lambda: build_hybrid_window(arr))
            feat = arr[:, :D]
        else:
            feat = arr

        # 4. scaler.transform (sklearn, numpy 연산)
        scaled = sw.lap('4_scaler', lambda: scaler.transform(feat).astype(np.float32))

        # 5. relative transform (numpy)
        x_np = sw.lap('5_relative', lambda: apply_relative_transform(scaled, args.relative))

        # 6. numpy → torch tensor + 장치 이동
        def _to_tensor():
            return torch.tensor(x_np).unsqueeze(0).to(device)
        x_t = sw.lap('6_to_tensor', _to_tensor)

        # 7. 모델 forward + MSE (reconstruction_error)
        def _infer():
            with torch.no_grad():
                return model.reconstruction_error(x_t).item()
        score = sw.lap('7_model_infer', _infer)

        inference_count += 1

    # ──────────────────────────────────────────────
    #  결과 출력
    # ──────────────────────────────────────────────
    steps = [
        ('1_buffer_append',  'buffer.append(frame)        — deque O(1)'),
        ('2_np_array',       'np.array(buffer)            — deque→numpy (50,6)'),
        ('3_hybrid',         'build_hybrid_window()       — norm 차원 추가'),
        ('4_scaler',         'scaler.transform()          — StandardScaler, numpy'),
        ('5_relative',       'apply_relative_transform()  — numpy 연산'),
        ('6_to_tensor',      'torch.tensor().to(device)  — numpy→tensor+장치이동'),
        ('7_model_infer',    'model.reconstruction_error()— forward+MSE'),
    ]

    print("=" * 80)
    print(f"  단계별 평균 지연시간 (반복 {n}회, 장치: {device})")
    print("=" * 80)
    total_preproc = 0.0
    total_all = 0.0
    for key, desc in steps:
        if key == '3_hybrid' and not args.hybrid:
            continue
        ms = sw.mean(key, n)
        bar = '█' * max(1, int(ms / 0.05))
        print(f"  {desc:<50} {ms:6.3f} ms  {bar}")
        if key not in ('1_buffer_append',):
            total_preproc += ms
        total_all += ms

    preproc_only = sum(sw.mean(k, n) for k in ['2_np_array','3_hybrid','4_scaler','5_relative']
                       if not (k == '3_hybrid' and not args.hybrid))
    tensor_move  = sw.mean('6_to_tensor', n)
    infer_only   = sw.mean('7_model_infer', n)

    print("-" * 80)
    print(f"  {'전처리 합계 (2~5단계)':<50} {preproc_only:6.3f} ms")
    print(f"  {'numpy→tensor 이동 (6단계)':<50} {tensor_move:6.3f} ms")
    print(f"  {'모델 추론만 (7단계)':<50} {infer_only:6.3f} ms")
    print(f"  {'추론 호출 1회 전체 (2~7단계)':<50} {preproc_only+tensor_move+infer_only:6.3f} ms")
    print(f"  {'update() 전체 (1~7단계)':<50} {total_all:6.3f} ms")
    print("=" * 80)

    frame_period_ms = 10.0   # 100Hz
    stride = args.stride
    print(f"""
[실시간성 판단] XELA 100Hz (10ms/frame), stride={stride}
  · 매 프레임 update() 비용      : {total_all:.3f} ms  (vs 프레임 주기 10ms)
  · stride={stride}마다 추론 호출 비용: {preproc_only+tensor_move+infer_only:.3f} ms
  · 추론 호출 주기               : {stride * frame_period_ms:.0f} ms 마다 1회
  · CPU 점유율 추정              : {total_all / frame_period_ms * 100:.1f}%/frame
                                   (매 프레임 {total_all:.2f}ms 소요 / 10ms 주기)

[벤치마크 vs 실제 비교]
  · 벤치마크(pure inference)     : {infer_only:.3f} ms  ← forward pass만
  · 실제 추론 호출 1회           : {preproc_only+tensor_move+infer_only:.3f} ms  ← 전처리+이동+추론
  · 차이 원인: scaler.transform({preproc_only:.3f}ms) + numpy→tensor({tensor_move:.3f}ms)
""")


if __name__ == '__main__':
    main()
