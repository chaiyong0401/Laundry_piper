"""
Ch-A / Ch-B 중복 감지 분석 + FP 성공 vs 실패 에피소드 시계열 비교
"""
import sys, os
sys.path.insert(0, '/home/dyros/mcy_ws')

import numpy as np
import glob
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# ─── 모델 정의 (train_gru_ae.py 와 동일) ───────────────────────────────────
class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
    def forward(self, x):
        _, h_n = self.gru(x)
        return h_n

class GRUDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, h_n):
        last_h = h_n[-1]
        inp = last_h.unsqueeze(1).expand(-1, self.seq_len, -1)
        out, _ = self.gru(inp, h_n)
        return self.fc(out)

class GRUAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, seq_len):
        super().__init__()
        self.encoder = GRUEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = GRUDecoder(hidden_dim, input_dim, num_layers, dropout, seq_len)
    def forward(self, x):
        return self.decoder(self.encoder(x))
    def reconstruction_error(self, x):
        with torch.no_grad():
            recon = self.forward(x)
            return ((recon - x) ** 2).mean(dim=(1, 2))

# ─── 설정 ────────────────────────────────────────────────────────────────────
DATA_DIR    = '/home/dyros/mcy_ws/transport_new'
MODEL_PATH  = f'{DATA_DIR}/transport_gru_ae_raw.pt'
SCALER_PATH = f'{DATA_DIR}/transport_scaler_raw.pkl'
THR_A_PATH  = f'{DATA_DIR}/transport_threshold_raw.npy'
THR_B_PATH  = f'{DATA_DIR}/transport_threshold_b_raw.npy'

SEQ_LEN     = 50
STRIDE      = 10
SMOOTH_WIN  = 3
WARMUP_A    = 0    # Ch-A (--warmup 0 사용 시)
WARMUP_B    = 30
CONSEC_A    = 2
CONSEC_B    = 3

# ─── 모델 로드 ───────────────────────────────────────────────────────────────
scaler = joblib.load(SCALER_PATH)
ckpt   = torch.load(MODEL_PATH, map_location='cpu')
hidden_dim = ckpt['encoder.gru.weight_hh_l0'].shape[1]
input_dim  = scaler.n_features_in_
model = GRUAE(input_dim=input_dim, hidden_dim=hidden_dim,
              num_layers=2, dropout=0.2, seq_len=SEQ_LEN)
model.load_state_dict(ckpt)
model.eval()

threshold_a = float(np.load(THR_A_PATH))
threshold_b = float(np.load(THR_B_PATH))
print(f"threshold_a={threshold_a:.6f}, threshold_b={threshold_b:.6f}")

# ─── 유틸 함수 ───────────────────────────────────────────────────────────────
def compute_mrl(frame: np.ndarray) -> float:
    vecs = frame[:48].reshape(16, 3).astype(float)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1e-8, norms)
    return float(np.linalg.norm((vecs / norms).mean(axis=0)))

def compute_force_mag(frame: np.ndarray) -> float:
    vecs = frame[:48].reshape(16, 3).astype(float)
    return float(np.linalg.norm(vecs, axis=1).mean())

def run_channel_b(seq, warmup=WARMUP_B, smooth_win=SMOOTH_WIN, threshold=threshold_b, consec=CONSEC_B):
    T = len(seq)
    mrl = np.array([compute_mrl(seq[t]) for t in range(T)])
    deltas = np.full(T, 0.0)
    for t in range(smooth_win, T):
        ref = mrl[max(0, t-smooth_win):t].mean()
        deltas[t] = abs(mrl[t] - ref)

    b_consec = 0
    flag_b = False
    flag_frame = -1
    for t in range(T):
        if t < warmup:
            b_consec = 0
            continue
        if deltas[t] > threshold:
            b_consec += 1
            if b_consec >= consec:
                flag_b = True
                flag_frame = t
                break
        else:
            b_consec = 0
    return flag_b, flag_frame, mrl, deltas

def run_channel_a(seq, warmup=WARMUP_A, threshold=threshold_a, consec=CONSEC_A):
    T = len(seq)
    scaled = scaler.transform(seq).astype(np.float32)
    errors = []
    starts = []
    a_consec = 0
    flag_a = False
    flag_win = -1

    for start in range(0, T - SEQ_LEN + 1, STRIDE):
        win = torch.tensor(scaled[start:start+SEQ_LEN]).unsqueeze(0)
        err = float(model.reconstruction_error(win).item())
        errors.append(err)
        starts.append(start)

        if start < warmup:
            a_consec = 0
            continue
        if err > threshold:
            a_consec += 1
            if a_consec >= consec:
                flag_a = True
                flag_win = start
                break
        else:
            a_consec = 0
    return flag_a, flag_win, errors, starts

# ─── 전체 에피소드 평가 ──────────────────────────────────────────────────────
failure_files = sorted(glob.glob(f'{DATA_DIR}/*failure_raw.npy'))
success_files = sorted(glob.glob(f'{DATA_DIR}/*success_raw.npy'))

results = {'failure': [], 'success': []}

for label, files in [('failure', failure_files), ('success', success_files)]:
    for fp in files:
        name = os.path.basename(fp).replace('_raw.npy', '')
        seq  = np.load(fp)
        T    = len(seq)

        flag_a, frame_a, errors, starts = run_channel_a(seq)
        flag_b, frame_b, mrl, deltas    = run_channel_b(seq)

        mag = np.array([compute_force_mag(seq[t]) for t in range(T)])

        results[label].append({
            'name': name, 'T': T,
            'flag_a': flag_a, 'flag_b': flag_b,
            'frame_a': frame_a, 'frame_b': frame_b,
            'mrl': mrl, 'deltas': deltas, 'mag': mag,
            'errors': errors, 'starts': starts,
        })

# ─── 중복 감지 분석 출력 ─────────────────────────────────────────────────────
print('\n' + '='*65)
print('=== 실패 에피소드 감지 분석 ===')
print('%-40s %5s %4s %4s %5s' % ('에피소드', '길이', 'ChA', 'ChB', 'OR'))
only_a = only_b = both_ab = neither = 0
for r in results['failure']:
    tag = ('YES' if r['flag_a'] else 'no ') + '  ' + ('YES' if r['flag_b'] else 'no ')
    detected = r['flag_a'] or r['flag_b']
    print('%-40s %5d  %s  %s' % (r['name'][-35:], r['T'],
                                  'YES' if r['flag_a'] else 'no ',
                                  'YES' if r['flag_b'] else 'no '))
    if r['flag_a'] and r['flag_b']:  both_ab += 1
    elif r['flag_a']:                only_a  += 1
    elif r['flag_b']:                only_b  += 1
    else:                            neither += 1

total_f = len(results['failure'])
print(f'\n  Ch-A만: {only_a}/{total_f}  Ch-B만: {only_b}/{total_f}  둘 다: {both_ab}/{total_f}  미감지: {neither}/{total_f}')
print(f'  전체 감지율: {(total_f - neither)/total_f*100:.1f}%')

fp_eps = [r for r in results['success'] if r['flag_b']]
print(f'\n=== Ch-B FP 성공 에피소드: {len(fp_eps)}/{len(results["success"])} ===')
for r in fp_eps:
    print('  %-40s  len=%-5d frame_b=%d  max_delta=%.4f' % (
          r['name'][-35:], r['T'], r['frame_b'], r['deltas'].max()))

# ─── Ch-B 전용 감지 실패 vs FP 비교 시각화 ──────────────────────────────────
# 감지 카테고리별 에피소드 수집
chb_only_fail = [r for r in results['failure'] if r['flag_b'] and not r['flag_a']]
both_fail     = [r for r in results['failure'] if r['flag_a'] and r['flag_b']]
cha_only_fail = [r for r in results['failure'] if r['flag_a'] and not r['flag_b']]
miss_fail     = [r for r in results['failure'] if not r['flag_a'] and not r['flag_b']]

print(f'\n  Ch-B only 실패: {len(chb_only_fail)}개 → 이게 Ch-B 고유 기여분')
print(f'  A&B 둘 다:     {len(both_fail)}개')
print(f'  Ch-A only:     {len(cha_only_fail)}개')
print(f'  미감지:         {len(miss_fail)}개')

# ─── Ch-B spike 지속성 분석 (급격한 낙하 에피소드) ─────────────────────────
print('\n=== Ch-B spike 지속성 분석 (max_delta > threshold_b 인 실패 에피소드) ===')
for r in results['failure']:
    above = (r['deltas'][WARMUP_B:] > threshold_b)
    if above.any():
        # 최대 연속 구간 계산
        max_consec = 0
        cur = 0
        for v in above:
            if v:
                cur += 1
                max_consec = max(max_consec, cur)
            else:
                cur = 0
        print('  %-38s  max_delta=%.3f  max_consec_above_thr=%d  (consec_b=%d)' % (
              r['name'][-35:], r['deltas'].max(), max_consec, CONSEC_B))

# ─── 시계열 그래프 ────────────────────────────────────────────────────────────
# FP 성공 에피소드 + max_delta 높은 실패 에피소드 비교
high_delta_fail = sorted(results['failure'], key=lambda r: r['deltas'].max(), reverse=True)[:6]
show_fp = fp_eps if fp_eps else []

n_cols = max(len(show_fp), len(high_delta_fail), 2)
row_labels = ['MRL', 'MRL Delta', 'Force Mag']
row_keys   = ['mrl', 'deltas', 'mag']

fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 12))
if axes.ndim == 1:
    axes = axes.reshape(3, -1)
fig.suptitle('Ch-B FP 성공 에피소드 vs 실패 에피소드 시계열 비교', fontsize=13, fontweight='bold')

# 위쪽 행들: FP 성공 에피소드
for ci in range(n_cols):
    for ri, key in enumerate(row_keys):
        ax = axes[ri][ci]
        if ci < len(show_fp):
            r = show_fp[ci]
            ax.plot(r[key], color='steelblue', linewidth=1.0)
            if key == 'deltas':
                ax.axhline(threshold_b, color='red', linestyle='--', linewidth=0.9,
                           label=f'thr={threshold_b:.3f}')
                if r['frame_b'] >= 0:
                    ax.axvline(r['frame_b'], color='red', linewidth=1.5, alpha=0.7)
            ax.axvline(WARMUP_B, color='gray', linestyle=':', linewidth=0.8)
            if ri == 0:
                ax.set_title(f'[FP 성공] {r["name"][-20:]}\nT={r["T"]}', fontsize=7)
        elif ci < len(high_delta_fail):
            r = high_delta_fail[ci]
            color = 'tomato' if (r['flag_a'] or r['flag_b']) else 'salmon'
            ax.plot(r[key], color=color, linewidth=1.0)
            if key == 'deltas':
                ax.axhline(threshold_b, color='darkred', linestyle='--', linewidth=0.9)
                above_thr = np.where(r['deltas'][WARMUP_B:] > threshold_b)[0]
                if len(above_thr) > 0:
                    for idx in above_thr:
                        ax.axvline(idx + WARMUP_B, color='darkred', alpha=0.3, linewidth=0.6)
            ax.axvline(WARMUP_B, color='gray', linestyle=':', linewidth=0.8)
            if ri == 0:
                tag = ('A' if r['flag_a'] else '') + ('B' if r['flag_b'] else '')
                tag = tag if tag else 'MISS'
                md = r['deltas'].max()
                ax.set_title(f'[실패 {tag}] {r["name"][-20:]}\nT={r["T"]}  max_d={md:.3f}', fontsize=7)
        else:
            ax.set_visible(False)
            continue
        ax.set_ylabel(row_labels[ri], fontsize=7)
        ax.tick_params(labelsize=6)
        if key == 'deltas':
            ax.legend(fontsize=6)

plt.tight_layout()
plt.savefig('/home/dyros/mcy_ws/transport_chb_overlap.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── MRL delta 분포 비교 (성공 FP vs 실패 그룹별) ──────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle('MRL Delta 분포 비교 (warmup 이후)', fontsize=12, fontweight='bold')

def collect_deltas(ep_list, warmup=WARMUP_B):
    all_d = []
    for r in ep_list:
        d = r['deltas'][warmup:]
        all_d.extend(d[~np.isnan(d)].tolist())
    return np.array(all_d)

d_fp      = collect_deltas(fp_eps)
d_fail    = collect_deltas(results['failure'])
d_success = collect_deltas([r for r in results['success'] if not r['flag_b']])

bins = np.linspace(0, 0.5, 60)
axes2[0].hist(d_fp, bins=bins, color='steelblue', alpha=0.7, label=f'FP 성공 ({len(fp_eps)}개)')
axes2[0].hist(d_fail, bins=bins, color='tomato', alpha=0.5, label=f'실패 ({len(results["failure"])}개)')
axes2[0].axvline(threshold_b, color='darkred', linestyle='--', label=f'thr={threshold_b:.3f}')
axes2[0].set_title('FP 성공 vs 실패 전체')
axes2[0].legend(fontsize=8)
axes2[0].set_xlabel('MRL delta')

axes2[1].hist(d_success, bins=bins, color='gray', alpha=0.7, label=f'정상 성공 ({len(results["success"])-len(fp_eps)}개)')
axes2[1].hist(d_fp, bins=bins, color='steelblue', alpha=0.6, label=f'FP 성공 ({len(fp_eps)}개)')
axes2[1].axvline(threshold_b, color='darkred', linestyle='--')
axes2[1].set_title('정상 성공 vs FP 성공')
axes2[1].legend(fontsize=8)
axes2[1].set_xlabel('MRL delta')

# Force magnitude 비교
def collect_mag(ep_list):
    return np.concatenate([r['mag'] for r in ep_list])

axes2[2].hist(collect_mag(results['success']), bins=50, color='steelblue', alpha=0.5, label='성공', density=True)
axes2[2].hist(collect_mag(results['failure']), bins=50, color='tomato', alpha=0.5, label='실패', density=True)
axes2[2].set_title('Force Magnitude 분포 (성공 vs 실패)')
axes2[2].set_xlabel('Mean force magnitude')
axes2[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('/home/dyros/mcy_ws/transport_chb_dist.png', dpi=150, bbox_inches='tight')
plt.close()

print('\n그래프 저장:')
print('  /home/dyros/mcy_ws/transport_chb_overlap.png')
print('  /home/dyros/mcy_ws/transport_chb_dist.png')
