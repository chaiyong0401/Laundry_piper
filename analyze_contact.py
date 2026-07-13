"""failure의 접촉 강도(norm)가 success와 다른지 확인."""
import numpy as np
import glob

DATA_DIR = 'grasp_05'
SEQ_LEN  = 50

success_files = sorted(glob.glob(f'{DATA_DIR}/grasp_*_success_raw.npy'))
failure_files = sorted(glob.glob(f'{DATA_DIR}/grasp_*_failure_raw.npy'))

def seq_stats(files, label):
    norms = []
    for f in files:
        seq = np.load(f).astype(np.float32)
        T = len(seq)
        # 각 프레임 L2 norm
        frame_norms = np.linalg.norm(seq, axis=1)  # (T,)
        # 마지막 50프레임 (inference window에 해당)
        if T >= SEQ_LEN:
            last_window_mean_norm = frame_norms[-SEQ_LEN:].mean()
            last_window_max_norm  = frame_norms[-SEQ_LEN:].max()
        else:
            last_window_mean_norm = frame_norms.mean()
            last_window_max_norm  = frame_norms.max()
        norms.append((f.split('/')[-1], last_window_mean_norm, last_window_max_norm,
                      frame_norms.mean(), T))

    print(f"\n=== {label} (n={len(norms)}) ===")
    print(f"{'파일':>45}  {'last50_mean':>12}  {'last50_max':>11}  {'all_mean':>9}  {'T':>5}")
    for fname, lm, lx, am, T in sorted(norms, key=lambda x: x[1]):
        print(f"  {fname:>45}  {lm:12.3f}  {lx:11.3f}  {am:9.3f}  {T:5d}")

    vals = np.array([x[1] for x in norms])
    print(f"\n  last50_mean_norm: mean={vals.mean():.3f}  std={vals.std():.3f}  "
          f"min={vals.min():.3f}  max={vals.max():.3f}  p25={np.percentile(vals,25):.3f}  p75={np.percentile(vals,75):.3f}")
    return vals

s_norms = seq_stats(success_files, 'SUCCESS')
f_norms = seq_stats(failure_files, 'FAILURE')

print(f"\n=== 접촉 강도 overlap 분석 ===")
s_med = np.median(s_norms)
f_med = np.median(f_norms)
print(f"Success median norm: {s_med:.3f}")
print(f"Failure median norm: {f_med:.3f}")

# failure 중 success norm 범위 밖에 있는 것
s_min, s_max = s_norms.min(), s_norms.max()
f_low  = np.sum(f_norms < np.percentile(s_norms, 5))
f_high = np.sum(f_norms > np.percentile(s_norms, 95))
f_overlap = np.sum((f_norms >= np.percentile(s_norms, 5)) & (f_norms <= np.percentile(s_norms, 95)))
print(f"Failure norm < success p5 ({np.percentile(s_norms,5):.3f}): {f_low}/15  (low-contact failure)")
print(f"Failure norm > success p95 ({np.percentile(s_norms,95):.3f}): {f_high}/15  (high-contact failure)")
print(f"Failure norm within success p5~p95: {f_overlap}/15  (overlap - hard to detect by norm alone)")
