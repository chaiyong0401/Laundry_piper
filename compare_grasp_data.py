#!/usr/bin/env python3
"""
inference 시 기록된 raw 버퍼와 학습 데이터 비교.
Usage:
    python compare_grasp_data.py [grasp_05/infer_debug_XXXXXXXX_raw.npy]
    인자 없으면 가장 최근 infer_debug 파일 자동 선택
"""
import sys, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRAIN_DIR = "grasp_new"
SEQ_LEN   = 50

# basket_ori.py: vecs=(16,3).flatten() → INTERLEAVED x0,y0,z0, x1,y1,z1...
X_IDX = np.arange(0, 48, 3)   # 0,3,6,...,45
Y_IDX = np.arange(1, 48, 3)   # 1,4,7,...,46
Z_IDX = np.arange(2, 48, 3)   # 2,5,8,...,47


def add_hybrid_norm(arr):
    """48D raw → 49D (norm 차원 추가). hybrid 모드 모델용."""
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return np.concatenate([arr, norm], axis=1).astype(np.float32)


def windowed_scores(arr, model, scaler, seq_len=50, stride=10, use_hybrid=False):
    import torch
    scores = []
    for s in range(0, len(arr) - seq_len + 1, stride):
        w  = arr[s:s+seq_len]
        if use_hybrid:
            w = add_hybrid_norm(w)
        ws = scaler.transform(w.astype(np.float32))
        x  = torch.tensor(ws, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            scores.append(float(model.reconstruction_error(x).item()))
    return np.array(scores)


def plot_per_taxel(infer_arr, success_files, failure_files, out_dir, n_train=5):
    """16 taxel × 3축 시계열 비교 (success=파랑, failure=주황, inference=빨강)"""
    s_arrs = [np.load(f).astype(np.float32) for f in success_files[:n_train]]
    f_arrs = [np.load(f).astype(np.float32) for f in failure_files[:n_train]]

    # INTERLEAVED: x0,y0,z0, x1,y1,z1, ... → taxel i의 x=3i, y=3i+1, z=3i+2
    axis_info = [
        ("cum_x", X_IDX, "compare_taxel_cumX.png"),
        ("cum_y", Y_IDX, "compare_taxel_cumY.png"),
        ("cum_z", Z_IDX, "compare_taxel_cumZ.png"),
    ]

    for axis_name, dim_range, fname in axis_info:
        fig, axes = plt.subplots(4, 4, figsize=(16, 10), sharex=False, sharey=False)
        fig.suptitle(
            f"Per-taxel {axis_name}  "
            f"(blue=success train, orange=failure train, red=inference)",
            fontsize=11
        )
        for taxel_i, dim in enumerate(dim_range):
            ax = axes[taxel_i // 4][taxel_i % 4]
            for a in s_arrs:
                ax.plot(a[:, dim], color='steelblue', alpha=0.45, lw=0.9)
            for a in f_arrs:
                ax.plot(a[:, dim], color='darkorange', alpha=0.45, lw=0.9)
            ax.plot(infer_arr[:, dim], color='crimson', lw=1.5, label='infer')
            ax.axhline(0, color='gray', lw=0.5, linestyle='--')
            ax.set_title(f"taxel {taxel_i}  (dim {dim})", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(True, linewidth=0.4)
        from matplotlib.lines import Line2D
        handles = [Line2D([0],[0],color='steelblue',lw=2,label='success'),
                   Line2D([0],[0],color='darkorange',lw=2,label='failure'),
                   Line2D([0],[0],color='crimson',lw=2,label='inference')]
        axes[0][0].legend(handles=handles, fontsize=7)
        plt.tight_layout()
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=130)
        plt.close()
        print(f"[Plot] 저장: {out_path}")


def plot_scaled_comparison(infer_arr, success_files, failure_files, scaler, out_dir,
                           n_train=5, use_hybrid=False):
    """scaler 변환 공간에서 per-dim mean 비교 + scaled cum_z per-taxel"""
    s_arrs = [np.load(f).astype(np.float32) for f in success_files[:n_train]]
    f_arrs = [np.load(f).astype(np.float32) for f in failure_files[:n_train]]

    def _sc(arr):
        return scaler.transform(add_hybrid_norm(arr) if use_hybrid else arr)

    all_s   = np.vstack(s_arrs)
    all_f   = np.vstack(f_arrs)
    all_i   = infer_arr

    s_sc = _sc(all_s)[:, :48]   # 시각화는 48D 기준
    f_sc = _sc(all_f)[:, :48]
    i_sc = _sc(all_i)[:, :48]

    s_mean = s_sc.mean(axis=0)
    f_mean = f_sc.mean(axis=0)
    i_mean = i_sc.mean(axis=0)
    dims   = np.arange(48)
    # INTERLEAVED dim 이름: dim 0=x0, 1=y0, 2=z0, 3=x1, 4=y1, 5=z1, ...
    labels = []
    for i in range(16):
        labels += [f"x{i}", f"y{i}", f"z{i}"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    w = 0.25
    axes[0].bar(dims - w, s_mean, w*1.8, label='success mean', color='steelblue', alpha=0.75)
    axes[0].bar(dims,     f_mean, w*1.8, label='failure mean', color='darkorange', alpha=0.75)
    axes[0].bar(dims + w, i_mean, w*1.8, label='inference mean', color='crimson',  alpha=0.75)
    axes[0].set_xticks(dims[::4]); axes[0].set_xticklabels(labels[::4], fontsize=7)
    axes[0].axhline(0, color='k', lw=0.5)
    axes[0].set_title("Scaled space: per-dim MEAN  (success / failure / inference)")
    axes[0].legend(); axes[0].grid(True, axis='y')

    diff_si = i_mean - s_mean
    diff_fi = i_mean - f_mean
    colors = ['crimson' if abs(d) > 1.0 else 'steelblue' for d in diff_si]
    axes[1].bar(dims - 0.2, diff_si, 0.38, label='infer − success', color=colors, alpha=0.8)
    axes[1].bar(dims + 0.2, diff_fi, 0.38, label='infer − failure', color='darkorange', alpha=0.6)
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].axhline(+1.0, color='gray', lw=1, linestyle='--', label='±1σ')
    axes[1].axhline(-1.0, color='gray', lw=1, linestyle='--')
    axes[1].set_xticks(dims[::4]); axes[1].set_xticklabels(labels[::4], fontsize=7)
    axes[1].set_title("Scaled MEAN DIFF  (red=|infer−success|>1σ)")
    axes[1].legend(); axes[1].grid(True, axis='y')

    # z_mean (signed) over time  [INTERLEAVED: z at Z_IDX]
    axes[2].set_title("Signed z_mean over time  (mean of 16 taxels, scaled)")
    for a in s_arrs:
        sc = _sc(a)[:, :48]
        axes[2].plot(sc[:, Z_IDX].mean(axis=1), color='steelblue', alpha=0.4, lw=0.8)
    for a in f_arrs:
        sc = _sc(a)[:, :48]
        axes[2].plot(sc[:, Z_IDX].mean(axis=1), color='darkorange', alpha=0.4, lw=0.8)
    axes[2].plot(i_sc[:, Z_IDX].mean(axis=1), color='crimson', lw=1.8, label='inference')
    axes[2].axhline(0, color='k', lw=0.7, linestyle='--')
    axes[2].set_xlabel("frame"); axes[2].set_ylabel("scaled z_mean")
    axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    out1 = os.path.join(out_dir, "compare_scaled_stats.png")
    plt.savefig(out1, dpi=130); plt.close()
    print(f"[Plot] 저장: {out1}")

    n_out = int(np.sum(np.abs(diff_si) > 1.0))
    print(f"[Scaled] |mean diff (infer−success)| > 1σ 차원 수: {n_out}/48")
    top5 = np.argsort(np.abs(diff_si))[-5:][::-1]
    for d in top5:
        print(f"  dim {d:2d} ({labels[d]}): success={s_mean[d]:+.3f}  "
              f"failure={f_mean[d]:+.3f}  infer={i_mean[d]:+.3f}")

    # scaled cum_z per-taxel (4×4)
    fig, axes = plt.subplots(4, 4, figsize=(16, 10))
    fig.suptitle("Scaled cum_z per-taxel  (blue=success, orange=failure, red=inference)", fontsize=11)
    for taxel_i in range(16):
        dim = Z_IDX[taxel_i]   # INTERLEAVED z 인덱스
        ax = axes[taxel_i // 4][taxel_i % 4]
        for a in s_arrs:
            ax.plot(_sc(a)[:, dim], color='steelblue', alpha=0.45, lw=0.9)
        for a in f_arrs:
            ax.plot(_sc(a)[:, dim], color='darkorange', alpha=0.45, lw=0.9)
        ax.plot(i_sc[:, dim], color='crimson', lw=1.5)
        ax.axhline(0, color='gray', lw=0.5, linestyle='--')
        ax.set_title(f"taxel {taxel_i}", fontsize=8)
        ax.tick_params(labelsize=6); ax.grid(True, linewidth=0.4)
    axes[0][0].legend(['success', 'failure', 'infer'], fontsize=7)
    plt.tight_layout()
    out2 = os.path.join(out_dir, "compare_scaled_cumZ.png")
    plt.savefig(out2, dpi=130); plt.close()
    print(f"[Plot] 저장: {out2}")


def main():
    infer_file = sys.argv[1] if len(sys.argv) > 1 else None
    if infer_file is None:
        cands = sorted(glob.glob(f"{TRAIN_DIR}/infer_debug_*_raw.npy"))
        if not cands:
            print("사용법: python compare_grasp_data.py <infer_debug_*_raw.npy>")
            return
        infer_file = cands[-1]
    print(f"Inference 파일: {infer_file}")

    infer_arr = np.load(infer_file).astype(np.float32)
    print(f"  shape={infer_arr.shape}")

    success_files = sorted(glob.glob(f"{TRAIN_DIR}/*success_raw.npy"))
    failure_files = sorted(glob.glob(f"{TRAIN_DIR}/*failure_raw.npy"))
    print(f"학습: success={len(success_files)}, failure={len(failure_files)}")

    # ── 기본 통계 출력 ──────────────────────────────────────────
    infer_z = infer_arr[:, Z_IDX]   # INTERLEAVED z
    print(f"\n[Inference z 통계] (INTERLEAVED 2::3)")
    print(f"  z[0:50]_mean = {infer_z[:50].mean():+.4f}")
    print(f"  z_all_mean   = {infer_z.mean():+.4f}")
    print(f"  z_end_mean   = {infer_z[-1].mean():+.4f}")

    # ── Model windowed score 비교 ──────────────────────────────
    try:
        import torch, joblib, pickle
        sys.path.insert(0, "/home/dyros/mcy_ws")
        from train_gru_ae import GRUAE

        scaler = joblib.load(f"{TRAIN_DIR}/grasp_scaler_raw.pkl")
        ckpt   = torch.load(f"{TRAIN_DIR}/grasp_gru_ae_raw.pt", map_location="cpu")
        # checkpoint에서 실제 input_dim, hidden_dim 자동 감지
        w_ih       = ckpt["encoder.gru.weight_ih_l0"]
        act_input  = w_ih.shape[1]        # 48 or 49 (hybrid)
        act_hidden = w_ih.shape[0] // 3   # 64 or 128
        use_hybrid = (act_input > 48)
        print(f"[Model] input_dim={act_input}, hidden_dim={act_hidden}, hybrid={use_hybrid}")
        model  = GRUAE(input_dim=act_input, hidden_dim=act_hidden, num_layers=2, dropout=0.2, seq_len=50)
        model.load_state_dict(ckpt); model.eval()
        file_thr   = float(np.load(f"{TRAIN_DIR}/grasp_threshold_raw.npy")[0])
        deploy_thr = 0.4

        print(f"\n[Windowed score]  file_thr={file_thr:.4f}  deploy_thr={deploy_thr}")

        s_scores_all, f_scores_all = [], []
        for f in success_files:
            s_scores_all.extend(windowed_scores(np.load(f).astype(np.float32), model, scaler, use_hybrid=use_hybrid))
        for f in failure_files:
            f_scores_all.extend(windowed_scores(np.load(f).astype(np.float32), model, scaler, use_hybrid=use_hybrid))
        infer_sc = windowed_scores(infer_arr, model, scaler, use_hybrid=use_hybrid)

        print(f"  Success train : mean={np.mean(s_scores_all):.4f}  "
              f"p95={np.percentile(s_scores_all, 95):.4f}  "
              f"max={np.max(s_scores_all):.4f}")
        print(f"  Failure train : mean={np.mean(f_scores_all):.4f}  "
              f"p95={np.percentile(f_scores_all, 95):.4f}  "
              f"max={np.max(f_scores_all):.4f}")
        print(f"  Inference     : mean={infer_sc.mean():.4f}  "
              f"p95={np.percentile(infer_sc, 95):.4f}  "
              f"max={infer_sc.max():.4f}")

        # ── 종합 시계열 plot ───────────────────────────────────
        fig, axes = plt.subplots(3, 1, figsize=(13, 11))
        fig.suptitle(f"Inference vs Training  |  {os.path.basename(infer_file)}", fontsize=11)

        # 1) signed z_mean over time
        axes[0].set_title("Signed z_mean over time  (16 taxels avg, INTERLEAVED 2::3)")
        for f in success_files[:6]:
            a = np.load(f).astype(np.float32)
            axes[0].plot(a[:, Z_IDX].mean(axis=1), color='steelblue', alpha=0.35, lw=0.8)
        for f in failure_files[:6]:
            a = np.load(f).astype(np.float32)
            axes[0].plot(a[:, Z_IDX].mean(axis=1), color='darkorange', alpha=0.35, lw=0.8)
        axes[0].plot(infer_z.mean(axis=1), color='crimson', lw=1.8, label='inference')
        axes[0].axhline(0, color='k', lw=0.7, linestyle='--')
        axes[0].axvline(50, color='purple', lw=1, linestyle=':', label='seq_len=50 (추론시작)')
        axes[0].set_ylabel("z_mean (signed)")
        axes[0].legend(fontsize=8); axes[0].grid(True)
        # 범례 patch
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0],[0], color='steelblue', lw=2, label='success train'),
            Line2D([0],[0], color='darkorange', lw=2, label='failure train'),
            Line2D([0],[0], color='crimson', lw=2, label='inference'),
            Line2D([0],[0], color='purple', lw=1, linestyle=':', label='seq_len=50'),
        ]
        axes[0].legend(handles=handles, fontsize=8)

        # 2) abs_mean (all 48) over time
        axes[1].set_title("abs_mean (48 dims) over time")
        for f in success_files[:6]:
            a = np.load(f).astype(np.float32)
            axes[1].plot(np.abs(a).mean(axis=1), color='steelblue', alpha=0.35, lw=0.8)
        for f in failure_files[:6]:
            a = np.load(f).astype(np.float32)
            axes[1].plot(np.abs(a).mean(axis=1), color='darkorange', alpha=0.35, lw=0.8)
        axes[1].plot(np.abs(infer_arr).mean(axis=1), color='crimson', lw=1.8, label='inference')
        axes[1].axvline(50, color='purple', lw=1, linestyle=':')
        axes[1].set_ylabel("abs_mean")
        axes[1].legend(handles=handles[:3], fontsize=8); axes[1].grid(True)

        # 3) windowed score
        axes[2].set_title("Windowed GRU-AE score over time (stride=10)")
        for f in success_files[:6]:
            a = np.load(f).astype(np.float32)
            sc = windowed_scores(a, model, scaler)
            xs = np.arange(len(sc)) * 5 + SEQ_LEN
            axes[2].plot(xs, sc, color='steelblue', alpha=0.35, lw=0.8)
        for f in failure_files[:6]:
            a = np.load(f).astype(np.float32)
            sc = windowed_scores(a, model, scaler)
            xs = np.arange(len(sc)) * 5 + SEQ_LEN
            axes[2].plot(xs, sc, color='darkorange', alpha=0.35, lw=0.8)
        xs_i = np.arange(len(infer_sc)) * 5 + SEQ_LEN
        axes[2].plot(xs_i, infer_sc, color='crimson', lw=1.8, label='inference')
        axes[2].axhline(file_thr,   color='black',  linestyle='--', lw=1.2,
                        label=f'file_thr={file_thr:.4f}')
        axes[2].axhline(deploy_thr, color='green', linestyle='--', lw=1.2,
                        label=f'deploy_thr={deploy_thr}')
        axes[2].set_xlabel("frame"); axes[2].set_ylabel("score")
        score_handles = handles[:3] + [
            Line2D([0],[0], color='black', lw=1.2, linestyle='--', label=f'file_thr={file_thr:.4f}'),
            Line2D([0],[0], color='green', lw=1.2, linestyle='--', label=f'deploy_thr={deploy_thr}'),
        ]
        axes[2].legend(handles=score_handles, fontsize=8); axes[2].grid(True)

        plt.tight_layout()
        out_path = f"{TRAIN_DIR}/compare_inference_vs_train.png"
        plt.savefig(out_path, dpi=150); plt.close()
        print(f"\n[Plot] 저장: {out_path}")

        # scaled 비교
        plot_scaled_comparison(infer_arr, success_files, failure_files, scaler, TRAIN_DIR,
                               use_hybrid=use_hybrid)

    except Exception as e:
        import traceback; traceback.print_exc()

    # per-taxel raw 비교
    plot_per_taxel(infer_arr, success_files, failure_files, TRAIN_DIR)


if __name__ == "__main__":
    main()
