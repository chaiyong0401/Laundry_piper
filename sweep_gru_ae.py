"""
GRU-AE 하이퍼파라미터 스윕
조합마다 학습 후 윈도우/시퀀스 단위 success/failure 감지율 비교 → 최적 옵션 탐색.

Usage:
  python3 sweep_gru_ae.py                        # grasp + transport 전체
  python3 sweep_gru_ae.py --model grasp
  python3 sweep_gru_ae.py --model transport
"""

import subprocess, sys, re, itertools, argparse
from datetime import datetime

# ── 스윕 대상 옵션 ──────────────────────────────
SWEEP = {
    "relative":     ["none", "window_mean", "first_frame"],
    "weight_alpha": [0.0, 2.0],
    "hybrid":       [False, True],
    "win_stride":   [10],
    "use_raw":      [False, True],
    "noisy_filter": [0.0],
    "score_filter": [0.0, 0.2],
}

MODEL_CONFIGS = {
    "grasp":     {"data_dir": "grasp_new",     "prefix": "grasp"},
    "transport": {"data_dir": "transport_new", "prefix": "transport"},
}

WS = "/home/dyros/mcy_ws"


def run_one(model: str, cfg: dict) -> dict | None:
    """단일 조합 학습 실행 후 stdout 파싱 → 결과 dict 반환."""
    mc = MODEL_CONFIGS[model]
    cmd = [
        sys.executable, f"{WS}/train_gru_ae.py",
        "--model", model,
        "--data_dir", f"{WS}/{mc['data_dir']}",
        "--windowed",
        "--eval",
        "--win_stride",     str(cfg["win_stride"]),
        "--relative",       cfg["relative"],
        "--weight_alpha",   str(cfg["weight_alpha"]),
        "--noisy_filter",   str(cfg["noisy_filter"]),
        "--score_filter",   str(cfg["score_filter"]),
    ]
    if cfg["hybrid"]:
        cmd.append("--hybrid")
    if cfg["use_raw"]:
        cmd.append("--use_raw")

    label = (f"relative={cfg['relative']} alpha={cfg['weight_alpha']} "
             f"hybrid={cfg['hybrid']} stride={cfg['win_stride']} raw={cfg['use_raw']} "
             f"nf={cfg['noisy_filter']} sf={cfg['score_filter']}")
    print(f"\n{'='*65}")
    print(f"[{model}] {label}")
    print(f"{'='*65}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("  [TIMEOUT]")
        return None

    stdout = result.stdout
    if result.returncode != 0:
        print("  [ERROR]")
        print(result.stderr[-500:])
        return None

    # ── 학습 손실
    es_match = re.search(r"\[EarlyStopping\] epoch (\d+), best_val=([\d.]+)", stdout)
    train_match = re.findall(r"train=([\d.]+)\s+val=([\d.]+)", stdout)
    if es_match:
        best_val = float(es_match.group(2))
        stopped_epoch = int(es_match.group(1))
    elif train_match:
        best_val = float(train_match[-1][1])
        stopped_epoch = -1
    else:
        best_val, stopped_epoch = None, None

    # ── 윈도우 단위 감지율 파싱
    # 예: [success] n=220, mean=0.064, anomaly rate = 11/220 = 5.0%
    win_rates = {}
    for lbl in ["success", "failure", "perturbation"]:
        m = re.search(
            rf"\[{lbl}\] n=(\d+), mean=([\d.]+), anomaly rate = (\d+)/(\d+) = ([\d.]+)%",
            stdout
        )
        if m:
            win_rates[lbl] = {"n": int(m.group(1)), "rate": float(m.group(5))}

    win_sep = None
    if "success" in win_rates and "failure" in win_rates:
        win_sep = win_rates["failure"]["rate"] - win_rates["success"]["rate"]

    # ── 시퀀스 단위 consec 결과 파싱
    # 블록 예:
    #   [consec_A=2, consec_B=3]
    #   ...
    #   success   2/21 (  9.5%)  0/21 (  0.0%)  2/21 (  9.5%)
    #   failure   7/7 (100.0%)  2/7 ( 28.6%)  7/7 (100.0%)
    pct = r"[\d.]+"
    row_pat = (
        r"(\w+)\s+"                                          # label
        r"\d+/\d+\s+\(\s*([\d.]+)%\)\s+"                   # Ch-A rate
        r"\d+/\d+\s+\(\s*([\d.]+)%\)\s+"                   # Ch-B rate
        r"\d+/(\d+)\s+\(\s*([\d.]+)%\)"                    # A OR B: total, rate
    )
    consec_blocks = re.findall(
        r"\[consec_A=(\d+), consec_B=(\d+)\](.*?)(?=\[consec_A=|\Z)",
        stdout, re.DOTALL
    )

    seq_results = []
    for ca_str, cb_str, block in consec_blocks:
        ca, cb = int(ca_str), int(cb_str)
        rows = {}
        for m in re.finditer(row_pat, block):
            lbl = m.group(1)
            rows[lbl] = {
                "a":  float(m.group(2)),
                "b":  float(m.group(3)),
                "ab": float(m.group(4)),   # m.group(3) was total; now m.group(4) is rate
            }
        # group 순서: (1)label (2)chA% (3)chB% (4)total (5)ab%  → 수정
        rows = {}
        for m in re.finditer(row_pat, block):
            lbl = m.group(1)
            rows[lbl] = {
                "a":  float(m.group(2)),
                "b":  float(m.group(3)),
                "ab": float(m.group(4)),
            }
        if "success" in rows and "failure" in rows:
            sep = rows["failure"]["ab"] - rows["success"]["ab"]
            seq_results.append({
                "ca": ca, "cb": cb,
                "succ_rate": rows["success"]["ab"],
                "fail_rate": rows["failure"]["ab"],
                "succ_a":    rows["success"]["a"],
                "succ_b":    rows["success"]["b"],
                "fail_a":    rows["failure"]["a"],
                "fail_b":    rows["failure"]["b"],
                "sep": sep,
            })

    # 시퀀스 단위 최적 consec (sep 최대)
    best_seq = max(seq_results, key=lambda r: r["sep"]) if seq_results else None

    row = {
        "model":        model,
        "relative":     cfg["relative"],
        "weight_alpha": cfg["weight_alpha"],
        "hybrid":       cfg["hybrid"],
        "win_stride":   cfg["win_stride"],
        "use_raw":      cfg["use_raw"],
        "noisy_filter": cfg["noisy_filter"],
        "score_filter": cfg["score_filter"],
        "best_val":     best_val,
        "stopped_epoch":stopped_epoch,
        # 윈도우 단위
        "win_succ%":    win_rates.get("success", {}).get("rate"),
        "win_fail%":    win_rates.get("failure", {}).get("rate"),
        "win_sep":      win_sep,
        # 시퀀스 단위 (최적 consec)
        "seq_ca":       best_seq["ca"]        if best_seq else None,
        "seq_cb":       best_seq["cb"]        if best_seq else None,
        "seq_succ%":    best_seq["succ_rate"] if best_seq else None,
        "seq_fail%":    best_seq["fail_rate"] if best_seq else None,
        "seq_sep":      best_seq["sep"]       if best_seq else None,
        # 채널별 breakdown (최적 consec 기준)
        "s_chA%":       best_seq["succ_a"]    if best_seq else None,
        "s_chB%":       best_seq["succ_b"]    if best_seq else None,
        "f_chA%":       best_seq["fail_a"]    if best_seq else None,
        "f_chB%":       best_seq["fail_b"]    if best_seq else None,
    }

    # 요약 출력
    ws_r = win_rates.get("success", {}).get("rate", "?")
    wf_r = win_rates.get("failure", {}).get("rate", "?")
    print(f"  [윈도우] success={ws_r}%  failure={wf_r}%  sep={win_sep}")
    if best_seq:
        print(f"  [시퀀스] best consec_A={best_seq['ca']} consec_B={best_seq['cb']} "
              f"→ success={best_seq['succ_rate']}%  failure={best_seq['fail_rate']}%  "
              f"sep={best_seq['sep']:.1f}%")
        print(f"    success: chA={best_seq['succ_a']:.1f}%  chB={best_seq['succ_b']:.1f}%")
        print(f"    failure: chA={best_seq['fail_a']:.1f}%  chB={best_seq['fail_b']:.1f}%")
    return row


def format_table(rows: list[dict], model: str):
    target = [r for r in rows if r["model"] == model]
    if not target:
        return

    # 시퀀스 단위 sep 기준 정렬 (없으면 윈도우 sep)
    def sort_key(r):
        return r.get("seq_sep") or r.get("win_sep") or -999

    target.sort(key=sort_key, reverse=True)

    W = 135
    print(f"\n{'='*W}")
    print(f"[{model}] 결과 순위  (sep = failure% - success%,  시퀀스 단위 기준)")
    print('='*W)
    hdr = (f"{'relative':<13} {'α':>4} {'hyb':>4} {'str':>4} {'raw':>4} {'nf':>5} {'sf':>5} | "
           f"{'val':>8} | "
           f"{'win_s%':>7} {'win_f%':>7} {'win_sep':>8} | "
           f"{'cA':>3} {'cB':>3} "
           f"{'s_A%':>6} {'s_B%':>6} {'seq_s%':>7} "
           f"{'f_A%':>6} {'f_B%':>6} {'seq_f%':>7} {'seq_sep':>8}")
    print(hdr)
    print('-'*W)
    for r in target:
        print(
            f"{r['relative']:<13} {r['weight_alpha']:>4} "
            f"{str(r['hybrid'])[0]:>4} {r['win_stride']:>4} {str(r['use_raw'])[0]:>4} "
            f"{r.get('noisy_filter', 0):>5.2f} {r.get('score_filter', 0):>5.2f} | "
            f"{(r['best_val'] or 0):>8.5f} | "
            f"{(r['win_succ%'] or 0):>7.1f} {(r['win_fail%'] or 0):>7.1f} "
            f"{(r['win_sep'] or 0):>8.1f} | "
            f"{(r['seq_ca'] or 0):>3} {(r['seq_cb'] or 0):>3} "
            f"{(r['s_chA%'] or 0):>6.1f} {(r['s_chB%'] or 0):>6.1f} {(r['seq_succ%'] or 0):>7.1f} "
            f"{(r['f_chA%'] or 0):>6.1f} {(r['f_chB%'] or 0):>6.1f} {(r['seq_fail%'] or 0):>7.1f} "
            f"{(r['seq_sep'] or 0):>8.1f}"
        )

    best = target[0]
    print(f"\n  ★ 최적: relative={best['relative']}, alpha={best['weight_alpha']}, "
          f"hybrid={best['hybrid']}, stride={best['win_stride']}, raw={best['use_raw']}, "
          f"noisy_filter={best.get('noisy_filter', 0)}, score_filter={best.get('score_filter', 0)}")
    if best.get("seq_sep") is not None:
        print(f"     시퀀스 단위 (consec_A={best['seq_ca']}, consec_B={best['seq_cb']}): "
              f"success 오경보={best['seq_succ%']:.1f}%  "
              f"  [chA={best['s_chA%']:.1f}% chB={best['s_chB%']:.1f}%]")
        print(f"     failure 감지={best['seq_fail%']:.1f}%  sep={best['seq_sep']:.1f}%"
              f"  [chA={best['f_chA%']:.1f}% chB={best['f_chB%']:.1f}%]")
    print(f"     윈도우 단위: success={best['win_succ%']:.1f}%  "
          f"failure={best['win_fail%']:.1f}%  sep={best['win_sep']:.1f}%")


def save_csv(rows: list[dict], path: str):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"\n[CSV] 저장: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=["all", "grasp", "transport"])
    args = parser.parse_args()

    models = list(MODEL_CONFIGS) if args.model == "all" else [args.model]

    keys   = list(SWEEP.keys())
    combos = list(itertools.product(*SWEEP.values()))
    total  = len(combos) * len(models)

    print(f"총 {total}개 조합 ({len(combos)} combos × {len(models)} models)")
    print(f"시작: {datetime.now().strftime('%H:%M:%S')}\n")

    rows = []
    idx  = 0
    for model in models:
        for combo in combos:
            idx += 1
            cfg = dict(zip(keys, combo))
            print(f"\n[{idx}/{total}]", end="")
            row = run_one(model, cfg)
            if row:
                rows.append(row)

    ts = datetime.now().strftime("%H%M%S")
    save_csv(rows, f"{WS}/sweep_results_{ts}.csv")

    for model in models:
        format_table(rows, model)

    print(f"\n완료: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
