# train.py
# 실제 LSTM 학습 돌리는 코드
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# helpers
# -----------------------------
def _to_str_1d(arr: np.ndarray) -> np.ndarray:
    """bytes/object -> np.str_ (1D)"""
    flat = arr.reshape(-1).tolist()
    out = []
    for x in flat:
        if isinstance(x, (bytes, bytearray)):
            out.append(x.decode("utf-8", errors="ignore"))
        else:
            out.append(str(x))
    return np.asarray(out, dtype=np.str_)

def _train_labels_to_int(train_labels_str: np.ndarray) -> np.ndarray:
    """
    train_labels_str: (N,) dtype str
    return: (N,) int32, IGNORE=-100
    """
    y = np.full((train_labels_str.shape[0],), -100, dtype=np.int32)
    y[train_labels_str == "NC"] = 0
    y[train_labels_str == "G"]  = 1
    y[train_labels_str == "C"]  = 2
    # 나머지는 -100 유지
    return y

def assert_trainlabel_coverage_or_raise(
    base_dir: Path,
    split: str,
    ep_name: str,
    items: List[Dict],
) -> Tuple[int, np.ndarray]:
    """
    val/test에서만 사용 권장.
    - episode 원본(train_labels)에서 need mask 생성
    - window start_idx + L로 covered 생성
    - need & ~covered 가 존재하면 raise

    return:
      N (episode length),
      gt_full_int (N,) with IGNORE=-100  (평가에서 그대로 사용)
    """
    ep_path = base_dir / split / ep_name
    if not ep_path.exists():
        raise FileNotFoundError(f"[COVERAGE] missing episode file: {ep_path}")

    with np.load(ep_path, allow_pickle=True) as d:
        if "train_labels" not in d:
            raise KeyError(f"[COVERAGE] {ep_name}: missing key train_labels")
        y_train_str = _to_str_1d(np.asarray(d["train_labels"]))  # (N,)
    N = int(y_train_str.shape[0])
    gt_full_int = _train_labels_to_int(y_train_str)  # (N,)

    need = (gt_full_int != -100)  # train_labels 기준 non-ignore

    # covered[t] = 어떤 윈도우라도 t를 포함하면 True
    covered = np.zeros((N,), dtype=bool)

    # 윈도우 길이 L은 items마다 동일하다고 가정하지만, 안전하게 각 윈도우별로 처리
    for it in items:
        s = int(it["start_idx"])
        Lw = int(it["y"].shape[0])  # window length
        e = min(N, s + Lw)
        if e > s:
            covered[s:e] = True

    missing = need & (~covered)
    if missing.any():
        idx = np.where(missing)[0]
        # 디버깅에 필요한 최소 정보만
        head = idx[:20].tolist()
        raise RuntimeError(
            f"[COVERAGE ERROR] split={split} ep={ep_name} | "
            f"missing_trainlabel_frames={int(missing.sum())}/{int(need.sum())} | "
            f"missing_idx_head={head}"
        )

    return N, gt_full_int


# -----------------------------
# Label mapping
# -----------------------------
LABEL2ID = {"NC": 0, "G": 1, "C": 2, "IGNORE": -100}
ID2LABEL = {0: "NC", 1: "G", 2: "C"}

# -----------------------------
# center_mask (수식 구현): 전역 중앙 c 기준으로 h를 늘려 V(h) >= M_min 될 때까지 확장
# -----------------------------
def make_center_mask_expand(
    base_mask: torch.Tensor,  # (B,L) bool
    h0: int,
    m_min: int,
    delta: int = 1,
    h_max: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    return:
      center_mask: (B,L) bool   (chi_t(h*)) : 각 샘플마다 "채택된 중앙 구간"만 True인 마스크, 조건 실패한 샘플은 전부 False
      ok:          (B,) bool    (V(h*) >= M_min 만족 여부) : 그 샘플이 Mmin개 이상의 유효 프레임을 중앙 구간에서 확보했는지 여부
      Vstar:       (B,) int64   (선택된 h*에서의 유효프레임 수 V(h*)) : 최종 선택된 중앙 구간에서의 유효 프레임 수 v
    """
    assert base_mask.ndim == 2
    B, L = base_mask.shape
    device = base_mask.device

    if h_max is None:
        h_max = L // 2
    h0 = max(0, int(h0))
    delta = max(1, int(delta))
    m_min = max(1, int(m_min))

    c = L // 2

    center_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
    ok = torch.zeros((B,), dtype=torch.bool, device=device)
    Vstar = torch.zeros((B,), dtype=torch.int64, device=device)

    for b in range(B):
        found = False
        h = h0
        while h <= h_max:
            s = max(0, c - h)
            e = min(L, c + h + 1)  # end-exclusive
            v = int(base_mask[b, s:e].sum().item())  # V(h)
            if v >= m_min:
                center_mask[b, s:e] = True  # chi_t(h*)
                ok[b] = True
                Vstar[b] = v
                found = True
                break
            h += delta

        if not found:
            # 끝까지 늘려도 V(h) < M_min 이면:
            # - center_mask는 전부 False 유지
            # - ok=False
            # - Vstar=0
            pass

    return center_mask, ok, Vstar

class EpisodeIndexDataset(Dataset):
    """
    episode 기반 index dataset.
    split 폴더 아래:
        train/index/selected.npz
        val/index/selected.npz
        test/index/selected.npz

    각 index 파일에는:
        episode_name
        start_idx
    """

    def __init__(self, base_dir: Path, split: str, window_length: int):
        self.base_dir = base_dir
        self.split = split
        self.window_length = int(window_length)

        self.split_dir = base_dir / split
        self.index_npz = self.split_dir / "index" / "selected.npz"

        if not self.index_npz.exists():
            raise FileNotFoundError(f"missing: {self.index_npz}")

        with np.load(self.index_npz, allow_pickle=True) as d:
            self.ep_names = d["episode_name"].astype(np.str_)
            self.start_idx = d["start_idx"].astype(np.int64)

        if len(self.ep_names) != len(self.start_idx):
            raise ValueError("index length mismatch")

        # episode 캐시
        self._ep_cache: Dict[str, Dict[str, np.ndarray]] = {}

    def __len__(self):
        return len(self.ep_names)

    def _load_episode(self, ep_name: str):
        if ep_name in self._ep_cache:
            return self._ep_cache[ep_name]

        fp = self.split_dir / ep_name
        if not fp.exists():
            raise FileNotFoundError(f"missing episode: {fp}")

        with np.load(fp, allow_pickle=True) as d:
            data = {
                "X": np.asarray(d["X"], dtype=np.float32),  # (N,6,4,4)
                "labels": np.asarray(d["train_labels"]),
            }

        self._ep_cache[ep_name] = data
        return data

    def __getitem__(self, i):
        ep = str(self.ep_names[i])
        s = int(self.start_idx[i])
        e = s + self.window_length

        data = self._load_episode(ep)

        X_full = data["X"]
        y_full_str = data["labels"]

        if e > len(X_full):
            raise ValueError(
                f"window out of range: {ep} start={s} len={self.window_length}"
            )

        Xw = X_full[s:e]
        yw_str = y_full_str[s:e]

        y_int = _train_labels_to_int(yw_str)

        return {
            "X": torch.from_numpy(Xw),
            "y": torch.from_numpy(y_int),
            "episode_name": ep,
            "start_idx": s,
        }
    
def collate_fn(batch: List[Dict]):
    '''
    Pytorch DataLoader가 여러 samle을 배치로 합치는 방법
    X = (B, L, 6, 4, 4)
    y = (B, L)
    로 stack
    '''
    X = torch.stack([b["X"] for b in batch], dim=0)  # (B, L, 6, 4, 4)
    y = torch.stack([b["y"] for b in batch], dim=0)  # (B, L)
    ep = [b["episode_name"] for b in batch]
    st = [b["start_idx"] for b in batch]
    return {"X": X, "y": y, "episode_name": ep, "start_idx": st}

# -----------------------------
# Simple model (baseline)
# -----------------------------
class SimpleTactileLSTM(nn.Module):
    """
    입력: (B, L, 6, 4, 4)
    출력: (B, L, 3) logits
    """
    def __init__(self, hidden: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        in_dim = 6 * 4 * 4
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C, H, W = x.shape
        x = x.view(B, L, C * H * W)
        h, _ = self.lstm(x)
        logits = self.head(h)  # (B, L, 3)
        return logits


# -----------------------------
# Loss + frame metrics
# -----------------------------
@torch.no_grad()
def confusion_update(conf: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    프레임 단위 예측 결과를 누적 혼동행렬에 반영.
    rows = 정답 클래스 | cols = 예측 클래스 
    """
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        conf[int(t), int(p)] += 1


@torch.no_grad()
def precision_recall_from_conf(conf: torch.Tensor) -> Tuple[Dict[str, float], Dict[str, float]]:
    '''
    누적된 혼동행렬부터 클래스별 precision/recall 계산
    '''
    prec = {}
    rec = {}
    for c in range(3):
        tp = conf[c, c].item()
        fp = conf[:, c].sum().item() - tp
        fn = conf[c, :].sum().item() - tp
        prec[ID2LABEL[c]] = tp / (tp + fp + 1e-9)
        rec[ID2LABEL[c]] = tp / (tp + fn + 1e-9)
    return prec, rec


def masked_ce_loss_adaptive_center(
    logits: torch.Tensor,  # (B,L,3)
    y: torch.Tensor,       # (B,L)  IGNORE=-100
    h0: int,
    m_min: int,
    delta: int = 1,
    h_max: int | None = None,
    eps: float = 1e-8,
    skip_if_insufficient: bool = True,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    네 수식 구현:
      IGNORE(-100) 프레임을 제외하면서, 중앙 주변 구간을 샘플마다 자동으로 늘려서 
      최소 유효프레임 Mmin개를 확보한 뒤 그 구간에서만 Cross-Entropy를 평균내는 loss를 계산
    """
    assert logits.ndim == 3 and y.ndim == 2
    B, L, C = logits.shape
    assert C == 3, "logits last dim should be 3 (NC/G/C)"

    base_mask = (y != -100)  # m_t

    # chi_t(h*)를 구하는 부분 (전역 중앙 기준 확장)
    #print("mmin in masked_ce_loss_adaptive_center 함수", m_min)
    center_mask, ok, Vstar = make_center_mask_expand(
        base_mask=base_mask,
        h0=h0,
        m_min=m_min,
        delta=delta,
        h_max=h_max,
    )

    # w_t = m_t * chi_t(h*)
    final_mask = base_mask & center_mask  # (B,L)
    S = final_mask.sum(dim=1)            # (B,) 유효 프레임 수

    # 프레임별 CE(reduction='none')
    ce = nn.functional.cross_entropy(
        logits.reshape(-1, 3),
        y.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(B, L)  # (B,L)

    # 샘플별 loss
    loss_per_sample = (ce * final_mask.float()).sum(dim=1) / (S.float() + eps)  # (B,)

    if skip_if_insufficient:
        keep = ok & (S > 0)
        if keep.any():
            loss = loss_per_sample[keep].mean()
        else:
            loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        skipped_count = (~keep).sum()
    else:
        keep = ok & (S > 0)
        lp = loss_per_sample.clone()
        lp[~keep] = 0.0
        loss = lp.mean()
        skipped_count = (~keep).sum()

    info = {
        "base_mask": base_mask,
        "center_mask": center_mask,
        "final_mask": final_mask,
        "ok": ok,
        "Vstar": Vstar,
        "S": S,
        "skipped_count": skipped_count,  # ✅ torch scalar tensor
        "keep": keep,                    # ✅ (B,)
        "loss_per_sample": loss_per_sample,
    }
    return loss, info


def run_length_encode(seq: List[int]) -> List[Tuple[int, int]]:
    """
    연속한 동일 라벨 구간을 (라벨, 길이)로 압축
    run 단위로 보고 "얼마나 지속됐는지"를 알기 위해 필요
    """
    if not seq:
        return []
    out = []
    cur = seq[0]
    cnt = 1
    for x in seq[1:]:
        if x == cur:
            cnt += 1
        else:
            out.append((cur, cnt))
            cur = x
            cnt = 1
    out.append((cur, cnt))
    return out


def tau_filter(rle: List[Tuple[int, int]], tau: int) -> List[int]:
    """
    tau 필터: 길이 < tau 인 run 제거.
    제거 후 인접 run이 같은 label이면 병합.
    """
    if tau <= 1:
        # 그대로 확장
        out = []
        for lab, ln in rle:
            out.extend([lab] * ln)
        return out

    kept = [(lab, ln) for (lab, ln) in rle if ln >= tau]
    # 병합
    merged: List[Tuple[int, int]] = []
    for lab, ln in kept:
        if merged and merged[-1][0] == lab:
            merged[-1] = (lab, merged[-1][1] + ln)
        else:
            merged.append((lab, ln))
    out = []
    for lab, ln in merged:
        out.extend([lab] * ln)
    return out

def compress_to_run_labels(seq):
    rle = run_length_encode(seq)
    return [lab for (lab, _) in rle]

# -----------------------------
# Train / Eval loops
# -----------------------------
@dataclass
class CheckpointScore:
    '''
    한 epoch의 validation 결과를 요약한 점수 묶음
    '''
    scenario_exact: float # 에피소드 단위 event-match 정확도
    recall_c: float # 클래스 C에 대한 recall
    val_loss: float # validation loss

    def better_than(self, other: Optional["CheckpointScore"]) -> bool:
        if other is None:
            return True
        # 1순위: scenario exact-match
        if self.scenario_exact != other.scenario_exact:
            return self.scenario_exact > other.scenario_exact
        # 2순위: recall(C)
        if self.recall_c != other.recall_c:
            return self.recall_c > other.recall_c
        # 3순위: val loss 낮은 쪽
        return self.val_loss < other.val_loss


def save_checkpoint(path: Path, model: nn.Module, optim: torch.optim.Optimizer, epoch: int, score: CheckpointScore, cfg: Dict):
    '''
    지금 epoch의 모델을 디스크에 저장
    '''
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "score": {
            "scenario_exact": score.scenario_exact,
            "recall_c": score.recall_c, 
            "val_loss": score.val_loss,
        },
        "config": cfg,
    }
    torch.save(payload, str(path))

def train_one_epoch(model, loader, optim, device, args):
    model.train()
    total_loss = 0.0
    total_batches = 0

    # 로그용: S 평균(keep 기준) + skip 비율
    sum_S_kept = 0.0
    cnt_kept = 0
    skipped_total = 0
    seen_total = 0

    for i, batch in enumerate(loader):
        X = batch["X"].to(device)
        y = batch["y"].to(device).long()

        logits = model(X)

        loss, info = masked_ce_loss_adaptive_center(
            logits, y,
            h0=args.h0,
            m_min=args.m_min_frames,
            delta=args.delta,
            h_max=args.h_max,
            skip_if_insufficient=True,
        )
        final_mask = info["final_mask"]  # 필요하면 사용

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        total_loss += float(loss.item())
        total_batches += 1

        # ---- 로그 집계 ----
        keep = info["keep"] # 배치 평균에 포함된 샘플 마스크
        S = info["S"] # 각 샘플에서 최종적으로 사용된 유효 프레임 수
        if keep.any():
            sum_S_kept += float(S[keep].float().sum().item())
            cnt_kept += int(keep.sum().item())

        skipped_total += int(info["skipped_count"].item())
        seen_total += int(keep.numel())

    train_loss = total_loss / max(1, total_batches)
    S_mean = (sum_S_kept / max(1, cnt_kept))  # keep 샘플당 평균 S
    skip_ratio = (skipped_total / max(1, seen_total))

    return {
        "train_loss": float(train_loss), # 배치 loss 평균 ( skip이 많으면 학습 의미가 약해짐 )
        "S_mean": float(S_mean), # 평균적으로 샘플당 몇 프레임이 loss에 쓰였는지
        "skip_ratio": float(skip_ratio), # 배치 내 샘플 중 skip된 비율
    }

# -----------------------------
# Validation
# -----------------------------
@torch.no_grad()
def forward_chunkwise(model, X_full: torch.Tensor, device, chunk_size: int = 4096):
    """
    X_full: (N, 6, 4, 4)
    return:
        logits_full: (1, N, 3)
    """
    model.eval()

    N = X_full.shape[0]
    logits_chunks = []

    for s in range(0, N, chunk_size):
        e = min(N, s + chunk_size)
        X_chunk = X_full[s:e].unsqueeze(0).to(device)  # (1, L, 6, 4, 4)
        logits_chunk = model(X_chunk)  # (1, L, 3)
        logits_chunks.append(logits_chunk.cpu())

    logits_full = torch.cat(logits_chunks, dim=1)  # (1, N, 3)
    return logits_full


@torch.no_grad()
def evaluate_validation(model, base_dir: Path, device, args):
    """
    Episode-wise validation
    Returns:
        dict with:
            val_loss
            recall_c
            scenario_exact
    """
    val_dir = base_dir / "val"
    episode_files = sorted(list(val_dir.glob("*.npz")))

    if len(episode_files) == 0:
        raise RuntimeError("No validation episodes found.")

    total_loss = 0.0
    num_episode = 0

    conf = torch.zeros((3, 3), dtype=torch.long)

    correct_episode = 0

    for ep_path in episode_files:
        with np.load(ep_path, allow_pickle=True) as d:
            X_full = torch.from_numpy(np.asarray(d["X"], dtype=np.float32))
            y_full_str = np.asarray(d["train_labels"])

        y_full = torch.from_numpy(_train_labels_to_int(y_full_str)).long()

        # ---- forward (chunk-wise) ----
        logits_full = forward_chunkwise(
            model, X_full, device, chunk_size=4096
        )  # (1, N, 3)

        y_full = y_full.unsqueeze(0).to(device).long()  # (1, N)

        # ---- validation loss: full valid frames ----
        logits_flat = logits_full.reshape(-1, 3)     # (N, 3)
        y_flat = y_full.reshape(-1)                  # (N,)

        loss = nn.functional.cross_entropy(
            logits_flat,
            y_flat,
            ignore_index=-100,   # invalid frame 제외
            reduction="mean",
        )

        total_loss += float(loss.item())

        # ---- frame-wise prediction ----
        pred = torch.argmax(logits_full, dim=-1).squeeze(0)  # (N,)
        gt = y_full.squeeze(0).cpu()
        # IGNORE 제거
        valid_mask = (gt != -100)
        pred_valid = pred[valid_mask]
        gt_valid = gt[valid_mask]

        if len(gt_valid) > 0:
            confusion_update(conf, gt_valid, pred_valid)

        # ---- scenario exact match ----
        gt_seq = gt_valid.tolist()
        pred_seq = pred_valid.tolist()

        gt_rle = run_length_encode(gt_seq)
        pred_rle = run_length_encode(pred_seq)
        print(compress_to_run_labels(gt_rle))
        print(compress_to_run_labels(pred_rle))

        gt_tau = tau_filter(gt_rle, args.tau_frames)
        pred_tau = tau_filter(pred_rle, args.tau_frames)

        gt_comp = compress_to_run_labels(gt_tau)
        pred_comp = compress_to_run_labels(pred_tau)

        if gt_comp == pred_comp:
            correct_episode += 1
            print(f"[SUCCEED] : Correct = Pred = {pred_comp}")
        else:
            print(f"Scenario exact match failed : Correct({gt_comp} \n Pred({pred_comp}))")

        num_episode += 1

    # ---- final metrics ----
    val_loss = total_loss / max(1, num_episode)

    _, rec = precision_recall_from_conf(conf)
    recall_c = rec["C"]

    scenario_exact = correct_episode / max(1, num_episode)

    return {
        "val_loss": float(val_loss),
        "recall_c": float(recall_c),
        "scenario_exact": float(scenario_exact),
    }
    

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True, help="data/preprocessed 경로")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--val_batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    
    # timebase
    ap.add_argument("--samplerate", type=int, default=500, help="sampling rate(Hz)")
    ap.add_argument("--l", type=float, default=0.6, help="window length in seconds (L = samplerate*l)")
    ap.add_argument("--mmin", type=float, default=0.3,
                    help="minimum valid frames in seconds(m_min = samplerate*mmin)."
                    "default: 0.3 (i.e., half window)")
    ap.add_argument("--tau", type=float, default=0.05, help="tau filter in seconds (tau = samplerate*tau)")

    # model
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    # loss mask params
    ap.add_argument("--h0", type=int, default=0)
    ap.add_argument("--delta", type=int, default=5)
    ap.add_argument("--h_max", type=int, default=None)

    # save
    ap.add_argument("--out_dir", type=str, default="./runs/lstm")
    ap.add_argument("--save_best_name", type=str, default="best.pt")

    args = ap.parse_args()

    
    args.SAMPLING_RATE = int(args.samplerate)
    args.L_frames = int(round(args.SAMPLING_RATE * float(args.l)))
    args.m_min_frames = int(round(args.SAMPLING_RATE * float(args.mmin)))
    args.tau_frames = int(round(args.SAMPLING_RATE * float(args.tau)))
    
    if args.L_frames <= 0:
        raise ValueError(f"L_frames must be > 0, got {args.L_frames} (samplerate={args.samplerate}, l={args.l})")
    if args.m_min_frames <= 0:
        raise ValueError(f"m_min_frames must be > 0, got {args.m_min_frames} (mmin={args.mmin})")
    if args.m_min_frames > args.L_frames:
        raise ValueError(
            f"m_min_frames({args.m_min_frames}) cannot exceed L_frames({args.L_frames}). "
            f"Reduce --mmin or increase --l."
        )
    if args.tau_frames < 1:
        args.tau_frames = 1
    print("[INFO] mmin frames:", args.m_min_frames)
    # ---- seed ----
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_dir = base_dir / "val"
    episode_files = sorted(list(val_dir.glob("*.npz")))

    if len(episode_files) == 0:
        raise RuntimeError("No validation episodes found.")

    # 🔥 첫 번째 episode 파일
    first_ep = episode_files[0]

    with np.load(first_ep, allow_pickle=True) as d:
        # labels.npy가 별도 key라면 아래처럼
        if "labels" in d:
            labels_str = np.asarray(d["labels"])
        if "train_labels" in d:
            train_labels_str = np.asarray(d["train_labels"])
        else:
            raise KeyError("No labels key found in npz file.")

    # int 변환
    labels_int = _train_labels_to_int(labels_str)
    train_labels_int = _train_labels_to_int(train_labels_str)

    # 0 존재 여부 확인
    has_zero = (labels_int == 0).any()
    train_has_zero = (train_labels_int == 0).any()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # ---- datasets / loaders ----
    train_ds = EpisodeIndexDataset(
        base_dir=base_dir,
        split="train",
        window_length=args.L_frames,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

        
    # ✅ (추천) 로드된 윈도우 길이와 args.L_frames 일치 검증
    # 샘플 하나만 뽑아서 길이 확인
    sample0 = train_ds[0]
    L_loaded = int(sample0["y"].shape[0])
    if L_loaded != args.L_frames:
        raise RuntimeError(
            f"[L MISMATCH] windows length={L_loaded} but args expects L_frames={args.L_frames}. "
            f"Fix windows generation or adjust --samplerate/--l."
        )

    # ---- model / optim ----
    model = SimpleTactileLSTM(
        hidden=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best: Optional[CheckpointScore] = None
    best_path = out_dir / args.save_best_name

    # config 저장(재현성)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    # ---- train loop ----
    for epoch in range(1, args.epochs + 1):

        print(f"\n===== Epoch {epoch} =====")

        tr = train_one_epoch(model, train_loader, optim, device, args)

        print(f"[Train] loss={tr['train_loss']:.6f} "
            f"S_mean={tr['S_mean']:.2f} "
            f"skip_ratio={tr['skip_ratio']:.3f}")

        # ---- Validation ----
        val = evaluate_validation(model, base_dir, device, args)

        print(f"[Val] loss={val['val_loss']:.6f} "
            f"recall_C={val['recall_c']:.4f} "
            f"scenario_exact={val['scenario_exact']:.4f}")

        # ---- checkpoint scoring ----
        score = CheckpointScore(
            scenario_exact=val["scenario_exact"],
            recall_c=val["recall_c"],
            val_loss=val["val_loss"],
        )

        if score.better_than(best):
            best = score
            save_checkpoint(
                best_path,
                model,
                optim,
                epoch,
                score,
                vars(args),
            )
            print("✅ Saved new best model")

if __name__ == "__main__":
    main()