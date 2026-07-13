"""
Dual-Channel Anomaly Detector
==============================
Channel A (slow): GRU-AE window MSE  → 점진적 slip 감지
Channel B (fast): 프레임별 MRL delta → 충격성 drop 감지

6D feature 순서: [mrl(0), norm(1), max_norm(2), unit_x(3), unit_y(4), unit_z(5)]
"""

from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch

from .model import GRUAE
from .preprocess import apply_relative_transform, build_hybrid_window


# ──────────────────────────────────────────────
#  Channel B
# ──────────────────────────────────────────────
class FramewiseDeltaDetector:
    """
    연속 프레임 간 MRL 변화량으로 충격성 이상 감지.
    smooth_window 프레임 평균을 기준점으로 사용.
    """

    MRL_IDX = 0

    def __init__(self, delta_threshold: float, smooth_window: int = 3):
        self.delta_threshold = delta_threshold
        self.smooth_window = smooth_window
        self._history: deque = deque(maxlen=smooth_window)
        self._prev_mrl: Optional[float] = None

    def reset(self):
        self._history.clear()
        self._prev_mrl = None

    @staticmethod
    def _compute_mrl(frame: np.ndarray) -> float:
        if frame.shape[0] == 6:
            return float(frame[0])
        # raw 48D: x0,y0,z0, x1,y1,z1, ..., x15,y15,z15
        vecs = frame[:48].reshape(16, 3)
        norms = np.linalg.norm(vecs, axis=1)
        valid_mask = norms > 0.1
        if valid_mask.sum() >= 2:
            unit_vecs = vecs[valid_mask] / norms[valid_mask, np.newaxis]
            mean_unit = np.mean(unit_vecs, axis=0)
            return float(np.linalg.norm(mean_unit))
        else:
            return 0.0

    def update(self, frame: np.ndarray) -> tuple[float, bool]:
        """
        frame: (D,) raw feature
        Returns: (delta, is_anomaly)
        """
        mrl = self._compute_mrl(frame)

        if self._prev_mrl is None:
            self._prev_mrl = mrl
            self._history.append(mrl)
            return 0.0, False

        ref = float(np.mean(self._history)) if len(self._history) == self.smooth_window \
              else self._prev_mrl

        delta = abs(mrl - ref)
        self._history.append(mrl)
        self._prev_mrl = mrl

        return delta, delta > self.delta_threshold


# ──────────────────────────────────────────────
#  Dual-Channel Detector
# ──────────────────────────────────────────────
class DualChannelDetector:
    """
    실시간 촉각 이상 감지기.

    Parameters
    ----------
    model       : 학습된 GRUAE 인스턴스
    scaler      : sklearn StandardScaler
    threshold_a : Channel A (GRU-AE) 이상 판정 임계값
    threshold_b : Channel B (MRL delta) 이상 판정 임계값
    infer_cfg   : dict with keys:
                    seq_len      (int,   default 50)
                    relative     (str,   default 'none')
                    weight_alpha (float, default 0.0)
                    hybrid       (bool,  default False)
                    n_rel_dims   (int|None, default None)
    device      : torch device (None → auto)
    smooth_window_b : Channel B smoothing window (default 3)
    """

    def __init__(
        self,
        model: GRUAE,
        scaler,
        threshold_a: float,
        threshold_b: float,
        infer_cfg: Optional[dict] = None,
        device: Optional[torch.device] = None,
        smooth_window_b: int = 3,
    ):
        self.model = model
        self.scaler = scaler
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b

        cfg = infer_cfg or {}
        self.seq_len      = int(cfg.get('seq_len', 50))
        self.relative     = str(cfg.get('relative', 'none'))
        self.weight_alpha = float(cfg.get('weight_alpha', 0.0))
        self.hybrid       = bool(cfg.get('hybrid', False))
        self.n_rel_dims   = cfg.get('n_rel_dims', None)
        self.mrl_thresh        = float(cfg.get('mrl_thresh', 0.3))
        self.cutoff_multiplier = float(cfg.get('cutoff_multiplier', 2.2))

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device).eval()

        self._buffer: deque = deque(maxlen=self.seq_len)
        self.channel_b = FramewiseDeltaDetector(threshold_b, smooth_window_b)
        self._max_mrl_seen = 0.0

    # ── 로드 헬퍼 ──────────────────────────────
    @classmethod
    def from_checkpoint(
        cls,
        model_path: str,
        scaler_path: str,
        threshold_a_path: str,
        threshold_b_path: str,
        infer_cfg_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> "DualChannelDetector":
        """

        Parameters
        ----------
        model_path       : *_gru_ae.pt
        scaler_path      : *_scaler.pkl
        threshold_a_path : *_threshold.npy
        threshold_b_path : *_threshold_b.npy
        infer_cfg_path   : *_infer_cfg.pkl  (없으면 기본값 사용)
        device           : None → auto
        """
        dev = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        infer_cfg: dict = {}
        if infer_cfg_path and Path(infer_cfg_path).exists():
            with open(infer_cfg_path, 'rb') as f:
                infer_cfg = pickle.load(f)

        scaler = joblib.load(scaler_path)

        input_dim = scaler.mean_.shape[0]
        seq_len   = int(infer_cfg.get('seq_len', 50))

        model = GRUAE(
            input_dim  = input_dim,
            hidden_dim = 64,
            num_layers = 2,
            dropout    = 0.2,
            seq_len    = seq_len,
        )
        state = torch.load(model_path, map_location=dev, weights_only=True)
        model.load_state_dict(state)

        threshold_a = float(np.load(threshold_a_path)[0])
        threshold_b = float(np.load(threshold_b_path)[0])

        return cls(
            model       = model,
            scaler      = scaler,
            threshold_a = threshold_a,
            threshold_b = threshold_b,
            infer_cfg   = infer_cfg,
            device      = dev,
        )

    def reset(self):
        """에피소드 시작 시 호출. 내부 버퍼와 Channel B 히스토리를 초기화."""
        self._buffer.clear()
        self.channel_b.reset()
        self._max_mrl_seen = 0.0

    def update(self, frame: np.ndarray,
               raw_frame: Optional[np.ndarray] = None) -> dict:
        """
        매 프레임 호출. 두 채널을 동시에 평가하여 결과를 반환.

        Parameters
        ----------
        frame     : (6,) 6D feature [mrl, norm, max_norm, unit_x, unit_y, unit_z]
                    Channel B (MRL delta)에 사용. 항상 6D여야 함.
        raw_frame : (48,) 48D raw [cum_x*16, cum_y*16, cum_z*16], optional.
                    제공되면 Channel A (GRU-AE)에 사용.
                    None이면 frame을 Channel A에도 사용 (6D 모델 호환).

        Returns
        -------
        dict with keys:
            anomaly  (bool)  : 이상 감지 여부
            channel  (str)   : 'A' | 'B' | 'AB' | 'none'
            score_a  (float) : Channel A reconstruction error (0 if buffer not full)
            delta_b  (float) : Channel B MRL delta
        """
        # ── Channel B: 항상 6D frame[0] (MRL) 사용
        delta_b, flag_b = self.channel_b.update(frame)

        # ── Channel A: raw_frame 우선(학습 차원 매칭 시), 없으면 frame 사용
        expected_dim = self.scaler.mean_.shape[0]
        if raw_frame is not None and len(raw_frame) == expected_dim:
            frame_a = raw_frame
        else:
            frame_a = frame
        self._buffer.append(frame_a)
        score_a, flag_a = 0.0, False

        # MRL (index 0) 최대값 추적 (Miss Guard Rail 용)
        mrl_val = float(frame[0])
        self._max_mrl_seen = max(self._max_mrl_seen, mrl_val)

        if len(self._buffer) == self.seq_len:
            # Rule 1: Miss Guard Rail (MRL threshold)
            if self._max_mrl_seen < self.mrl_thresh:
                flag_a = True
            else:
                window = np.array(self._buffer, dtype=np.float32)

                if self.hybrid:
                    window = build_hybrid_window(window)

                x = self.scaler.transform(window).astype(np.float32)
                x = apply_relative_transform(x, self.relative, self.n_rel_dims)
                x_t = torch.tensor(x).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    score_a = float(
                        self.model.reconstruction_error(x_t, self.weight_alpha).item()
                    )
                
                # Rule 2 & 3: Super High Error Cutoff & Regular Threshold
                if score_a > self.threshold_a * self.cutoff_multiplier:
                    flag_a = True
                elif score_a > self.threshold_a:
                    flag_a = True

        if flag_a and flag_b:
            channel = 'AB'
        elif flag_a:
            channel = 'A'
        elif flag_b:
            channel = 'B'
        else:
            channel = 'none'

        return {
            'anomaly': flag_a or flag_b,
            'channel': channel,
            'score_a': score_a,
            'delta_b': delta_b,
        }


def calibrate_threshold_b(
    success_sequences: list[np.ndarray],
    smooth_window: int = 3,
    percentile: int = 99,
    warmup: int = 30,
) -> float:
    """
    Parameters
    ----------
    success_sequences : list of (T, D) raw feature arrays (scaler 적용 전)
    smooth_window     : smoothing window for delta calculation
    percentile        : threshold = N-th percentile of delta distribution
    warmup            : number of warmup frames to skip at sequence start

    Returns
    -------
    threshold_b (float)
    """
    deltas = []
    for seq in success_sequences:
        mrl_series = np.array([FramewiseDeltaDetector._compute_mrl(seq[t]) for t in range(len(seq))])
        for t in range(max(smooth_window, warmup), len(mrl_series)):
            ref = float(mrl_series[t - smooth_window:t].mean())
            deltas.append(abs(float(mrl_series[t]) - ref))

    if not deltas:
        print("[calibrate_threshold_b] delta 데이터 없음 → 기본값 0.5 사용")
        return 0.5

    deltas = np.array(deltas)
    threshold_b = float(np.percentile(deltas, percentile))
    print(f"[calibrate_threshold_b] delta: mean={deltas.mean():.4f}, "
          f"std={deltas.std():.4f}, max={deltas.max():.4f}")
    print(f"[calibrate_threshold_b] threshold ({percentile}th pct) = {threshold_b:.4f}")
    return threshold_b
