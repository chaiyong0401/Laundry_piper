"""
Dual-Channel Anomaly Detector for Tactile-Based Grasp State Estimation
=======================================================================
채널 A (slow): GRU-AE window MSE  → 점진적 slip 감지
채널 B (fast): 프레임별 MRL delta → 충격성 drop 감지

6D feature 순서: [mrl(0), norm(1), max_norm(2), unit_x(3), unit_y(4), unit_z(5)]
"""

import numpy as np
from collections import deque
import torch


# ──────────────────────────────────────────────
#  Channel B: 프레임별 MRL delta 감지기
# ──────────────────────────────────────────────
class FramewiseDeltaDetector:
    """
    연속 프레임 간 MRL 변화량으로 충격성 이상 감지.
    smooth_window 프레임 평균을 기준점으로 사용해 단발 노이즈를 억제.

    6D:  frame[0] = MRL (precomputed)
    48D raw: interleaved x0,y0,z0,...,x15,y15,z15 → taxel별 norm 평균으로 MRL 계산
    """

    MRL_IDX = 0  # 6D feature 중 MRL 위치

    def __init__(self, delta_threshold: float, smooth_window: int = 3):
        self.delta_threshold = delta_threshold
        self.smooth_window = smooth_window
        self._history = deque(maxlen=smooth_window)
        self._prev_mrl = None

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
        frame: (D,) raw feature (scaler 적용 전)
        반환: (delta, is_anomaly)
        """
        mrl = self._compute_mrl(frame)

        if self._prev_mrl is None:
            self._prev_mrl = mrl
            self._history.append(mrl)
            return 0.0, False

        if len(self._history) == self.smooth_window:
            ref = float(np.mean(self._history))
        else:
            ref = self._prev_mrl

        delta = abs(mrl - ref)
        self._history.append(mrl)
        self._prev_mrl = mrl

        return delta, delta > self.delta_threshold


# ──────────────────────────────────────────────
#  Dual-Channel Detector (독립 사용 / 분석용)
# ──────────────────────────────────────────────
class DualChannelDetector:
    """
    채널 A: GRU-AE window MSE  (점진적 이상)
    채널 B: 프레임별 MRL delta (충격성 이상)

    basket_ori.py 통합 시에는 스레드 분리를 위해
    FramewiseDeltaDetector를 직접 사용하고,
    이 클래스는 오프라인 분석/검증에 활용.
    """

    def __init__(self, model, scaler, threshold_a: float, threshold_b: float,
                 infer_cfg: dict, smooth_window: int = 3):
        self.model = model
        self.scaler = scaler
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b
        self.cfg = infer_cfg

        seq_len = infer_cfg.get('seq_len', 50)
        self._buffer = deque(maxlen=seq_len)
        self.channel_b = FramewiseDeltaDetector(threshold_b, smooth_window)

        self.mrl_thresh        = float(infer_cfg.get('mrl_thresh', 0.3))
        self.cutoff_multiplier = float(infer_cfg.get('cutoff_multiplier', 2.2))

        self.score_a_log: list[float] = []
        self.delta_b_log: list[float] = []
        self._max_mrl_seen = 0.0

    def reset(self):
        self._buffer.clear()
        self.channel_b.reset()
        self.score_a_log.clear()
        self.delta_b_log.clear()
        self._max_mrl_seen = 0.0

    def update(self, frame: np.ndarray) -> dict:
        """
        frame: (D,) raw feature (scaler 적용 전)
        반환: {anomaly, channel, score_a, delta_b}
        """
        from train_gru_ae import apply_relative_transform, build_hybrid_window

        # ── 채널 B (매 프레임)
        delta_b, flag_b = self.channel_b.update(frame)
        self.delta_b_log.append(delta_b)

        # ── 채널 A (버퍼가 찬 경우)
        self._buffer.append(frame)
        score_a, flag_a = 0.0, False

        mrl_val = FramewiseDeltaDetector._compute_mrl(frame)
        self._max_mrl_seen = max(self._max_mrl_seen, mrl_val)

        if len(self._buffer) == self.cfg.get('seq_len', 50):
            # Rule 1: Miss Guard Rail (MRL threshold)
            if self._max_mrl_seen < self.mrl_thresh:
                flag_a = True
            else:
                window = np.array(self._buffer, dtype=np.float32)

                if self.cfg.get('hybrid'):
                    window = build_hybrid_window(window)

                x = self.scaler.transform(window).astype(np.float32)
                x = apply_relative_transform(x, self.cfg.get('relative', 'none'),
                                             self.cfg.get('n_rel_dims', None))
                x_t = torch.tensor(x).unsqueeze(0)

                with torch.no_grad():
                    score_a = float(self.model.reconstruction_error(
                        x_t, self.cfg.get('weight_alpha', 0.0)).item())
                
                # Rule 2 & 3: Super High Error Cutoff & Regular Threshold
                if score_a > self.threshold_a * self.cutoff_multiplier:
                    flag_a = True
                elif score_a > self.threshold_a:
                    flag_a = True

        self.score_a_log.append(score_a)

        # ── 최종 판단
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


# ──────────────────────────────────────────────
#  Threshold B 캘리브레이션
# ──────────────────────────────────────────────
def calibrate_threshold_b(success_sequences: list[np.ndarray],
                           smooth_window: int = 3,
                           percentile: int = 99,
                           warmup: int = 30) -> float:
    """
    정상 파지 시퀀스에서 프레임 간 MRL delta 분포를 계산하여
    상위 percentile을 threshold_B로 반환.

    success_sequences: list of (T, D) raw feature arrays
    """
    deltas = []
    for seq in success_sequences:
        mrl_series = np.array([FramewiseDeltaDetector._compute_mrl(seq[t]) for t in range(len(seq))])
        for t in range(max(smooth_window, warmup), len(mrl_series)):
            ref = float(mrl_series[t - smooth_window:t].mean())
            deltas.append(abs(float(mrl_series[t]) - ref))

    if not deltas:
        print("[채널B] delta 데이터 없음 → threshold_B=0.5 기본값 사용")
        return 0.5

    deltas = np.array(deltas)
    threshold_b = float(np.percentile(deltas, percentile))
    print(f"[채널B] delta 분포: mean={deltas.mean():.4f}, std={deltas.std():.4f}, max={deltas.max():.4f}")
    print(f"[채널B] threshold ({percentile}th percentile) = {threshold_b:.4f}")
    return threshold_b
