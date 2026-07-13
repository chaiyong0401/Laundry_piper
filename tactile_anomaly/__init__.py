"""
tactile_anomaly
===============
XELA 촉각센서 기반 실시간 이상 감지 패키지.

Quick start
-----------
    from tactile_anomaly import DualChannelDetector

    detector = DualChannelDetector.from_checkpoint(
        model_path       = 'grasp_gru_ae.pt',
        scaler_path      = 'grasp_scaler.pkl',
        threshold_a_path = 'grasp_threshold.npy',
        threshold_b_path = 'grasp_threshold_b.npy',
        infer_cfg_path   = 'grasp_infer_cfg.pkl',   # optional
    )

    detector.reset()   # 에피소드 시작 시

    for frame in tactile_stream:          # frame: (6,) numpy array
        result = detector.update(frame)
        if result['anomaly']:
            print(f"이상 감지! 채널={result['channel']}, "
                  f"score_A={result['score_a']:.4f}, "
                  f"delta_B={result['delta_b']:.4f}")
"""

from .detector import DualChannelDetector, FramewiseDeltaDetector, calibrate_threshold_b
from .model import GRUAE
from .preprocess import apply_relative_transform, build_hybrid_window
from .xela_bridge import XelaBridge

__all__ = [
    'DualChannelDetector',
    'FramewiseDeltaDetector',
    'calibrate_threshold_b',
    'GRUAE',
    'apply_relative_transform',
    'build_hybrid_window',
    'XelaBridge',
]
