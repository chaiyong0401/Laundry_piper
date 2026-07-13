"""
Preprocessing utilities for tactile feature windows.
"""

import numpy as np
import torch


def build_hybrid_window(window: np.ndarray) -> np.ndarray:
    """
    (T, D) window에 프레임별 L2 norm을 마지막 차원으로 추가.
    Returns: (T, D+1) float32

    norm 차원은 접촉 강도를 나타냄:
      - 접촉 있음: norm > 0
      - 접촉 없음: norm ≈ 0
    """
    per_frame_norm = np.linalg.norm(window, axis=1, keepdims=True)
    return np.concatenate([window, per_frame_norm], axis=1).astype(np.float32)


def apply_relative_transform(window: np.ndarray, mode: str,
                              n_rel_dims: int = None) -> np.ndarray:
    """
    window: (T, D) float32
    mode:
      'none'        → 원본 그대로
      'window_mean' → window 평균 기준 중심화 (자세 드리프트 제거)
      'first_frame' → 첫 프레임 기준 차분 (점진적 drift 제거)
    n_rel_dims: 변환 적용할 앞 차원 수 (None=전체, hybrid 모드에서는 D-1)
    Returns: (T, D) float32
    """
    if mode == 'none':
        return window.copy()
    d = window.shape[1] if n_rel_dims is None else n_rel_dims
    feat = window[:, :d].copy()
    rest = window[:, d:]
    if mode == 'window_mean':
        feat = feat - feat.mean(axis=0, keepdims=True)
    elif mode == 'first_frame':
        feat = feat - feat[0:1]
    if rest.shape[1] > 0:
        return np.concatenate([feat, rest], axis=1)
    return feat


def apply_relative_transform_tensor(x: torch.Tensor, mode: str,
                                     n_rel_dims: int = None) -> torch.Tensor:
    """
    x: (B, T, D) tensor
    """
    if mode == 'none':
        return x
    d = x.shape[2] if n_rel_dims is None else n_rel_dims
    feat = x[:, :, :d]
    rest = x[:, :, d:]
    if mode == 'window_mean':
        feat = feat - feat.mean(dim=1, keepdim=True)
    elif mode == 'first_frame':
        feat = feat - feat[:, 0:1, :]
    if rest.shape[2] > 0:
        return torch.cat([feat, rest], dim=2)
    return feat
