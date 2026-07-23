"""Shared transport feature extraction for training and online inference."""

from __future__ import annotations

import numpy as np


TRANSPORT_11D_CODEX_NAMES = (
    "mrl",
    "mean_delta_norm",
    "top3_delta_norm",
    "delta_fx",
    "delta_fy",
    "delta_fz",
    "delta_mx",
    "delta_my",
    "delta_mz",
    "active_delta_count",
    "std_delta_norm",
)

TRANSPORT_9D_RELATIVE_NAMES = (
    "delta_mrl",
    "delta_fx",
    "delta_fy",
    "delta_fz",
    "delta_mx",
    "delta_my",
    "delta_mz",
    "changed_taxel_count",
    "peak_delta_norm",
)


_TAXEL_POSITIONS = np.asarray(
    [[col - 1.5, 1.5 - row, 0.0]
     for row in range(4) for col in range(4)],
    dtype=np.float64,
)


def compute_transport_11d_delta(
    raw_cumulative: np.ndarray,
    active_threshold: float = 0.1,
    active_sharpness: float = 10.0,
) -> np.ndarray:
    """Convert cumulative 48D taxel displacement into revised transport 11D.

    Parameters
    ----------
    raw_cumulative:
        Array shaped ``(T, 48)`` in interleaved taxel order
        ``x0,y0,z0,...,x15,y15,z15``. In the current data pipeline this is
        effectively ``current_sensor - held_transport_baseline``.
    active_threshold:
        Delta-vector norm at which a taxel contributes 0.5 to the soft active
        count. It is expressed in the same range-normalised units as the input.
    active_sharpness:
        Dimensionless sigmoid slope relative to ``active_threshold``.

    Returns
    -------
    np.ndarray
        Float32 array shaped ``(T, 11)`` with columns listed in
        :data:`TRANSPORT_11D_CODEX_NAMES`.
    """
    raw = np.asarray(raw_cumulative, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 48:
        raise ValueError(f"raw_cumulative must have shape (T, 48), got {raw.shape}")
    if active_threshold <= 0:
        raise ValueError("active_threshold must be positive")
    if active_sharpness <= 0:
        raise ValueError("active_sharpness must be positive")

    vecs = raw.reshape(-1, 16, 3)
    norms = np.linalg.norm(vecs, axis=2)

    # Direction coherence of taxels whose displacement exceeds sensor noise.
    valid = norms > active_threshold
    unit_vecs = np.zeros_like(vecs)
    np.divide(vecs, norms[:, :, None], out=unit_vecs,
              where=valid[:, :, None])
    valid_count = valid.sum(axis=1)
    mean_unit = np.zeros((len(raw), 3), dtype=np.float64)
    usable = valid_count >= 2
    if np.any(usable):
        mean_unit[usable] = (
            unit_vecs[usable].sum(axis=1) /
            valid_count[usable, None]
        )
    mrl = np.linalg.norm(mean_unit, axis=1)

    mean_delta_norm = norms.mean(axis=1)
    top3_delta_norm = np.sort(norms, axis=1)[:, -3:].mean(axis=1)
    std_delta_norm = norms.std(axis=1)

    resultant = vecs.sum(axis=1)
    moments = np.cross(_TAXEL_POSITIONS[None, :, :], vecs).sum(axis=1)

    sigmoid_arg = np.clip(
        active_sharpness * (norms - active_threshold) / active_threshold,
        -60.0,
        60.0,
    )
    active_delta_count = (1.0 / (1.0 + np.exp(-sigmoid_arg))).sum(axis=1)

    features = np.column_stack((
        mrl,
        mean_delta_norm,
        top3_delta_norm,
        resultant,
        moments,
        active_delta_count,
        std_delta_norm,
    ))
    return features.astype(np.float32)


def compute_transport_9d_relative(
    raw_cumulative: np.ndarray,
    baseline_frames: int = 30,
    change_threshold: float = 0.1,
    change_sharpness: float = 10.0,
) -> np.ndarray:
    """Convert cumulative transport raw data into baseline-relative C9D.

    This is a relative analogue of the paper's ``C8D + MRL`` representation,
    not an absolute contact vector.  The median of the first stable-grasp
    frames is removed taxel by taxel:

    ``delta_vec(t) = raw_cumulative(t) - median(raw_cumulative[:B])``.

    The resulting features are MRL, resultant delta force (3), resultant
    delta moment (3), soft changed-taxel count, and peak 3-axis delta norm.
    ``changed_taxel_count`` and ``peak_delta_norm`` represent the spatial
    extent and local intensity of change; they must not be interpreted as
    physical contact area and absolute peak pressure.
    """
    raw = np.asarray(raw_cumulative, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 48:
        raise ValueError(
            f"raw_cumulative must have shape (T, 48), got {raw.shape}")
    if baseline_frames <= 0:
        raise ValueError("baseline_frames must be positive")
    if len(raw) < baseline_frames:
        raise ValueError(
            f"need at least {baseline_frames} frames for baseline, got {len(raw)}")
    if change_threshold <= 0:
        raise ValueError("change_threshold must be positive")
    if change_sharpness <= 0:
        raise ValueError("change_sharpness must be positive")

    baseline = np.median(raw[:baseline_frames], axis=0)
    return compute_transport_9d_from_baseline(
        raw,
        baseline,
        change_threshold=change_threshold,
        change_sharpness=change_sharpness,
    )


def compute_transport_9d_from_baseline(
    raw_cumulative: np.ndarray,
    baseline: np.ndarray,
    change_threshold: float = 0.1,
    change_sharpness: float = 10.0,
) -> np.ndarray:
    """Compute relative C9D using an already fixed 48D held baseline.

    This form is intended for online inference: collect the first stable-grasp
    frames, fix their median once, then use exactly the same baseline for all
    subsequent frames.
    """
    raw = np.asarray(raw_cumulative, dtype=np.float64)
    held_baseline = np.asarray(baseline, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 48:
        raise ValueError(
            f"raw_cumulative must have shape (T, 48), got {raw.shape}")
    if held_baseline.shape != (48,):
        raise ValueError(
            f"baseline must have shape (48,), got {held_baseline.shape}")
    if not np.isfinite(raw).all() or not np.isfinite(held_baseline).all():
        raise ValueError("raw_cumulative and baseline must be finite")
    if change_threshold <= 0:
        raise ValueError("change_threshold must be positive")
    if change_sharpness <= 0:
        raise ValueError("change_sharpness must be positive")

    delta_vecs = (raw - held_baseline).reshape(-1, 16, 3)
    delta_norms = np.linalg.norm(delta_vecs, axis=2)

    valid = delta_norms > change_threshold
    unit_vecs = np.zeros_like(delta_vecs)
    np.divide(
        delta_vecs,
        delta_norms[:, :, None],
        out=unit_vecs,
        where=valid[:, :, None],
    )
    valid_count = valid.sum(axis=1)
    mean_unit = np.zeros((len(raw), 3), dtype=np.float64)
    usable = valid_count >= 2
    if np.any(usable):
        mean_unit[usable] = (
            unit_vecs[usable].sum(axis=1) /
            valid_count[usable, None]
        )
    delta_mrl = np.linalg.norm(mean_unit, axis=1)

    delta_force = delta_vecs.sum(axis=1)
    delta_moment = np.cross(
        _TAXEL_POSITIONS[None, :, :], delta_vecs).sum(axis=1)

    sigmoid_arg = np.clip(
        change_sharpness *
        (delta_norms - change_threshold) / change_threshold,
        -60.0,
        60.0,
    )
    changed_taxel_count = (
        1.0 / (1.0 + np.exp(-sigmoid_arg))).sum(axis=1)
    peak_delta_norm = delta_norms.max(axis=1)

    features = np.column_stack((
        delta_mrl,
        delta_force,
        delta_moment,
        changed_taxel_count,
        peak_delta_norm,
    ))
    return features.astype(np.float32)
