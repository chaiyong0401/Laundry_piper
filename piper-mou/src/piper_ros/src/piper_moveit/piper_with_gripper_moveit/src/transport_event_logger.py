"""Thread-safe logging utilities for transport anomaly event evaluation.

The logger intentionally keeps detector output separate from ground-truth event
markers.  A trial is saved as one compressed NPZ file plus a small JSON metadata
file so raw tactile vectors do not make the metadata difficult to inspect.
"""

from __future__ import annotations

import datetime
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class TransportEventLogger:
    """Collect sensor frames, recurrent updates, and independent event markers."""

    def __init__(self, enabled: bool, output_dir: str) -> None:
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir).expanduser()
        self._lock = threading.Lock()
        self._trial: Optional[Dict[str, Any]] = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self.enabled and self._trial is not None

    def start_trial(
        self,
        *,
        trial_type: str,
        config: Dict[str, Any],
        sensor_zero_raw: Optional[np.ndarray],
    ) -> Optional[str]:
        if not self.enabled:
            return None
        now_wall = datetime.datetime.now(datetime.timezone.utc)
        trial_id = now_wall.astimezone().strftime("%Y%m%d_%H%M%S_%f")
        with self._lock:
            if self._trial is not None:
                self._finish_locked("superseded_by_new_trial")
            self._trial = {
                "trial_id": trial_id,
                "trial_type": str(trial_type),
                "wall_start_iso": now_wall.isoformat(),
                "start_monotonic_ns": time.monotonic_ns(),
                "end_monotonic_ns": None,
                "status": "recording",
                "config": dict(config),
                "sensor_zero_raw": (
                    None if sensor_zero_raw is None
                    else np.asarray(sensor_zero_raw, dtype=np.float64).tolist()
                ),
                "events": [],
                "triggers": [],
                "sensor_rows": [],
                "a_rows": [],
            }
        return trial_id

    def set_trial_type(self, trial_type: str) -> bool:
        with self._lock:
            if not self.enabled or self._trial is None:
                return False
            self._trial["trial_type"] = str(trial_type)
            return True

    def mark_event(self, name: str, note: str = "") -> bool:
        timestamp_ns = time.monotonic_ns()
        with self._lock:
            if not self.enabled or self._trial is None:
                return False
            self._trial["events"].append({
                "name": str(name),
                "timestamp_ns": timestamp_ns,
                "elapsed_s": (
                    timestamp_ns - self._trial["start_monotonic_ns"]
                ) / 1e9,
                "note": str(note),
            })
            return True

    def has_event(self, name: str, include_local_fallback: bool = True) -> bool:
        """Return whether the active trial contains a named event.

        The marker utility always writes its local stream before publishing ROS.
        Looking at that stream as a fallback prevents a successfully pressed
        ``l`` key from being classified as ``no_loss`` during a brief ROS
        discovery failure.
        """
        with self._lock:
            if not self.enabled or self._trial is None:
                return False
            if any(
                    event.get("name") == name
                    for event in self._trial.get("events", [])):
                return True
            if not include_local_fallback:
                return False
            start_ns = int(self._trial["start_monotonic_ns"])
            end_ns = time.monotonic_ns()

        marker_path = self.output_dir / "marker_stream.jsonl"
        if not marker_path.exists():
            return False
        try:
            with marker_path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    marker = json.loads(line)
                    timestamp_ns = int(marker["timestamp_ns"])
                    if (start_ns <= timestamp_ns <= end_ns
                            and str(marker["name"]) == name):
                        return True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False
        return False

    def log_trigger(self, channel: str) -> None:
        timestamp_ns = time.monotonic_ns()
        with self._lock:
            if not self.enabled or self._trial is None:
                return
            self._trial["triggers"].append({
                "channel": str(channel),
                "timestamp_ns": timestamp_ns,
                "elapsed_s": (
                    timestamp_ns - self._trial["start_monotonic_ns"]
                ) / 1e9,
            })

    def log_sensor_frame(
        self,
        *,
        timestamp_ns: int,
        frame_index: int,
        raw_48d: np.ndarray,
        model_raw_48d: np.ndarray,
        feature: np.ndarray,
        contact: float,
        contact_baseline: float,
        contact_drop: float,
        force: float,
        force_baseline: float,
        force_drop: float,
        b_instant: bool,
        b_confirmed: bool,
        fusion_confirmed: bool,
    ) -> None:
        with self._lock:
            if not self.enabled or self._trial is None:
                return
            self._trial["sensor_rows"].append({
                "timestamp_ns": int(timestamp_ns),
                "frame_index": int(frame_index),
                "raw_48d": np.asarray(raw_48d, dtype=np.float32).reshape(-1),
                "model_raw_48d": np.asarray(
                    model_raw_48d, dtype=np.float32).reshape(-1),
                "feature": np.asarray(feature, dtype=np.float32).reshape(-1),
                "contact": float(contact),
                "contact_baseline": float(contact_baseline),
                "contact_drop": float(contact_drop),
                "force": float(force),
                "force_baseline": float(force_baseline),
                "force_drop": float(force_drop),
                "b_instant": bool(b_instant),
                "b_confirmed": bool(b_confirmed),
                "fusion_confirmed": bool(fusion_confirmed),
            })

    def log_a_update(
        self,
        *,
        timestamp_ns: int,
        frame_index: int,
        score: float,
        threshold: float,
        rule1: bool,
        rule2: bool,
        rule3: bool,
        a_instant: bool,
        a_confirmed: bool,
    ) -> None:
        with self._lock:
            if not self.enabled or self._trial is None:
                return
            self._trial["a_rows"].append({
                "timestamp_ns": int(timestamp_ns),
                "frame_index": int(frame_index),
                "score": float(score),
                "threshold": float(threshold),
                "rule1": bool(rule1),
                "rule2": bool(rule2),
                "rule3": bool(rule3),
                "a_instant": bool(a_instant),
                "a_confirmed": bool(a_confirmed),
            })

    def finish_trial(self, status: str) -> Optional[Path]:
        with self._lock:
            if not self.enabled or self._trial is None:
                return None
            return self._finish_locked(status)

    def _finish_locked(self, status: str) -> Path:
        trial = self._trial
        assert trial is not None
        trial["end_monotonic_ns"] = time.monotonic_ns()
        trial["status"] = str(status)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._merge_external_markers(trial)

        stem = f"transport_event_{trial['trial_id']}"
        npz_path = self.output_dir / f"{stem}.npz"
        json_path = self.output_dir / f"{stem}.json"

        sensor_rows = trial.pop("sensor_rows")
        a_rows = trial.pop("a_rows")
        sensor_arrays = self._sensor_arrays(sensor_rows)
        a_arrays = self._a_arrays(a_rows)
        np.savez_compressed(npz_path, **sensor_arrays, **a_arrays)

        trial["npz_file"] = npz_path.name
        trial["sensor_frame_count"] = len(sensor_rows)
        trial["a_update_count"] = len(a_rows)
        with json_path.open("w", encoding="utf-8") as stream:
            json.dump(trial, stream, ensure_ascii=False, indent=2)

        self._trial = None
        return json_path

    def _merge_external_markers(self, trial: Dict[str, Any]) -> None:
        """Merge keyboard timestamps if ROS marker delivery was unavailable."""
        marker_path = self.output_dir / "marker_stream.jsonl"
        if not marker_path.exists():
            return
        start_ns = int(trial["start_monotonic_ns"])
        end_ns = int(trial["end_monotonic_ns"])
        external = []
        try:
            with marker_path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    marker = json.loads(line)
                    timestamp_ns = int(marker["timestamp_ns"])
                    if start_ns <= timestamp_ns <= end_ns:
                        external.append({
                            "name": str(marker["name"]),
                            "timestamp_ns": timestamp_ns,
                            "elapsed_s": (timestamp_ns - start_ns) / 1e9,
                            "note": "keyboard marker (local fallback)",
                        })
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return

        existing_names = {event.get("name") for event in trial["events"]}
        for marker in external:
            # The ROS callback timestamp and keypress timestamp differ slightly.
            # Prefer the existing ROS marker when both paths succeeded.
            if marker["name"] not in existing_names:
                trial["events"].append(marker)
                existing_names.add(marker["name"])
        trial["events"].sort(key=lambda event: int(event["timestamp_ns"]))

    @staticmethod
    def _sensor_arrays(rows: list[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        count = len(rows)
        raw_dim = len(rows[0]["raw_48d"]) if rows else 48
        feature_dim = len(rows[0]["feature"]) if rows else 0

        def values(name: str, dtype: Any) -> np.ndarray:
            return np.asarray([row[name] for row in rows], dtype=dtype)

        return {
            "sensor_timestamp_ns": values("timestamp_ns", np.int64),
            "sensor_frame_index": values("frame_index", np.int64),
            "raw_48d": (
                np.stack([row["raw_48d"] for row in rows])
                if rows else np.empty((0, raw_dim), dtype=np.float32)
            ),
            "model_raw_48d": (
                np.stack([row["model_raw_48d"] for row in rows])
                if rows else np.empty((0, raw_dim), dtype=np.float32)
            ),
            "feature": (
                np.stack([row["feature"] for row in rows])
                if rows else np.empty((0, feature_dim), dtype=np.float32)
            ),
            "contact": values("contact", np.float32),
            "contact_baseline": values("contact_baseline", np.float32),
            "contact_drop": values("contact_drop", np.float32),
            "force": values("force", np.float32),
            "force_baseline": values("force_baseline", np.float32),
            "force_drop": values("force_drop", np.float32),
            "b_instant": values("b_instant", np.bool_),
            "b_confirmed": values("b_confirmed", np.bool_),
            "fusion_confirmed": values("fusion_confirmed", np.bool_),
            "sensor_row_count": np.asarray([count], dtype=np.int64),
        }

    @staticmethod
    def _a_arrays(rows: list[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        def values(name: str, dtype: Any) -> np.ndarray:
            return np.asarray([row[name] for row in rows], dtype=dtype)

        return {
            "a_timestamp_ns": values("timestamp_ns", np.int64),
            "a_frame_index": values("frame_index", np.int64),
            "a_score": values("score", np.float32),
            "a_threshold": values("threshold", np.float32),
            "a_rule1": values("rule1", np.bool_),
            "a_rule2": values("rule2", np.bool_),
            "a_rule3": values("rule3", np.bool_),
            "a_instant": values("a_instant", np.bool_),
            "a_confirmed": values("a_confirmed", np.bool_),
            "a_row_count": np.asarray([len(rows)], dtype=np.int64),
        }
