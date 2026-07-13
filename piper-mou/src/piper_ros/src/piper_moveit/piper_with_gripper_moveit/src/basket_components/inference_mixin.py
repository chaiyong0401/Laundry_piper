import os
import sys
import numpy as np


class InferenceMixin:
    """GRU-AE 기반 촉각 이상 감지 inference 메서드."""

    def _inference_worker(self):
        """별도 스레드에서 grasp/transport 이중 채널 inference를 처리."""
        while self._infer_thread_running:
            grasp_fired  = self._infer_trigger.wait(timeout=0.05)
            insert_fired = self._insert_trigger.is_set()

            if grasp_fired:
                self._infer_trigger.clear()
                score_a = self._run_windowed_inference()
                if score_a is not None:
                    with self._infer_lock:
                        flag_a = score_a > self.gru_ae_threshold
                        flag_b = self.grasp_channel_b_flag
                        self.grasp_channel_b_flag = False
                        is_anomaly = flag_a or flag_b
                        channel = ('AB' if flag_a and flag_b else
                                   'A'  if flag_a else
                                   'B'  if flag_b else 'none')
                        self.infer_latest_score = score_a
                        if is_anomaly:
                            self.infer_consec_anomaly += 1
                            self.infer_consec_normal = 0
                            if self.infer_consec_anomaly >= self.infer_consec_anomaly_needed:
                                self.infer_grasp_anomaly_detected = True
                        else:
                            self.infer_consec_normal += 1
                            self.infer_consec_anomaly = 0
                        self.get_logger().info(
                            f"[GRASP] score_A={score_a:.4f} "
                            f"thr_A={self.gru_ae_threshold:.4f} "
                            f"delta_B={self.grasp_delta_b_latest:.4f} "
                            f"thr_B={self.gru_ae_threshold_b:.4f} "
                            f"ch={channel} "
                            f"{'ANOMALY' if is_anomaly else 'normal'} "
                            f"consec_anom={self.infer_consec_anomaly} "
                            f"max_norm={self.max_norm:.3f} norm={self.norm:.3f}"
                        )

            if insert_fired:
                self._insert_trigger.clear()
                score_a = self._run_insert_windowed_inference()
                if score_a is not None:
                    with self._infer_lock:
                        flag_a = score_a > self.insert_gru_ae_threshold
                        flag_b = self.transport_channel_b_flag
                        is_anomaly = flag_a or flag_b
                        channel = ('AB' if flag_a and flag_b else
                                   'A'  if flag_a else
                                   'B'  if flag_b else 'none')
                        self.insert_infer_latest_score = score_a
                        if is_anomaly:
                            self.insert_infer_consec_anomaly += 1
                            self.insert_infer_consec_normal = 0
                            if (self.insert_infer_consec_anomaly >=
                                    self.insert_infer_consec_anomaly_needed):
                                if not self.insert_anomaly_ever_detected:
                                    self.insert_anomaly_ever_detected = True
                                    self._transport_abort.set()
                                    self.get_logger().warn(
                                        "🚨 Slip Recorded! Will recover after motion.")
                        else:
                            self.insert_infer_consec_normal += 1
                            self.insert_infer_consec_anomaly = 0
                        self.get_logger().info(
                            f"[TRANSPORT] score_A={score_a:.4f} "
                            f"thr_A={self.insert_gru_ae_threshold:.4f} "
                            f"delta_B={self.transport_delta_b_latest:.4f} "
                            f"thr_B={self.insert_gru_ae_threshold_b:.4f} "
                            f"ch={channel} "
                            f"{'ANOMALY' if is_anomaly else 'normal'} "
                            f"consec_anom={self.insert_infer_consec_anomaly} "
                            f"ever={self.insert_anomaly_ever_detected} "
                            f"max_norm={self.max_norm:.3f} norm={self.norm:.3f}"
                        )

    def _load_gru_ae(self):
        """학습된 GRU-AE 모델 + scaler + threshold A/B + infer_cfg 로드 (Model 1)."""
        import torch
        import joblib
        import pickle
        data_dir = self.data_save_dir

        use_raw = os.path.exists(os.path.join(data_dir, "grasp_gru_ae_raw.pt"))
        suffix = "_raw" if use_raw else ""

        model_path    = os.path.join(data_dir, f"grasp_gru_ae{suffix}.pt")
        scaler_path   = os.path.join(data_dir, f"grasp_scaler{suffix}.pkl")
        thresh_path   = os.path.join(data_dir, f"grasp_threshold{suffix}.npy")
        thresh_b_path = os.path.join(data_dir, f"grasp_threshold_b{suffix}.npy")
        cfg_path      = os.path.join(data_dir, f"grasp_infer_cfg{suffix}.pkl")

        for p in [model_path, scaler_path, thresh_path]:
            if not os.path.exists(p):
                self.get_logger().warn(f"[INFERENCE] 파일 없음: {p}")
                return
        try:
            sys.path.insert(0, "/home/dyros/mcy_ws")
            from train_gru_ae import GRUAE
            from dual_channel_detector import FramewiseDeltaDetector
            checkpoint = torch.load(model_path, map_location="cpu")
            scaler = joblib.load(scaler_path)
            input_dim = scaler.n_features_in_
            model = GRUAE(input_dim=input_dim, hidden_dim=64,
                          num_layers=2, dropout=0.2, seq_len=50)
            model.load_state_dict(checkpoint)
            model.eval()
            self.gru_ae_model     = model
            self.gru_ae_scaler    = scaler
            self.gru_ae_threshold = float(np.load(thresh_path)[0])
            if os.path.exists(thresh_b_path):
                self.gru_ae_threshold_b = float(np.load(thresh_b_path)[0])
            if os.path.exists(cfg_path):
                with open(cfg_path, "rb") as f:
                    self.gru_ae_infer_cfg = pickle.load(f)
            self.gru_ae_infer_cfg["use_raw"] = use_raw
            self.grasp_channel_b = FramewiseDeltaDetector(
                delta_threshold=self.gru_ae_threshold_b, smooth_window=3)
            self.get_logger().info(
                f"[INFERENCE] Grasp GRU-AE 로드 완료 "
                f"(input_dim={input_dim}, use_raw={use_raw}, "
                f"thr_A={self.gru_ae_threshold:.6f}, "
                f"thr_B={self.gru_ae_threshold_b:.6f})"
            )
        except Exception as e:
            self.get_logger().error(f"[INFERENCE] Grasp 모델 로드 실패: {e}")

    def _load_grasp_api_detector(self):
        """tactile_anomaly.DualChannelDetector 로드 (use_tactile_api=True 시 사용)."""
        try:
            sys.path.insert(0, "/home/dyros/mcy_ws")
            from tactile_anomaly.detector import DualChannelDetector
            data_dir  = self.data_save_dir
            use_raw   = os.path.exists(os.path.join(data_dir, "grasp_gru_ae_raw.pt"))
            suffix    = "_raw" if use_raw else ""
            model_path    = os.path.join(data_dir, f"grasp_gru_ae{suffix}.pt")
            scaler_path   = os.path.join(data_dir, f"grasp_scaler{suffix}.pkl")
            thresh_path   = os.path.join(data_dir, f"grasp_threshold{suffix}.npy")
            thresh_b_path = os.path.join(data_dir, f"grasp_threshold_b{suffix}.npy")
            cfg_path      = os.path.join(data_dir, f"grasp_infer_cfg{suffix}.pkl")
            for p in [model_path, scaler_path, thresh_path, thresh_b_path]:
                if not os.path.exists(p):
                    self.get_logger().warn(f"[API] 파일 없음, API detector 로드 불가: {p}")
                    return
            self.grasp_api_detector = DualChannelDetector.from_checkpoint(
                model_path       = model_path,
                scaler_path      = scaler_path,
                threshold_a_path = thresh_path,
                threshold_b_path = thresh_b_path,
                infer_cfg_path   = cfg_path if os.path.exists(cfg_path) else None,
            )
            self.get_logger().info(
                f"[API] DualChannelDetector 로드 완료 (use_raw={use_raw})"
            )
        except Exception as e:
            self.get_logger().error(f"[API] DualChannelDetector 로드 실패: {e}")

    def _run_windowed_inference(self) -> float | None:
        """realtime_buffer의 현재 50프레임 window로 즉시 anomaly score 계산."""
        if self.gru_ae_model is None or self.gru_ae_scaler is None:
            return None
        if len(self.realtime_buffer) < 50:
            return None
        try:
            import torch
            from train_gru_ae import apply_relative_transform, build_hybrid_window
            relative_mode = self.gru_ae_infer_cfg.get("relative", "none")
            weight_alpha  = self.gru_ae_infer_cfg.get("weight_alpha", 0.0)
            hybrid        = self.gru_ae_infer_cfg.get("hybrid", False)
            n_rel_dims    = self.gru_ae_infer_cfg.get("n_rel_dims", None)
            use_raw       = self.gru_ae_infer_cfg.get("use_raw", False)
            buf = self.realtime_raw_buffer if use_raw else self.realtime_buffer
            seq = np.array(list(buf), dtype=np.float32)
            if hybrid:
                seq = build_hybrid_window(seq)
            seq_s = self.gru_ae_scaler.transform(seq).astype(np.float32)
            seq_r = apply_relative_transform(seq_s, relative_mode,
                                             n_rel_dims=n_rel_dims)
            x = torch.tensor(seq_r).unsqueeze(0)
            with torch.no_grad():
                score = float(self.gru_ae_model.reconstruction_error(
                    x, weight_alpha=weight_alpha).item())
            return score
        except Exception as e:
            self.get_logger().error(f"[SLIDING] inference error: {e}")
            return None

    def _load_insert_gru_ae(self):
        """학습된 Transport GRU-AE 모델 + scaler + threshold A/B + infer_cfg 로드 (Model 2)."""
        import torch
        import joblib
        import pickle
        data_dir    = self.insert_save_dir
        use_raw     = os.path.exists(os.path.join(data_dir, "transport_gru_ae_raw.pt"))
        suffix      = "_raw" if use_raw else ""
        model_path    = os.path.join(data_dir, f"transport_gru_ae{suffix}.pt")
        scaler_path   = os.path.join(data_dir, f"transport_scaler{suffix}.pkl")
        thresh_path   = os.path.join(data_dir, f"transport_threshold{suffix}.npy")
        thresh_b_path = os.path.join(data_dir, f"transport_threshold_b{suffix}.npy")
        cfg_path      = os.path.join(data_dir, f"transport_infer_cfg{suffix}.pkl")

        for p in [model_path, scaler_path, thresh_path]:
            if not os.path.exists(p):
                self.get_logger().warn(f"[INFERENCE] Transport 파일 없음: {p}")
                return
        try:
            sys.path.insert(0, "/home/dyros/mcy_ws")
            from train_gru_ae import GRUAE
            from dual_channel_detector import FramewiseDeltaDetector
            checkpoint = torch.load(model_path, map_location="cpu")
            scaler = joblib.load(scaler_path)
            input_dim = scaler.n_features_in_
            model = GRUAE(input_dim=input_dim, hidden_dim=64,
                          num_layers=2, dropout=0.2, seq_len=50)
            model.load_state_dict(checkpoint)
            model.eval()
            self.insert_gru_ae_model     = model
            self.insert_gru_ae_scaler    = scaler
            self.insert_gru_ae_threshold = float(np.load(thresh_path)[0])
            if os.path.exists(thresh_b_path):
                self.insert_gru_ae_threshold_b = float(np.load(thresh_b_path)[0])
            if os.path.exists(cfg_path):
                with open(cfg_path, "rb") as f:
                    self.insert_gru_ae_infer_cfg = pickle.load(f)
            self.transport_channel_b = FramewiseDeltaDetector(
                delta_threshold=self.insert_gru_ae_threshold_b, smooth_window=3)
            self.insert_gru_ae_infer_cfg["use_raw"] = use_raw
            self.get_logger().info(
                f"[INFERENCE] Transport GRU-AE 로드 완료 "
                f"(input_dim={input_dim}, use_raw={use_raw}, "
                f"thr_A={self.insert_gru_ae_threshold:.6f}, "
                f"thr_B={self.insert_gru_ae_threshold_b:.6f})"
            )
        except Exception as e:
            self.get_logger().error(f"[INFERENCE] Transport 모델 로드 실패: {e}")

    def _run_insert_windowed_inference(self) -> float | None:
        """insert_realtime_buffer의 현재 50프레임 window로 anomaly score 계산."""
        if self.insert_gru_ae_model is None or self.insert_gru_ae_scaler is None:
            return None
        try:
            import torch
            from train_gru_ae import apply_relative_transform, build_hybrid_window
            relative_mode = self.insert_gru_ae_infer_cfg.get("relative", "none")
            weight_alpha  = self.insert_gru_ae_infer_cfg.get("weight_alpha", 0.0)
            hybrid        = self.insert_gru_ae_infer_cfg.get("hybrid", False)
            n_rel_dims    = self.insert_gru_ae_infer_cfg.get("n_rel_dims", None)
            use_raw       = self.insert_gru_ae_infer_cfg.get("use_raw", False)
            buf = self.insert_realtime_raw_buffer if use_raw else self.insert_realtime_buffer
            if len(buf) < 50:
                return None
            seq = np.array(list(buf), dtype=np.float32)
            if hybrid:
                seq = build_hybrid_window(seq)
            seq_s = self.insert_gru_ae_scaler.transform(seq).astype(np.float32)
            seq_r = apply_relative_transform(seq_s, relative_mode,
                                             n_rel_dims=n_rel_dims)
            x = torch.tensor(seq_r).unsqueeze(0)
            with torch.no_grad():
                score = float(self.insert_gru_ae_model.reconstruction_error(
                    x, weight_alpha=weight_alpha).item())
            return score
        except Exception as e:
            self.get_logger().error(f"[SLIDING-INSERT] inference error: {e}")
            return None
