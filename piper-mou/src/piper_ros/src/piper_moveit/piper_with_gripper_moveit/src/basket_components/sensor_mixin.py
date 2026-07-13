import math
import time
import queue
import numpy as np
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState
from basket_components.constants import URDF_JOINT_NAMES, DISPLAY_IDX, A, B, FOUR_SIGMA


class SensorMixin:
    """XELA 촉각 센서 처리 및 토크 모니터링."""

    def state_cb(self, msg: JointTrajectoryControllerState):
        if msg.joint_names:
            jmap = {n: i for i, n in enumerate(msg.joint_names)}
        else:
            jmap = {n: i for i, n in enumerate(URDF_JOINT_NAMES)}

        desired = msg.desired
        q  = np.zeros(self.nj)
        qd = np.zeros(self.nj)
        qdd= np.zeros(self.nj)

        for j, n in enumerate(URDF_JOINT_NAMES):
            if n in jmap:
                k = jmap[n]
                if len(desired.positions) > k:     q[j]  = desired.positions[k]
                if len(desired.velocities) > k:    qd[j] = desired.velocities[k]
                if len(desired.accelerations) > k:
                    a = desired.accelerations[k]
                    qdd[j] = a if math.isfinite(a) else 0.0

        self.qd, self.qdotd, self.qddotd = q, qd, qdd
        self.have_desired = True

    def joint_cb(self, msg: JointState):
        self.last_js_msg = msg
        idx_map = {n: i for i, n in enumerate(msg.name)}

        for j, n in enumerate(URDF_JOINT_NAMES):
            if n in idx_map:
                k = idx_map[n]
                if len(msg.effort) > k:
                    val = msg.effort[k]
                    if math.isfinite(val):
                        self.effort[j] = val
                        self.effort_last_valid[j] = val
                    else:
                        self.effort[j] = self.effort_last_valid[j]

    def tick(self):
        import pinocchio as pin
        if not self.monitor_torque:
            return
        if not self.have_desired or self.paused:
            return
        if self.stop_requested:
            return

        try:
            tau_pred = pin.rnea(self.model, self.data, self.qd, self.qdotd, self.qddotd)
            tau_pred = np.abs(tau_pred)
        except Exception as e:
            self.get_logger().warn(f"rnea failed: {e}")
            return

        tau_real = self.effort.copy()
        tau_pre = A @ tau_pred[0:6] + B

        for j in DISPLAY_IDX:
            pred_val = float(tau_pre[j - 2])
            real_val = float(tau_real[j])
            err = abs(pred_val - real_val)

            if err > FOUR_SIGMA:
                reason = f"4σ torque violation ({URDF_JOINT_NAMES[j]} err={err:.4f} > {FOUR_SIGMA:.4f})"
                self.get_logger().warn(f"[4σ 위반] {reason}")
                self._request_stop(reason)
                self.torque_stop = True
                self.recover_pending = True
                break

    def reset_sensor_data(self):
        self.prev_x, self.prev_y, self.prev_z = None, None, None
        self.cum_x, self.cum_y, self.cum_z = [0.0] * 16, [0.0] * 16, [0.0] * 16
        self.z_aligned, self.contacted = False, False
        self.get_logger().info("센서 기준값 및 상태 초기화 완료.")

    def joint_state_callback(self, msg):
        current_map = {n: p for n, p in zip(msg.name, msg.position)}
        pos = []
        for i in range(1, 7):
            jn = f"joint{i}"
            pos.append(current_map.get(jn, 0.0))
        pos.append(current_map.get("joint7", 0.0))
        pos.append(current_map.get("joint8", 0.0))
        self.current_joint_state = pos

    def sensor_callback(self, msg):
        """DDS 블로킹 방지: raw 데이터만 큐에 넣고 즉시 리턴."""
        try:
            x_vals = [t.x for t in msg.sensors[0].taxels]
            y_vals = [t.y for t in msg.sensors[0].taxels]
            z_vals = [t.z for t in msg.sensors[0].taxels]
            try:
                self._raw_sensor_queue.put_nowait((x_vals, y_vals, z_vals))
            except queue.Full:
                pass
        except Exception:
            pass

    def _sensor_processing_worker(self):
        """raw 센서 데이터를 큐에서 꺼내 feature 계산 + 상태 업데이트."""
        _prev_raw_x = None
        _prev_raw_y = None
        _prev_raw_z = None
        _last_data_time = time.time()
        _STALE_TIMEOUT = 0.5

        while True:
            try:
                x_vals, y_vals, z_vals = self._raw_sensor_queue.get(timeout=0.1)
            except queue.Empty:
                if (self.prev_x is not None and
                        time.time() - _last_data_time > _STALE_TIMEOUT):
                    self.get_logger().warn(
                        "[SENSOR] WebSocket 끊김 감지 → 누적값 리셋"
                    )
                    self.cum_x = [0.0] * 16
                    self.cum_y = [0.0] * 16
                    self.cum_z = [0.0] * 16
                    self.prev_x = None
                    _prev_raw_x = _prev_raw_y = _prev_raw_z = None
                    _last_data_time = time.time()
                continue

            _last_data_time = time.time()

            if x_vals == _prev_raw_x and y_vals == _prev_raw_y and z_vals == _prev_raw_z:
                continue
            _prev_raw_x, _prev_raw_y, _prev_raw_z = x_vals, y_vals, z_vals

            try:
                x_vals, y_vals, z_vals = self._normalize_taxels(x_vals, y_vals, z_vals)

                if self.prev_x is None:
                    self.prev_x, self.prev_y, self.prev_z = x_vals, y_vals, z_vals
                    self.z_aligned = self.contacted = False
                    self.z_aligned_k = self.contacted_k = False
                    self.z_aligned_f = self.contacted_f = False
                    self.contacted_ff = self.contacted_kk = False
                    continue

                dx = [c - p for c, p in zip(x_vals, self.prev_x)]
                dy = [c - p for c, p in zip(y_vals, self.prev_y)]
                dz = [c - p for c, p in zip(z_vals, self.prev_z)]

                _max_jump = max(
                    max(abs(d) for d in dx),
                    max(abs(d) for d in dy),
                    max(abs(d) for d in dz),
                )
                if _max_jump > 0.5:
                    self.cum_x = [0.0] * 16
                    self.cum_y = [0.0] * 16
                    self.cum_z = [0.0] * 16
                    self.prev_x, self.prev_y, self.prev_z = x_vals, y_vals, z_vals
                    _prev_raw_x = _prev_raw_y = _prev_raw_z = None
                    self.get_logger().warn(
                        f"[SENSOR] 점프 감지 (max_diff={_max_jump:.3f}) → 누적값 리셋"
                    )
                    continue

                for i in range(16):
                    self.cum_x[i] += dx[i]
                    self.cum_y[i] += dy[i]
                    self.cum_z[i] += dz[i]

                vecs = np.array([self.cum_x, self.cum_y, self.cum_z]).T
                norms = np.linalg.norm(vecs, axis=1)
                max_norm = np.max(norms)
                if max_norm == 0:
                    max_norm = 1e-6
                weights = norms.copy()
                weights[weights == 0] = 1e-6

                mean_vec_biased = np.sum(vecs, axis=0) / np.sum(weights)

                norm = np.linalg.norm(mean_vec_biased)
                cos_theta = np.dot(mean_vec_biased, np.array([0, 0, 1])) / (norm + 1e-8)
                theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

                self.norm = norm
                self.theta = theta_deg
                self.max_norm = max_norm

                valid_mask = norms > 0.1
                if valid_mask.sum() >= 2:
                    unit_vecs = vecs[valid_mask] / norms[valid_mask, np.newaxis]
                    mean_unit = np.mean(unit_vecs, axis=0)
                    mrl = float(np.linalg.norm(mean_unit))
                else:
                    mean_unit = np.zeros(3)
                    mrl = 0.0
                self.mrl = mrl

                frame = [
                    mrl,
                    float(norm),
                    float(max_norm),
                    float(mean_unit[0]),
                    float(mean_unit[1]),
                    float(mean_unit[2]),
                ]
                raw_frame = vecs.flatten().tolist()

                if self.is_recording:
                    self.grasp_buffer.append(frame)
                    self.grasp_raw_buffer.append(raw_frame)
                    if self.run_mode == 'inference':
                        if self.use_tactile_api and self.grasp_api_detector is not None:
                            frame_arr = np.array(frame,     dtype=np.float32)
                            raw_arr   = np.array(raw_frame, dtype=np.float32)
                            self.inference_frame_cnt += 1
                            res      = self.grasp_api_detector.update(frame_arr, raw_frame=raw_arr)
                            is_anom  = res['anomaly']
                            score_a  = res['score_a']
                            delta_b  = res['delta_b']
                            channel  = res['channel']
                            with self._infer_lock:
                                self.infer_latest_score   = score_a
                                self.grasp_delta_b_latest = delta_b
                                if is_anom:
                                    self.infer_consec_anomaly += 1
                                    self.infer_consec_normal   = 0
                                    if self.infer_consec_anomaly >= self.infer_consec_anomaly_needed:
                                        self.infer_grasp_anomaly_detected = True
                                else:
                                    self.infer_consec_normal   += 1
                                    self.infer_consec_anomaly   = 0
                            if score_a > 0:
                                self.get_logger().info(
                                    f"[API-GRASP] score_A={score_a:.4f} "
                                    f"delta_B={delta_b:.4f} ch={channel} "
                                    f"{'ANOMALY' if is_anom else 'normal'} "
                                    f"consec={self.infer_consec_anomaly}"
                                )
                        else:
                            self.realtime_buffer.append(frame)
                            self.realtime_raw_buffer.append(raw_frame)
                            self.inference_frame_cnt += 1
                            if (len(self.realtime_buffer) >= 50 and
                                    self.inference_frame_cnt % self.inference_stride == 0):
                                self._infer_trigger.set()
                            if (self.grasp_channel_b is not None and
                                    self.inference_frame_cnt > self.grasp_channel_b_warmup_needed and
                                    self.max_norm > 0.3):
                                frame_arr = np.array(frame, dtype=np.float32)
                                delta_b, flag_b = self.grasp_channel_b.update(frame_arr)
                                if flag_b:
                                    self.grasp_channel_b_consec += 1
                                    if (self.grasp_channel_b_consec >=
                                            self.grasp_channel_b_consec_needed):
                                        with self._infer_lock:
                                            self.grasp_channel_b_flag = True
                                            self.grasp_delta_b_latest = delta_b
                                            self.infer_consec_normal = 0
                                        self.get_logger().error(
                                            f"[CH-B] Grasp 이상 즉시 감지! "
                                            f"delta={delta_b:.4f} thr={self.gru_ae_threshold_b:.4f}"
                                        )
                                else:
                                    self.grasp_channel_b_consec = 0

                if self.is_insert_recording:
                    self.insert_buffer.append(frame)
                    self.insert_raw_buffer.append(raw_frame)
                    if self.run_mode == 'inference':
                        self.insert_realtime_buffer.append(frame)
                        self.insert_realtime_raw_buffer.append(raw_frame)
                        self.insert_infer_frame_cnt += 1
                        if (len(self.insert_realtime_buffer) >= 50 and
                                self.insert_infer_frame_cnt % self.inference_stride == 0):
                            self._insert_trigger.set()
                        if (self.transport_channel_b is not None and
                                self.insert_infer_frame_cnt > self.transport_channel_b_warmup_needed and
                                self.max_norm > 0.3):
                            frame_arr = np.array(frame, dtype=np.float32)
                            delta_b, flag_b = self.transport_channel_b.update(frame_arr)
                            if flag_b:
                                self.transport_channel_b_consec += 1
                                if (self.transport_channel_b_consec >=
                                        self.transport_channel_b_consec_needed):
                                    with self._infer_lock:
                                        self.transport_channel_b_flag = True
                                        self.transport_delta_b_latest = delta_b
                                        if not self.insert_anomaly_ever_detected:
                                            self.insert_anomaly_ever_detected = True
                                            self._transport_abort.set()
                                            self.get_logger().warn(
                                                "🚨 Slip Recorded! Will recover after motion.")
                                    self.get_logger().error(
                                        f"[CH-B] Transport 낙하 즉시 감지! "
                                        f"delta={delta_b:.4f} thr={self.insert_gru_ae_threshold_b:.4f}"
                                    )
                            else:
                                self.transport_channel_b_consec = 0

                if getattr(self, "monitor_slip", False):
                    if self.monitor_initial_norm > 0 and norm < (self.monitor_initial_norm * 0.5):
                        self.get_logger().error(
                            f"🚨 [MONITOR] Slip Detected! Norm Dropped: "
                            f"{self.monitor_initial_norm:.3f} -> {norm:.3f}")
                        self._trigger_slip_stop()
                    elif abs(theta_deg - self.monitor_initial_theta) > 20.0:
                        self.get_logger().error(
                            f"🚨 [MONITOR] Slip Detected! Theta Shift: "
                            f"{self.monitor_initial_theta:.1f} -> {theta_deg:.1f}")
                        self._trigger_slip_stop()

                if max_norm < 0.5:
                    self.z_aligned = False
                    self.contacted = False
                elif norm < 0.35:
                    self.z_aligned = False
                    self.contacted = False
                else:
                    if norm < 1.0:
                        if theta_deg < 55.0:
                            self.z_aligned = True
                            self.contacted = False
                        elif theta_deg <= 80.0:
                            self.contacted = True
                            self.z_aligned = False
                        else:
                            self.z_aligned = False
                            self.contacted = False
                    else:
                        if theta_deg < 35.0:
                            self.z_aligned = True
                            self.contacted = False
                        elif theta_deg <= 80.0:
                            self.contacted = True
                            self.z_aligned = False
                        else:
                            self.z_aligned = False
                            self.contacted = False

                self.prev_x, self.prev_y, self.prev_z = x_vals, y_vals, z_vals

            except Exception as e:
                self.get_logger().error(f"[SENSOR] 처리 오류: {e}")

    def _load_minmax(self, path: str):
        """taxel_minmax_norm.npy -> (gmin(48,), grng(48,)) as float64"""
        try:
            mm = np.load(path, allow_pickle=True).item()
            gmin = np.asarray(mm["min"], dtype=np.float64)
            grng = np.asarray(mm["range"], dtype=np.float64)
            if gmin.shape[0] != 48 or grng.shape[0] != 48:
                raise ValueError(f"min/range shape must be 48, got {gmin.shape}, {grng.shape}")
            grng[~np.isfinite(grng) | (grng <= 0.0)] = 1.0
            return gmin, grng
        except Exception as e:
            self.get_logger().warn(f"MinMax load failed ({e}); using identity (no-op)")
            gmin = np.zeros(48, dtype=np.float64)
            grng = np.ones(48, dtype=np.float64)
            return gmin, grng

    def _normalize_taxels(self, x_vals, y_vals, z_vals):
        """raw 16×(x,y,z) -> 48축 Min–Max 정규화 결과를 분해해서 (x,y,z) 반환"""
        A_arr = np.empty(48, dtype=np.float64)
        A_arr[0:16] = np.asarray(x_vals, dtype=np.float64)
        A_arr[16:32] = np.asarray(y_vals, dtype=np.float64)
        A_arr[32:48] = np.asarray(z_vals, dtype=np.float64)
        An = (A_arr - self.gmin) / self.grng
        return An[0:16].copy(), An[16:32].copy(), An[32:48].copy()
