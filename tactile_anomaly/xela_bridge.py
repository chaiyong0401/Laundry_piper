#!/usr/bin/env python3
"""
XELA Bridge
============
xela_server WebSocket → 6D feature 계산 → GRU-AE 추론 → TCP 소켓 서버

C 클라이언트 프로토콜:
  C → Bridge : 1 byte 커맨드
                0x01 = RESET  에피소드 시작 (누적 버퍼 + 모델 버퍼 초기화)
                0x02 = GET    최신 추론 결과 요청

  Bridge → C : 12 bytes (GET 응답)
                [0]    uint8   anomaly  (0 or 1)
                [1]    uint8   channel  (0=none, 1=A, 2=B, 3=AB)
                [2-5]  float32 score_a
                [6-9]  float32 delta_b
                [10-11] uint8  reserved

Usage:
  python3 -m tactile_anomaly.xela_bridge \\
    --model     grasp_gru_ae.pt \\
    --scaler    grasp_scaler.pkl \\
    --thr_a     grasp_threshold.npy \\
    --thr_b     grasp_threshold_b.npy \\
    --cfg       grasp_infer_cfg.pkl \\
    --ws_host   localhost \\
    --ws_port   5000 \\
    --sensor_id 1 \\
    --tcp_port  9001
"""

import argparse
import json
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np
import websocket

from .detector import DualChannelDetector

API_FOLDER = Path(__file__).parent / 'api_folder'

_PRESETS = {
    'grasp_05': {
        'model':  API_FOLDER / 'grasp_05' / 'grasp_gru_ae.pt',
        'scaler': API_FOLDER / 'grasp_05' / 'grasp_scaler.pkl',
        'thr_a':  API_FOLDER / 'grasp_05' / 'grasp_threshold.npy',
        'thr_b':  API_FOLDER / 'grasp_05' / 'grasp_threshold_b.npy',
        'cfg':    API_FOLDER / 'grasp_05' / 'grasp_infer_cfg.pkl',
    },
    'grasp_05_raw': {
        'model':  API_FOLDER / 'grasp_05' / 'grasp_gru_ae_raw.pt',
        'scaler': API_FOLDER / 'grasp_05' / 'grasp_scaler_raw.pkl',
        'thr_a':  API_FOLDER / 'grasp_05' / 'grasp_threshold_raw.npy',
        'thr_b':  API_FOLDER / 'grasp_05' / 'grasp_threshold_b_raw.npy',
        'cfg':    API_FOLDER / 'grasp_05' / 'grasp_infer_cfg_raw.pkl',
        # preset 기본값 (CLI 인자로 덮어쓰기 가능)
        'defaults': {'thr_a_val': 0.4, 'anomaly_consec': 2, 'max_norm_gate': 0.3},
    },
    'transport_05': {
        'model':  API_FOLDER / 'transport_05' / 'transport_gru_ae.pt',
        'scaler': API_FOLDER / 'transport_05' / 'transport_scaler.pkl',
        'thr_a':  API_FOLDER / 'transport_05' / 'transport_threshold.npy',
        'thr_b':  API_FOLDER / 'transport_05' / 'transport_threshold_b.npy',
        'cfg':    API_FOLDER / 'transport_05' / 'transport_infer_cfg.pkl',
    },
    'transport_05_raw': {
        'model':  API_FOLDER / 'transport_05' / 'transport_gru_ae_raw.pt',
        'scaler': API_FOLDER / 'transport_05' / 'transport_scaler_raw.pkl',
        'thr_a':  API_FOLDER / 'transport_05' / 'transport_threshold_raw.npy',
        'thr_b':  API_FOLDER / 'transport_05' / 'transport_threshold_b_raw.npy',
        'cfg':    API_FOLDER / 'transport_05' / 'transport_infer_cfg_raw.pkl',
    },
}

CMD_RESET   = 0x01
CMD_GET     = 0x02
CHANNEL_MAP = {'none': 0, 'A': 1, 'B': 2, 'AB': 3}
N_TAXELS    = 16
JUMP_THR    = 0.5     # basket_ori.py 와 동일 (정규화 후 기준)


def _load_minmax(minmax_path):
    """basket_ori.py _load_minmax 와 동일."""
    try:
        mm = np.load(str(minmax_path), allow_pickle=True).item()
        gmin = np.asarray(mm["min"],   dtype=np.float64)
        grng = np.asarray(mm["range"], dtype=np.float64)
        grng[~np.isfinite(grng) | (grng <= 0.0)] = 1.0
        print(f"[Bridge] taxel min-max 로드 완료: {minmax_path}")
        return gmin, grng
    except Exception as e:
        print(f"[Bridge] taxel min-max 로드 실패 ({e}) → 정규화 비활성")
        return np.zeros(48, dtype=np.float64), np.ones(48, dtype=np.float64)


def _normalize_taxels(x_vals, y_vals, z_vals, gmin, grng):
    """basket_ori.py _normalize_taxels 와 동일."""
    A = np.empty(48, dtype=np.float64)
    A[0:16]  = x_vals
    A[16:32] = y_vals
    A[32:48] = z_vals
    An = (A - gmin) / grng
    return An[0:16].copy(), An[16:32].copy(), An[32:48].copy()


def _compute_features(cum_x, cum_y, cum_z):
    """basket_ori.py _compute_features 와 동일한 6D feature 계산."""
    vecs  = np.array([cum_x, cum_y, cum_z], dtype=np.float32).T   # (16, 3)
    norms = np.linalg.norm(vecs, axis=1)                           # (16,)
    max_norm = float(np.max(norms)) if np.max(norms) > 0 else 1e-6

    weights = norms.copy()
    weights[weights == 0] = 1e-6
    mean_vec = np.sum(vecs, axis=0) / np.sum(weights)
    norm     = float(np.linalg.norm(mean_vec))

    valid = norms > 0.1
    if valid.sum() >= 2:
        unit_vecs = vecs[valid] / norms[valid, np.newaxis]
        mean_unit = np.mean(unit_vecs, axis=0)
        mrl       = float(np.linalg.norm(mean_unit))
    else:
        mean_unit = np.zeros(3, dtype=np.float32)
        mrl       = 0.0

    return np.array(
        [mrl, norm, max_norm,
         float(mean_unit[0]), float(mean_unit[1]), float(mean_unit[2])],
        dtype=np.float32,
    )


class XelaBridge:
    def __init__(
        self,
        detector:    DualChannelDetector,
        sensor_id:   int,
        ws_host:     str,
        ws_port:     int,
        tcp_port:    int,
        minmax_path: Path,
        anomaly_consec_needed:  int   = 3,    # 이상 확정에 필요한 연속 프레임 수
        max_norm_gate:          float = 0.3,  # basket_ori 와 동일: 실제 접촉 힘 게이트
    ):
        self.detector              = detector
        self.sensor_id             = str(sensor_id)
        self.ws_url                = f"ws://{ws_host}:{ws_port}"
        self.tcp_port              = tcp_port
        self.anomaly_consec_needed = anomaly_consec_needed
        self.max_norm_gate         = max_norm_gate

        self._gmin, self._grng = _load_minmax(minmax_path)

        self._lock            = threading.Lock()
        self._cum_x           = [0.0] * N_TAXELS
        self._cum_y           = [0.0] * N_TAXELS
        self._cum_z           = [0.0] * N_TAXELS
        self._prev_x          = None
        self._prev_y          = None
        self._prev_z          = None
        self._anomaly_latched    = False 
        self._anomaly_consec     = 0      
        self._waiting_for_reset  = True   
        self._result             = {
            'anomaly': False, 'channel': 'none',
            'score_a': 0.0,   'delta_b': 0.0,
            'state':   0,     # 0=IDLE, 1=NORMAL, 2=ANOMALY
        }

    # ── 내부 리셋 (lock 보유 상태에서 호출) ─────────────────
    def _reset_locked(self):
        self._cum_x              = [0.0] * N_TAXELS
        self._cum_y              = [0.0] * N_TAXELS
        self._cum_z              = [0.0] * N_TAXELS
        self._prev_x             = None
        self._prev_y             = None
        self._prev_z             = None
        self._anomaly_latched    = False
        self._anomaly_consec     = 0
        self._waiting_for_reset  = False  
        self.detector.reset()
        self._result = {
            'anomaly': False, 'channel': 'none',
            'score_a': 0.0,   'delta_b': 0.0,
            'state':   0,
        }

    # ── WebSocket 콜백 ────────────────────────────────────
    def _on_open(self, wsapp):
        print(f"[WS] 연결됨: {self.ws_url}")

    def _on_error(self, wsapp, error):
        print(f"[WS] 오류: {error}")

    def _on_close(self, wsapp, code, msg):
        print(f"[WS] 연결 종료: {code} {msg}")

    def _on_message(self, wsapp, message):
        try:
            data = json.loads(message)
        except Exception:
            return

        if data.get("message") == "Welcome":
            return

        sensor_data = data.get(self.sensor_id)
        if sensor_data is None:
            return

        taxels = sensor_data.get("taxels", 0)
        raw    = sensor_data.get("data", "").split(",")
        if len(raw) != 3 * taxels:
            return

        try:
            values = [int(v, 16) for v in raw]
        except ValueError:
            return

        x_vals = [float(values[i * 3])     for i in range(taxels)]
        y_vals = [float(values[i * 3 + 1]) for i in range(taxels)]
        z_vals = [float(values[i * 3 + 2]) for i in range(taxels)]

        # basket_ori.py 와 동일: 차분 전 min-max 정규화
        x_vals, y_vals, z_vals = _normalize_taxels(x_vals, y_vals, z_vals,
                                                    self._gmin, self._grng)

        with self._lock:
            if self._waiting_for_reset:
                return  # RESET 전까지 추론 안 함

            if self._prev_x is None:
                self._prev_x, self._prev_y, self._prev_z = x_vals, y_vals, z_vals
                return

            dx = [x_vals[i] - self._prev_x[i] for i in range(taxels)]
            dy = [y_vals[i] - self._prev_y[i] for i in range(taxels)]
            dz = [z_vals[i] - self._prev_z[i] for i in range(taxels)]

            max_diff = max(abs(d) for d in dx + dy + dz)
            if max_diff > JUMP_THR:
                self._cum_x  = [0.0] * taxels
                self._cum_y  = [0.0] * taxels
                self._cum_z  = [0.0] * taxels
                self._prev_x, self._prev_y, self._prev_z = x_vals, y_vals, z_vals
                return

            for i in range(taxels):
                self._cum_x[i] += dx[i]
                self._cum_y[i] += dy[i]
                self._cum_z[i] += dz[i]

            self._prev_x, self._prev_y, self._prev_z = x_vals, y_vals, z_vals
            frame_6d = _compute_features(self._cum_x, self._cum_y, self._cum_z)
            vecs = np.array([self._cum_x, self._cum_y, self._cum_z],
                                  dtype=np.float32).T       
            raw_frame = vecs.flatten()                        
            max_norm = float(frame_6d[2])
            if max_norm < self.max_norm_gate:
                return

            result = self.detector.update(frame_6d, raw_frame=raw_frame)

            score_a = result['score_a']
            if score_a == 0.0:
                state = 0  # IDLE
                self._anomaly_consec = 0
            elif self._anomaly_latched:
                state = 2  # ANOMALY latch 
            elif result['anomaly']:
                self._anomaly_consec += 1
                if self._anomaly_consec >= self.anomaly_consec_needed:
                    self._anomaly_latched = True
                    state = 2  # ANOMALY latch
                else:
                    state = 1  
            else:
                self._anomaly_consec = 0
                state = 1  # NORMAL

            result['state']   = state
            result['anomaly'] = (state == 2)
            self._result = result

    # ── WebSocket 스레드 (자동 재연결) ────────────────────
    def _ws_thread(self):
        while True:
            wsapp = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            wsapp.run_forever(reconnect=5)
            print("[WS] 재연결 대기 중...")
            time.sleep(2)

    # ── TCP 클라이언트 핸들러 ─────────────────────────────
    def _handle_client(self, conn, addr):
        print(f"[TCP] 연결: {addr}  (Enter/RESET 신호 대기 중)")
        try:
            while True:
                data = conn.recv(1)
                if not data:
                    break
                cmd = data[0]

                if cmd == CMD_RESET:
                    with self._lock:
                        self._reset_locked()

                elif cmd == CMD_GET:
                    with self._lock:
                        r = dict(self._result)
                    payload = struct.pack(
                        '<BBffBB',
                        int(r['anomaly']),
                        CHANNEL_MAP.get(r['channel'], 0),
                        float(r['score_a']),
                        float(r['delta_b']),
                        int(r.get('state', 0)), 0,
                    )
                    conn.sendall(payload)

        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            conn.close()
            print(f"[TCP] 연결 종료: {addr}")

    # ── 메인 루프 ─────────────────────────────────────────
    def run(self):
        t = threading.Thread(target=self._ws_thread, daemon=True)
        t.start()

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('0.0.0.0', self.tcp_port))
        srv.listen(4)
        print(f"[TCP] 0.0.0.0:{self.tcp_port} 대기 중...")

        while True:
            conn, addr = srv.accept()
            ct = threading.Thread(
                target=self._handle_client, args=(conn, addr), daemon=True
            )
            ct.start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset',    default='grasp_05',
                    choices=list(_PRESETS.keys()),
                    help='api_folder 내 모델 preset (기본: grasp_05)')
    ap.add_argument('--model',     default=None, help='*_gru_ae.pt (preset 대신 직접 지정)')
    ap.add_argument('--scaler',    default=None, help='*_scaler.pkl')
    ap.add_argument('--thr_a',     default=None, help='*_threshold.npy (파일 경로)')
    ap.add_argument('--thr_b',     default=None, help='*_threshold_b.npy (파일 경로)')
    ap.add_argument('--thr_a_val', type=float, default=None,
                    help='Channel A threshold 직접 지정 (float). 제공 시 --thr_a 파일 값 무시.'
                         ' 예: --thr_a_val 0.12')
    ap.add_argument('--cfg',       default=None, help='*_infer_cfg.pkl (optional)')
    ap.add_argument('--ws_host',     default='localhost', help='xela_server 호스트')
    ap.add_argument('--ws_port',     type=int, default=5000, help='xela_server WebSocket 포트')
    ap.add_argument('--sensor_id',   type=int, default=1,    help='사용할 센서 ID')
    ap.add_argument('--tcp_port',    type=int, default=9001, help='C 클라이언트용 TCP 포트')
    ap.add_argument('--anomaly_consec', type=int, default=3,
                    help='이상 확정 연속 프레임 수 (기본: 3)')
    ap.add_argument('--max_norm_gate', type=float, default=0.3,
                    help='추론 활성화 최소 max_norm (basket_ori 기본값=0.3, 0이면 게이트 비활성)')
    ap.add_argument('--minmax',      default=None,
                    help='taxel minmax 정규화 파일 경로 (npy) (기본: tactile_anomaly/taxel_minmax_norm.npy)')
    args = ap.parse_args()

    p = _PRESETS[args.preset]
    preset_defaults = p.get('defaults', {})
    cli_provided = {a.dest for a in ap._actions if a.option_strings
                    and a.dest in vars(args)
                    and vars(args)[a.dest] != a.default}
    for key, val in preset_defaults.items():
        if key not in cli_provided:
            setattr(args, key, val)

    model_path  = args.model  or str(p['model'])
    scaler_path = args.scaler or str(p['scaler'])
    thr_a_path  = args.thr_a  or str(p['thr_a'])
    thr_b_path  = args.thr_b  or str(p['thr_b'])
    cfg_path    = args.cfg    or str(p['cfg'])

    default_minmax = Path(__file__).parent / 'taxel_minmax_norm.npy'
    minmax_path = Path(args.minmax) if args.minmax else default_minmax

    print(f"[Bridge] preset={args.preset}")
    print(f"         model         = {model_path}")
    print(f"         scaler        = {scaler_path}")
    print(f"         thr_a         = {thr_a_path}")
    print(f"         thr_b         = {thr_b_path}")
    print(f"         cfg           = {cfg_path}")
    print(f"         minmax        = {minmax_path}")
    print(f"         thr_a_val     = {args.thr_a_val}")
    print(f"         anomaly_consec= {args.anomaly_consec}")
    print(f"         max_norm_gate = {args.max_norm_gate}")

    detector = DualChannelDetector.from_checkpoint(
        model_path       = model_path,
        scaler_path      = scaler_path,
        threshold_a_path = thr_a_path,
        threshold_b_path = thr_b_path,
        infer_cfg_path   = cfg_path,
    )
    print("[Bridge] 모델 로드 완료.")

    if args.thr_a_val is not None:
        old = detector.threshold_a
        detector.threshold_a = args.thr_a_val
        print(f"[Bridge] threshold_A 오버라이드: {old:.6f} → {args.thr_a_val:.6f}")

    print(f"[Bridge] RESET 신호(0x01)를 받으면 누적 버퍼를 초기화하고 추론을 시작합니다.")
    print(f"[Bridge] max_norm_gate={args.max_norm_gate} (0이면 비활성)")
    bridge = XelaBridge(
        detector              = detector,
        sensor_id             = args.sensor_id,
        ws_host               = args.ws_host,
        ws_port               = args.ws_port,
        tcp_port              = args.tcp_port,
        minmax_path           = minmax_path,
        anomaly_consec_needed = args.anomaly_consec,
        max_norm_gate         = args.max_norm_gate,
    )
    bridge.run()


if __name__ == '__main__':
    main()
