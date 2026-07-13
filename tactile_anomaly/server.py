"""
Tactile Anomaly Detection Server
=================================
C 클라이언트에서 6D feature를 전송하면 이상 감지 결과를 반환.

Protocol (binary):
  C → Python : 6 x float32 = 24 bytes  (raw feature 1 frame)
  Python → C : 12 bytes
                [0]   uint8  : anomaly (0 or 1)
                [1]   uint8  : channel (0=none, 1=A, 2=B, 3=AB)
                [2-5] float32: score_a
                [6-9] float32: delta_b
                [10]  uint8  : reserved
                [11]  uint8  : reserved

Usage:
  python3 -m tactile_anomaly.server \
    --model    grasp_gru_ae.pt \
    --scaler   grasp_scaler.pkl \
    --thr_a    grasp_threshold.npy \
    --thr_b    grasp_threshold_b.npy \
    --cfg      grasp_infer_cfg.pkl \
    --host     0.0.0.0 \
    --port     9000
"""

import argparse
import socket
import struct
import sys

import numpy as np

from .detector import DualChannelDetector

CHANNEL_MAP = {'none': 0, 'A': 1, 'B': 2, 'AB': 3}
FRAME_SIZE  = 6 * 4       # 6 float32
RESULT_SIZE = 12           # 2 uint8 + 2 float32 + 2 uint8 padding


def run_server(detector: DualChannelDetector, host: str, port: int):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[Server] {host}:{port} 에서 대기 중...")

    while True:
        conn, addr = srv.accept()
        print(f"[Server] 연결: {addr}")
        detector.reset()

        try:
            while True:
                # 프레임 수신 (24 bytes)
                data = b''
                while len(data) < FRAME_SIZE:
                    chunk = conn.recv(FRAME_SIZE - len(data))
                    if not chunk:
                        raise ConnectionResetError
                    data += chunk

                frame = np.frombuffer(data, dtype=np.float32)  # (6,)
                result = detector.update(frame)

                # 결과 전송 (12 bytes)
                anomaly = int(result['anomaly'])
                channel = CHANNEL_MAP.get(result['channel'], 0)
                score_a = result['score_a']
                delta_b = result['delta_b']

                payload = struct.pack('<BBffBB',
                                     anomaly, channel,
                                     score_a, delta_b,
                                     0, 0)
                conn.sendall(payload)

        except (ConnectionResetError, BrokenPipeError):
            print(f"[Server] 연결 끊김: {addr} → 버퍼 초기화")
        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  required=True)
    parser.add_argument('--scaler', required=True)
    parser.add_argument('--thr_a',  required=True)
    parser.add_argument('--thr_b',  required=True)
    parser.add_argument('--cfg',    default=None)
    parser.add_argument('--host',   default='0.0.0.0')
    parser.add_argument('--port',   type=int, default=9000)
    args = parser.parse_args()

    detector = DualChannelDetector.from_checkpoint(
        model_path       = args.model,
        scaler_path      = args.scaler,
        threshold_a_path = args.thr_a,
        threshold_b_path = args.thr_b,
        infer_cfg_path   = args.cfg,
    )
    print("[Server] 모델 로드 완료.")
    run_server(detector, args.host, args.port)


if __name__ == '__main__':
    main()
