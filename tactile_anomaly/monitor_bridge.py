#!/usr/bin/env python3
"""
xela_bridge 실시간 상태 모니터링 클라이언트.

상태:
  IDLE    - RESET 직후 버퍼 충전 중 (seq_len 프레임 채우는 중)
  NORMAL  - 정상 파지 (score_A < threshold)
  ANOMALY - 이상 감지 (score_A or delta_B > threshold, latch)

Enter 키: 브릿지에 RESET 신호 전송 (그리퍼 닫는 순간에 누름)

Usage:
  python3 tactile_anomaly/monitor_bridge.py
  python3 tactile_anomaly/monitor_bridge.py --port 9001 --hz 10
"""

import argparse
import socket
import struct
import threading
import time

CMD_RESET   = bytes([0x01])
CMD_GET     = bytes([0x02])

CHANNEL_STR = {0: 'none', 1: 'A', 2: 'B', 3: 'AB'}
STATE_STR   = {
    0: ('IDLE   ', '\033[90m'),    # 회색  - 비접촉
    1: ('NORMAL ', '\033[32m'),    # 초록  - 정상 파지
    2: ('ANOMALY', '\033[31m'),    # 빨강  - 이상 감지 (latch)
}
RESET_COLOR = '\033[0m'


def recv_exact(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionResetError
        data += chunk
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=9001)
    ap.add_argument('--hz',   type=float, default=10.0, help='폴링 주기 (Hz)')
    ap.add_argument('--no_color', action='store_true', help='색상 출력 비활성화')
    args = ap.parse_args()

    interval  = 1.0 / args.hz
    use_color = not args.no_color

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))

    print(f"[Monitor] {args.host}:{args.port} 연결됨  |  {args.hz}Hz  |  Enter=RESET  |  Ctrl+C 종료")
    print(f"{'time':>8}  {'state':<7}  {'channel':<6}  {'score_A':>12}  {'delta_B':>9}  log")
    print("-" * 72)

    reset_flag = threading.Event()

    def _stdin_thread():
        while True:
            try:
                input()   # Enter 대기
                reset_flag.set()
            except EOFError:
                break

    t = threading.Thread(target=_stdin_thread, daemon=True)
    t.start()

    prev_state = -1
    frame = 0

    try:
        while True:
            t0 = time.time()

            if reset_flag.is_set():
                reset_flag.clear()
                sock.sendall(CMD_RESET)
                ts = time.strftime('%H:%M:%S')
                print(f"{ts}  \033[33m{'RESET  '}\033[0m  {'':27}  [Enter] 브릿지 리셋")
                prev_state = -1

            sock.sendall(CMD_GET)
            data = recv_exact(sock, 12)

            anomaly  = data[0]
            channel  = CHANNEL_STR.get(data[1], '??')
            score_a, delta_b = struct.unpack_from('<ff', data, 2)
            state    = data[10]   # 0=IDLE, 1=NORMAL, 2=ANOMALY

            state_label, color = STATE_STR.get(state, ('UNKNOWN', ''))
            if not use_color:
                color = ''
                reset = ''
            else:
                reset = RESET_COLOR

            # 상태 전환 로그
            log_msg = ''
            if state != prev_state:
                transitions = {
                    (0, 1): '→ 버퍼 완료, 정상 파지',
                    (0, 2): '→ 버퍼 완료, 즉시 이상!',
                    (1, 2): '⚠ 이상 감지 (slip/drop?) — RESET 전까지 유지',
                }
                log_msg = transitions.get((prev_state, state), f'state {prev_state}→{state}')
                prev_state = state

            ts = time.strftime('%H:%M:%S')
            print(f"{ts}  {color}{state_label}{reset}  "
                  f"ch={channel:<4}  "
                  f"score_A={score_a:>10.4f}  "
                  f"dB={delta_b:>7.4f}  "
                  f"{color}{log_msg}{reset}")

            frame += 1
            elapsed = time.time() - t0
            time.sleep(max(0.0, interval - elapsed))

    except KeyboardInterrupt:
        print("\n[Monitor] 종료")
    finally:
        sock.close()


if __name__ == '__main__':
    main()
