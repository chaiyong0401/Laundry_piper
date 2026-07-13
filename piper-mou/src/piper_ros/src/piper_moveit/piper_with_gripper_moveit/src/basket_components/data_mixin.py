import os
import datetime
import threading
import numpy as np


class DataMixin:
    """촉각 데이터 수집/저장 및 사람 레이블 인터페이스."""

    def _spin_input(self, prompt: str) -> str:
        """input()을 별도 스레드에서 실행하며 대기."""
        result = [None]
        done = threading.Event()

        def _ask():
            try:
                result[0] = input(prompt).strip().lower()
            except EOFError:
                result[0] = ''
            finally:
                done.set()

        t = threading.Thread(target=_ask, daemon=True)
        t.start()
        done.wait()
        t.join(timeout=1.0)
        return result[0] if result[0] is not None else ''

    def _save_grasp_data(self, label: str):
        """수집된 파지 구간 시계열을 레이블과 함께 저장."""
        if len(self.grasp_buffer) < 10:
            self.get_logger().warn(f"[DATA] 프레임 수 부족 ({len(self.grasp_buffer)}), 저장 스킵")
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        seq = np.array(self.grasp_buffer, dtype=np.float32)
        fname = os.path.join(self.data_save_dir, f"grasp_{ts}_{label}.npy")
        np.save(fname, seq)
        raw = np.array(self.grasp_raw_buffer, dtype=np.float32)
        fname_raw = os.path.join(self.data_save_dir, f"grasp_{ts}_{label}_raw.npy")
        np.save(fname_raw, raw)
        self.get_logger().info(f"[DATA] 저장 완료: {fname} (shape={seq.shape}), raw: {raw.shape}")

    def _save_insert_data(self, label: str):
        """수집된 이송 구간 시계열을 레이블과 함께 저장."""
        if len(self.insert_buffer) < 10:
            self.get_logger().warn(f"[DATA] 이송 프레임 수 부족 ({len(self.insert_buffer)}), 저장 스킵")
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        seq = np.array(self.insert_buffer, dtype=np.float32)
        fname = os.path.join(self.insert_save_dir, f"transport_{ts}_{label}.npy")
        np.save(fname, seq)
        raw = np.array(self.insert_raw_buffer, dtype=np.float32)
        fname_raw = os.path.join(self.insert_save_dir, f"transport_{ts}_{label}_raw.npy")
        np.save(fname_raw, raw)
        self.get_logger().info(f"[DATA] 이송 저장 완료: {fname} (shape={seq.shape}), raw: {raw.shape}")

    def _ask_insertion_result(self):
        """open_gripper 직전: 이송 구간 파지 유지 성공 여부를 사람에게 확인.
        Returns: True=성공 / False=실패 / None=skip
        """
        print("\n" + "="*50)
        print("[DATA COLLECTION] 이송 구간 파지 유지 결과 확인")
        print("  y = 성공 (세탁물을 놓치지 않고 세탁기에 넣음)")
        print("  n = 실패 (이송 중 낙하)")
        print("  x = skip")
        print("="*50)
        while True:
            key = self._spin_input("입력 (y/n/x): ")
            if key in ('y', 'n', 'x'):
                break
            print("  y / n / x 중 하나를 입력하세요")
        if key == 'y':
            return True
        elif key == 'n':
            return False
        else:
            return None

    def _ask_gripper_held(self):
        """그리퍼에 물체가 잡혀있는지 사람에게 확인.
        Returns: True=잡혀있음 / False=비어있음 / None=skip
        """
        print("\n" + "="*50)
        print("[DATA COLLECTION] 그리퍼 상태 확인")
        print("  y = 물체가 잡혀있음  → 이송 시도 후 결과로 레이블 결정")
        print("  n = 그리퍼 비어있음  → failure 저장, 재시도")
        print("  x = skip             → rule-based fallback")
        print(f"  [참고] norm={self.norm:.3f}, theta={self.theta:.1f}, max_norm={self.max_norm:.3f}")
        print("="*50)
        while True:
            key = self._spin_input("입력 (y/n/x): ")
            if key in ('y', 'n', 'x'):
                break
            print("  y / n / x 중 하나를 입력하세요")
        if key == 'y':
            return True
        elif key == 'n':
            return False
        else:
            return None
