import os
import sys
import shutil
import numpy as np
import torch
import joblib
import pickle

# Setup path to import tactile_anomaly
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tactile_anomaly.detector import DualChannelDetector
from tactile_anomaly.model import GRUAE
from sklearn.preprocessing import StandardScaler

def create_mock_checkpoints(dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    
    scaler = StandardScaler()
    dummy_data = np.random.randn(100, 48)
    scaler.fit(dummy_data)
    joblib.dump(scaler, os.path.join(dest_dir, 'mock_scaler.pkl'))
    
    model = GRUAE(input_dim=48, hidden_dim=64, num_layers=2, dropout=0.2, seq_len=50)
    torch.save(model.state_dict(), os.path.join(dest_dir, 'mock_gru_ae.pt'))
    
    np.save(os.path.join(dest_dir, 'mock_threshold.npy'), np.array([0.02]))
    np.save(os.path.join(dest_dir, 'mock_threshold_b.npy'), np.array([0.15]))
    
    infer_cfg = {
        "relative": "none",
        "weight_alpha": 0.0,
        "use_raw": True,
        "seq_len": 50,
        "hybrid": False,
        "n_rel_dims": None
    }
    with open(os.path.join(dest_dir, 'mock_infer_cfg.pkl'), 'wb') as f:
        pickle.dump(infer_cfg, f)


def run_simulation(detector):
    print("\n--- 실시간 이상 감지 시뮬레이션 시작 ---")
    detector.reset()
    
    base_frame = np.array([0.8, 1.0, 1.0, 0.1, 0.1, 0.98])
    base_raw_frame = np.zeros(48)
    
    print("\n[테스트 1] 정상 파지 유지 상태 (50프레임 입력)")
    for i in range(50):
        noise = np.random.normal(0, 0.005, size=6)
        frame = base_frame + noise
        frame[0] = np.clip(frame[0], 0.0, 1.0)
        
        raw_noise = np.random.normal(0, 0.05, size=48)
        raw_frame = base_raw_frame + raw_noise
        
        result = detector.update(frame, raw_frame=raw_frame)
        
        if i % 10 == 0 or i == 49:
            print(f"  Frame {i:02d} | anomaly={result['anomaly']:<5} | channel={result['channel']:<5} | "
                  f"score_A={result['score_a']:.4f} | delta_B={result['delta_b']:.4f}")

    print("\n[테스트 2] 급격한 미끄러짐/충격 발생 (채널 B 감지 테스트)")
    impact_frame = base_frame.copy()
    impact_frame[0] = 0.2  
    raw_frame = base_raw_frame + np.random.normal(0, 0.05, size=48)
    
    result = detector.update(impact_frame, raw_frame=raw_frame)
    print(f"  Impact Frame | anomaly={result['anomaly']:<5} | channel={result['channel']:<5} | "
          f"score_A={result['score_a']:.4f} | delta_B={result['delta_b']:.4f}")
    if result['anomaly'] and 'B' in result['channel']:
        print("  >> Channel B (MRL delta) 이상 감지")
    else:
        print("  >> Channel B 이상 감지에 실패")
    print("\n[테스트 3] 지속적인 편차 누적 (채널 A 감지 테스트)")
    detector.reset()
    
    for i in range(50):
        frame = base_frame + np.random.normal(0, 0.005, size=6)
        frame[0] = np.clip(frame[0], 0.0, 1.0)
        
        raw_frame = base_raw_frame + 4.5
        
        result = detector.update(frame, raw_frame=raw_frame)
        
        if i == 49:
            print(f"  Abnormal Frame 50 | anomaly={result['anomaly']:<5} | channel={result['channel']:<5} | "
                  f"score_A={result['score_a']:.4f} | delta_B={result['delta_b']:.4f}")
            if result['anomaly'] and 'A' in result['channel']:
                print("  >> 성공: Channel A (GRU-AE) 이상 감지")
            else:
                print("  >> 실패: Channel A 이상 감지에 실패")




if __name__ == "__main__":
    mock_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_mock_temp')
    try:
        create_mock_checkpoints(mock_dir)
        
        detector = DualChannelDetector.from_checkpoint(
            model_path       = os.path.join(mock_dir, 'mock_gru_ae.pt'),
            scaler_path      = os.path.join(mock_dir, 'mock_scaler.pkl'),
            threshold_a_path = os.path.join(mock_dir, 'mock_threshold.npy'),
            threshold_b_path = os.path.join(mock_dir, 'mock_threshold_b.npy'),
            infer_cfg_path   = os.path.join(mock_dir, 'mock_infer_cfg.pkl'),
        )
        
        run_simulation(detector)
        
    except Exception as e:
        print(f"\n[오류 발생] 작동 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(mock_dir):
            shutil.rmtree(mock_dir)
            print("\n3. 임시 파일 정리 완료.")
