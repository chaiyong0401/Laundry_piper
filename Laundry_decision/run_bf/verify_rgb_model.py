
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import resnet18
from PIL import Image
import os
import sys

# Paths
MODEL_PATH = "/home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/rgb_only.pth"
TRAIN_IMG = "/home/dyros/mcy_ws/Laundry_decision/dataset_for_rgb_gate/empty/empty_1768453519565.jpg"
DEBUG_IMG = "/home/dyros/mcy_ws/Laundry_decision/run_bf/_dbg/rgb_gate_input_1768456148.703.jpg"

def create_model():
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m

def load_model(path, device):
    print(f"Loading {path}...")
    ckpt = torch.load(path, map_location=device)
    state_dict = None
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
        
    model = create_model().to(device).eval()
    
    # Handle 'model.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    return model

def predict(model, img_path, device):
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        return

    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        img = Image.open(img_path).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            
        p_empty = probs[0].item()
        p_cloth = probs[1].item()
        
        pred = "CLOTH" if p_cloth > p_empty else "EMPTY"
        print(f"[{img_path}]")
        print(f"   -> Empty: {p_empty:.4f}, Cloth: {p_cloth:.4f} => Prediction: {pred}")
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(MODEL_PATH, device)
    
    print("-" * 50)
    print("Testing Training Data (Should be EMPTY):")
    predict(model, TRAIN_IMG, device)
    
    print("-" * 50)
    print("Testing Live Debug Data (The problematic one):")
    predict(model, DEBUG_IMG, device)

if __name__ == "__main__":
    main()
