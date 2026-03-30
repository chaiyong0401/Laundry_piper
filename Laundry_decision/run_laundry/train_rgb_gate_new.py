
import os
import glob
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SimpleRGBDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=224):
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        
        # Expected structure:
        # root_dir/
        #   cloth/  (Positive, Class 1)
        #   empty/  (Negative, Class 0)
        
        cloth_files = glob.glob(os.path.join(root_dir, 'cloth', '*.*'))
        empty_files = glob.glob(os.path.join(root_dir, 'empty', '*.*'))
        
        # Filter only images
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        cloth_files = [f for f in cloth_files if os.path.splitext(f)[1].lower() in valid_exts]
        empty_files = [f for f in empty_files if os.path.splitext(f)[1].lower() in valid_exts]

        print(f"Found {len(cloth_files)} cloth images, {len(empty_files)} empty images.")
        
        # Simple split (80% train, 20% val) - shuffling with fixed seed
        random.shuffle(cloth_files)
        random.shuffle(empty_files)
        
        n_cloth_train = int(len(cloth_files) * 0.8)
        n_empty_train = int(len(empty_files) * 0.8)
        
        if split == 'train':
            self.files = cloth_files[:n_cloth_train] + empty_files[:n_empty_train]
            self.labels = [1] * n_cloth_train + [0] * n_empty_train
        else:
            self.files = cloth_files[n_cloth_train:] + empty_files[n_empty_train:]
            self.labels = [1] * (len(cloth_files) - n_cloth_train) + [0] * (len(empty_files) - n_empty_train)
            
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) if split=='train' else T.Identity(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros((3, self.img_size, self.img_size)), label

def train(args):
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    train_ds = SimpleRGBDataset(args.data_dir, split='train')
    val_ds = SimpleRGBDataset(args.data_dir, split='val')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model (ResNet18)
    print("Creating ResNet18 model...")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # Binary: Empty(0) vs Cloth(1)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    best_acc = 0.0
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    print("Start training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_ds)
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f} Val Acc: {val_acc:.4f}")
        
        if val_acc >= best_acc:
            best_acc = val_acc
            print(f"  >> New Best! Saving to {args.save_path}")
            # Save raw state dict (key 'model.' prefix issue fixed in loader, but let's save standard)
            torch.save(model.state_dict(), args.save_path)
            
            # Save Meta
            with open(args.save_path + ".json", "w") as f:
                json.dump({"idx_to_class": {0: "empty", 1: "cloth"}}, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset/ (must contain cloth/ and empty/)')
    parser.add_argument('--save_path', type=str, default='/home/dyros/mcy_ws/Laundry_decision/run_laundry/models/rgb_only_best.pth')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    args = parser.parse_args()
    train(args)
