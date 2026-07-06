import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# ========== 配置 ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

EMOTIONS_CN = ['愤怒', '厌恶', '恐惧', '快乐', '悲伤', '惊讶', '中性']

# ========== 数据预处理 ==========
train_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ========== 数据集类 ==========
class FER2013ParquetDataset(Dataset):
    def __init__(self, parquet_path, transform=None, min_quality=0.0):
        print(f"读取: {parquet_path}")
        self.df = pd.read_parquet(parquet_path)
        print(f"  原始样本数: {len(self.df)}")
        
        if 'quality_score' in self.df.columns:
            self.df = self.df[self.df['quality_score'] >= min_quality].reset_index(drop=True)
            print(f"  过滤后样本数: {len(self.df)} (quality >= {min_quality})")
        
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        pixels = row['pixels']
        if isinstance(pixels, str):
            pixel_list = pixels.split()
        else:
            pixel_list = list(pixels)
        
        img_array = np.array(pixel_list, dtype=np.uint8).reshape(48, 48)
        image = Image.fromarray(img_array, mode='L')
        
        if self.transform:
            image = self.transform(image)
        
        label = int(row['emotion'])
        return image, label

# ========== 加载数据 ==========
print("="*60)
print("加载本地数据集")
print("="*60)

train_dataset = FER2013ParquetDataset('data/train.parquet', transform=train_transform, min_quality=0.3)
val_dataset = FER2013ParquetDataset('data/val.parquet', transform=test_transform, min_quality=0.3)
test_dataset = FER2013ParquetDataset('data/test.parquet', transform=test_transform, min_quality=0.0)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ========== 模型定义 ==========
class EmotionResNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model=models.resnet18(weights='IMAGENET1K_V1') 
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# ========== 训练函数 ==========
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), 100 * correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(loader), 100 * correct / total, all_preds, all_labels

# ========== 主训练 ==========
def main():
    model = EmotionResNet().to(DEVICE)
    
    emotion_counts = [0] * 7
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        emotion_counts[label] += 1
    
    print(f"\n各类别数量: {emotion_counts}")
    # 手动调整权重，缓解 Happy/Sad/Neutral 混淆
    custom_weights = [7.0, 65.0, 7.0, 15.0, 4.0, 9.0, 2.5]
    class_weights = torch.FloatTensor(custom_weights).to(DEVICE)
    print(f"类别权重: {class_weights.cpu().numpy().round(2)}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_acc = 0
    
    print("\n开始训练...")
    print("="*70)
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch [{epoch+1:2d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_emotion_model.pth')
            print(f"  -> 保存最佳模型 (验证准确率: {best_acc:.2f}%)")
    
    print("="*70)
    print(f"训练完成！最佳验证准确率: {best_acc:.2f}%")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, label='Train')
    axes[0].plot(val_losses, label='Validation')
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    
    axes[1].plot(train_accs, label='Train')
    axes[1].plot(val_accs, label='Validation')
    axes[1].set_title('Accuracy Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("\n训练曲线已保存: training_curves.png")
    
    print("\n在测试集上评估...")
    model.load_state_dict(torch.load('best_emotion_model.pth'))
    _, test_acc, preds, labels = evaluate(model, test_loader, criterion)
    print(f"测试集准确率: {test_acc:.2f}%")
    
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS_CN, yticklabels=EMOTIONS_CN)
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("混淆矩阵已保存: confusion_matrix.png")
    
    print("\n分类报告:")
    print(classification_report(labels, preds, target_names=EMOTIONS_CN, digits=4))

if __name__ == '__main__':
    main()