# models.py
import torch
import torch.nn as nn
from torchvision import models

class EnhancedOfflineBERT(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True, dropout=0.1),
            num_layers=3
        )
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, 0, :]
        x = self.dropout(x)
        return self.fc(x)

class EmotionResNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)