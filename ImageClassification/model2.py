#DEFINITIONE DELLA RETE NEURALE CON AGGIUNTA DELLA BATCH NORMALIZATION

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ClassificationCNN, self).__init__()
        
        # --- BLOCCO 1 ---
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # --- BLOCCO 2 ---
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # --- BLOCCO 3 ---
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        
        # Flatten: 64x64 -> 32 -> 16 -> 8. Output: 64 canali * 8 * 8
        self.flatten_dim = 64 * 8 * 8
        
        # Classificatore (Senza Dropout, come da "Best Configuration")
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Feature Extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten & Decision
        x = x.view(-1, self.flatten_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class RegressionCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(RegressionCNN, self).__init__()
        
        # --- VISUAL STREAM ---
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        # Flatten: 128x128 -> 64 -> 32 -> 16 -> 8. Output: 128 * 8 * 8
        self.visual_dim = 128 * 8 * 8
        
        # --- SEMANTIC STREAM ---
        self.cls_fc = nn.Linear(5, 16)
        
        # --- FUSION ---
        self.fusion_dim = self.visual_dim + 16
        
        self.fc1 = nn.Linear(self.fusion_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2) # Output x, y

    def forward(self, x, class_idx):
        # 1. Visual Processing
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        v_img = x.view(-1, self.visual_dim)
        
        # 2. Semantic Processing
        batch_size = x.size(0)
        one_hot = torch.zeros(batch_size, 5, device=x.device)
        one_hot.scatter_(1, class_idx.view(-1, 1), 1)
        
        v_sem = F.relu(self.cls_fc(one_hot))
        
        # 3. Late Fusion
        v_fused = torch.cat((v_img, v_sem), dim=1)
        
        # 4. Regression Head (Senza Dropout)
        out = F.relu(self.fc1(v_fused))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out