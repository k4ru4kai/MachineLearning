import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

# --- IMPORTAZIONI DAI TUOI FILE ---
from dataset import SoccerDataset
from model2 import ClassificationCNN, RegressionCNN

# --- CONFIGURAZIONE GLOBALE ---
# Definiamo qui le costanti usate da tutti
CSV_FILE = "train_labels.csv"
IMG_DIR = "train_images"  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Hardware in uso: {DEVICE}")

    # ==========================================
    # PARTE 1: CLASSIFICAZIONE
    # ==========================================
    print("\n" + "="*40)
    print(" AVVIO TRAINING CLASSIFICAZIONE ")
    print("="*40)

    # 1. Carichiamo i dati (Nota: uso 'img_dir' invece di 'root_dir' per evitare errori)
    print("Caricamento dataset Classificazione...")
    train_dataset_cls = SoccerDataset(csv_file=CSV_FILE, img_dir=IMG_DIR, task='classification')
    train_loader_cls = DataLoader(train_dataset_cls, batch_size=32, shuffle=True)

    # 2. Prepariamo il modello
    model_cls = ClassificationCNN(num_classes=5).to(DEVICE)
    criterion_cls = nn.CrossEntropyLoss()
    optimizer_cls = optim.Adam(model_cls.parameters(), lr=0.001)

    # 3. Loop di Training (5 Epoche)
    epochs_cls = 5
    model_cls.train()

    for epoch in range(epochs_cls):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader_cls):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer_cls.zero_grad()
            outputs = model_cls(images)
            loss = criterion_cls(outputs, labels)
            loss.backward()
            optimizer_cls.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"[Classificazione] Epoch {epoch+1}/{epochs_cls} | Loss: {running_loss/len(train_loader_cls):.4f} | Acc: {acc:.2f}%")

    # 4. Salvataggio
    torch.save(model_cls.state_dict(), "model_classification.pth")
    print("✅ Modello Classificazione salvato!")


    # ==========================================
    # PARTE 2: REGRESSIONE
    # ==========================================
    print("\n" + "="*40)
    print(" AVVIO TRAINING REGRESSIONE ")
    print("="*40)

    # 1. Carichiamo i dati (Mode: Regression)
    print("Caricamento dataset Regressione...")
    train_dataset_reg = SoccerDataset(csv_file=CSV_FILE, img_dir=IMG_DIR, task='regression')
    train_loader_reg = DataLoader(train_dataset_reg, batch_size=32, shuffle=True)

    # 2. Prepariamo il modello
    model_reg = RegressionCNN().to(DEVICE)
    
    # IMPORTANTE: Usiamo SmoothL1Loss come deciso dall'analisi
    criterion_reg = nn.SmoothL1Loss() 
    optimizer_reg = optim.Adam(model_reg.parameters(), lr=0.001)

    # 3. Loop di Training (10 Epoche)
    epochs_reg = 10
    model_reg.train()

    for epoch in range(epochs_reg):
        running_loss = 0.0
        
        for i, (images, targets, labels_idx) in enumerate(train_loader_reg):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            labels_idx = labels_idx.to(DEVICE)
            
            optimizer_reg.zero_grad()
            # Late Fusion: passiamo immagine E classe
            outputs = model_reg(images, labels_idx)
            loss = criterion_reg(outputs, targets)
            loss.backward()
            optimizer_reg.step()
            
            running_loss += loss.item()
        
        print(f"[Regressione] Epoch {epoch+1}/{epochs_reg} | Loss (SmoothL1): {running_loss/len(train_loader_reg):.4f}")

    # 4. Salvataggio
    torch.save(model_reg.state_dict(), "model_regression.pth")
    print("✅ Modello Regressione salvato!")

    print("\n--- PIPELINE COMPLETATA CON SUCCESSO ---")

if __name__ == "__main__":
    main()