import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SoccerDataset
from model1 import ClassificationCNN
import time  # <--- NUOVA IMPORTAZIONE

# --- CONFIGURAZIONE ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5
CSV_PATH = 'spqr_dataset/raw/bbx_annotations.csv'
IMG_DIR = 'spqr_dataset/images'

def train():
    # Avvia il cronometro
    start_time = time.time()
    
    print("--- INIZIO TRAINING CLASSIFICAZIONE ---")
    
    # 1. Carichiamo i dati
    print("Caricamento dataset...")
    train_dataset = SoccerDataset(csv_file=CSV_PATH, img_dir=IMG_DIR, task='classification')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    num_classes = len(train_dataset.classes)
    print(f"Classi trovate: {num_classes}")

    # 2. Prepariamo il Modello
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sto usando il device: {device}")
    
    model = ClassificationCNN(num_classes=num_classes).to(device)
    
    # 3. Loss e Ottimizzatore
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Loop di Training
    model.train()
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i+1) % 50 == 0:
                print(f"  > Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        epoch_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Fine Epoca {epoch+1} -> Loss: {avg_loss:.4f} | Acc: {epoch_acc:.2f}%")

    # 5. Salvataggio
    print("\nSalvataggio modello...")
    torch.save(model.state_dict(), "classification_model.pth")
    print("Modello salvato.")
    
    # --- CALCOLO TEMPO FINALE ---
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\n--------------------------------------------------")
    print(f"TEMPO TOTALE ADDESTRAMENTO: {minutes} min {seconds} sec")
    print(f"--------------------------------------------------")

if __name__ == "__main__":
    train()