import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SoccerDataset
from model1 import RegressionCNN
import time  # <--- NUOVA IMPORTAZIONE

# --- CONFIGURAZIONE ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
CSV_PATH = 'spqr_dataset/raw/bbx_annotations.csv'
IMG_DIR = 'spqr_dataset/images'

def train():
    # Avvia il cronometro
    start_time = time.time()
    
    print("--- INIZIO TRAINING REGRESSIONE ---")
    
    print("Caricamento dataset...")
    train_dataset = SoccerDataset(csv_file=CSV_PATH, img_dir=IMG_DIR, task='regression')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    num_classes = len(train_dataset.classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sto usando il device: {device}")
    
    model = RegressionCNN(num_classes=num_classes).to(device)
    
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        for i, (images, labels, targets) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, labels)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i+1) % 50 == 0:
                print(f"  > Batch {i+1}/{len(train_loader)} - MSE: {loss.item():.6f}")

        avg_loss = running_loss / len(train_loader)
        print(f"Fine Epoca {epoch+1} -> Loss Media (MSE): {avg_loss:.6f}")

    print("\nSalvataggio modello...")
    torch.save(model.state_dict(), "regression_model.pth")
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