import torch
import cv2
import numpy as np
import random
import os
import torch.nn.functional as F
from dataset import SoccerDataset
from model2 import ClassificationCNN, RegressionCNN

# --- CONFIGURAZIONI ---
CSV_PATH = 'spqr_dataset/raw/bbx_annotations.csv'
IMG_DIR = 'spqr_dataset/images'

# Funzione per centrare il testo
def draw_centered_text(img, text, y_pos, color, font_scale=0.6, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_w = text_size[0]
    img_w = img.shape[1]
    
    x_pos = (img_w - text_w) // 2
    cv2.putText(img, text, (x_pos, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)

def visualize_results():
    print("--- GENERAZIONE IMMAGINI REPORT (HIGH RES) ---")
    
    device = torch.device("cpu")
    
    # 1. Carichiamo Dataset
    ds = SoccerDataset(csv_file=CSV_PATH, img_dir=IMG_DIR, task='regression')
    
    # 2. Carichiamo Modelli
    print("Caricamento modelli...")
    
    # Classificazione
    model_cls = ClassificationCNN(num_classes=5).to(device)
    try:
        model_cls.load_state_dict(torch.load("classification_model.pth", map_location=device))
        model_cls.eval() 
    except Exception as e:
        print(f"ERRORE Classificazione: {e}")
        return

    # Regressione
    model_reg = RegressionCNN(num_classes=5).to(device)
    try:
        model_reg.load_state_dict(torch.load("regression_model.pth", map_location=device))
        model_reg.eval()
    except Exception as e:
        print(f"ERRORE Regressione: {e}")
        return

    # 3. Indici casuali
    indices = random.sample(range(len(ds)), 20)
    
    for i, idx in enumerate(indices):
        img_tensor, true_label_idx, true_target = ds[idx]
        
        # --- PREDIZIONI (Sui dati originali 128x128) ---
        input_tensor = img_tensor.unsqueeze(0).to(device)
        
        # 1. Classe
        input_cls = F.interpolate(input_tensor, size=(64, 64), mode='bilinear', align_corners=False)
        out_cls = model_cls(input_cls)
        pred_label_idx = torch.argmax(out_cls, dim=1).item()
        pred_label_str = ds.classes[pred_label_idx]
        true_label_str = ds.classes[true_label_idx]
        
        # 2. Coordinate
        label_input = torch.tensor([pred_label_idx]).to(device)
        out_reg = model_reg(input_tensor, label_input)
        
        # --- PREPARAZIONE VISIVA (UPSCALING) ---
        img_numpy = img_tensor.permute(1, 2, 0).numpy()
        img_display = (img_numpy * 255).astype(np.uint8).copy()
        img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
        
        # INGRANDIAMO L'IMMAGINE A 300x300 per vederla bene
        DISPLAY_SIZE = (300, 300)
        img_display = cv2.resize(img_display, DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)
        h, w, _ = img_display.shape
        
        # Ricalcoliamo le coordinate dei pallini sulla nuova dimensione
        pred_x = int(out_reg[0][0].item() * w)
        pred_y = int(out_reg[0][1].item() * h)
        true_x = int(true_target[0].item() * w)
        true_y = int(true_target[1].item() * h)
        
        # --- DISEGNO ---
        # Pallino Rosso (Predizione) - Più grande perché l'img è più grande
        cv2.circle(img_display, (pred_x, pred_y), 8, (0, 0, 255), -1) 
        # Cerchio Verde (Vero)
        cv2.circle(img_display, (true_x, true_y), 10, (0, 255, 0), 3)  
        
        # --- BORDI POLAROID ---
        top_border = 60
        bottom_border = 60
        img_final = cv2.copyMakeBorder(
            img_display, 
            top_border, 
            bottom_border, 
            0, 0, 
            cv2.BORDER_CONSTANT, 
            value=(255, 255, 255) # Bianco
        )
        
        # --- TESTO (Ora centrato e leggibile) ---
        if pred_label_idx == true_label_idx:
            info_text = f"Class: {pred_label_str} (Correct)"
            text_color = (0, 0, 0)
        else:
            info_text = f"Pred: {pred_label_str} | True: {true_label_str}"
            text_color = (0, 0, 255)
            
        draw_centered_text(img_final, info_text, 40, text_color, font_scale=0.7, thickness=2)

        # Legenda in basso
        # Disegno i pallini legenda
        cv2.circle(img_final, (50, h + top_border + 30), 8, (0, 0, 255), -1)
        cv2.putText(img_final, "Predizione", (70, h + top_border + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 1)
        
        cv2.circle(img_final, (180, h + top_border + 30), 8, (0, 255, 0), 2)
        cv2.putText(img_final, "Target", (200, h + top_border + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 1)

        filename = f"report_HD_{i}.png"
        cv2.imwrite(filename, img_final)
        print(f" > Salvata: {filename}")

    print("--- FINE ---")

if __name__ == "__main__":
    visualize_results()