#HOMEWORK 2-IMAGE CLASSIFICATION
#Gestore dei dati


import os  #permette di interaggire con il sistema operativo
import pandas as pd 
import cv2 #gestisce le immagini
import torch
from torch.utils.data import Dataset 
import numpy as np
import traceback

#standardardizzazione delle dimensioni delle immagini
SIZE_RITAGLI = (64,64)  #Dimensione fissa per ritagli 
SIZE_INTERE = (128,128) #Dimensione fissa per le immagini intere

class SoccerDataset(Dataset): #Questa classe permette di impostare l'accesso al dataset fornito

    # 1. Inizializzazione dell'oggetto
    def __init__(self, csv_file, img_dir, task='classification', transform=None): #costruttore con i parametri che dovrà leggere
        """
        Docstring for reader:
            csv_file (string): Percorso al file csv
            img_dir (string): Cartella con le immagini
            task (string): scelta tra classificazione o regressione
                    
        """
        #questi medoti vanno identati cosi altrimenti Python si confonde e pensa che siano funzioni separate, non leggendomi self
        df = pd.read_csv(csv_file, header=0) 
        df.columns = ['filename', 'width', 'height', 'label', 'xmin', 'ymin', 'xmax', 'ymax']
        self.data_frame = df
        self.img_dir = img_dir
        self.task = task
        self.transform = transform

        self.classes = self.data_frame['label'].unique() #estraiamo la colonna label dal dataframe perchè python lavora solo con i numeri
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}   #dizionario di mapping per convertire le etichette testuali in numeri

        print(f"Dataset caricato. Trovate {len(df)} immagini e {len(self.classes)} classi.")

    # 2. Capiamo quanto è grande il Dataset
    def __len__(self):
        return len(self.data_frame) #restituisce il numero di elementi nel dataset
    
    # 3. Accesso agli elementi
    def __getitem__(self, idx):
        try:
            #recupera info dalla riga corrente
            row = self.data_frame.iloc[idx] #iloc serve proprio a recuperare la riga numero idx
            img_name = row['filename'] #nome della colonna
            img_path = os.path.join(self.img_dir, img_name) #costruiamo il percorso completo dell'immagine 

            # 2. Caricamento immagine ROBUSTO
            if not os.path.exists(img_path):
                # Se il file non esiste, stampiamo un warning (solo la prima volta magari)
                 #erestituiamo un'immagine nera per non bloccare il training.
                # print(f"[WARNING] File mancante: {img_name}") 
                image = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                image = cv2.imread(img_path)
                if image is None:
                    # File esiste ma è corrotto
                    image = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            #Recupera dimesioni originali e coordinate
            h_orig, w_orig, _ = image.shape #ottengo rispettivamente altezza, larghezza e num di canali
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])
            label_str = row['label']
            label_idx = self.class_to_idx[label_str]

            # TASK 1: CLASSIFICATION
            if self.task == 'classification':
                crop_img = image[ymin:ymax, xmin:xmax] #Bounding Box

                #Gestione caso limite
                if crop_img.size == 0:
                    crop_img = image #Fallback sull'immagine intera

                #Conversiamo l'imagine in un oggetto PyTorch
                img_resized = cv2.resize(crop_img, SIZE_RITAGLI) #ridimesiona il crop ad una dimensione fissa
                img_tensor = torch.from_numpy(img_resized).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1)

                return img_tensor, label_idx
    
        #  TASK 2: REGRESSIONE -------------------------------------------------------
            elif self.task == 'regression':

                #Calcolo il centro originale
                center_x = (xmin+xmax) / 2.0
                center_y = (ymin+ymax) / 2.0

                #Resize
                img_resized = cv2.resize(image, SIZE_INTERE)
                img_tensor = torch.tensor(img_resized / 255.0, dtype=torch.float32).permute(2, 0, 1)
                

                # Ricacolo delle coordinate del centro in proporzione al resize
                if w_orig == 0 or h_orig == 0:
                    target_x_norm, target_y_norm = 0.5, 0.5
                else:
                    scale_x = SIZE_INTERE[0] / w_orig
                    scale_y = SIZE_INTERE[1] / h_orig
                    #Normalizza i target tra 0 e 1
                    target_x_norm = (center_x * scale_x) / SIZE_INTERE[0]
                    target_y_norm = (center_y * scale_y) / SIZE_INTERE[1]

                target_tensor = torch.tensor([target_x_norm, target_y_norm], dtype=torch.float32)
                return img_tensor, label_idx, target_tensor
                
        except Exception:
            # Stampa l'errore ma ci dice anche DOVE è successo
            print(f"\n[ERRORE INTERNO a __getitem__ indice {idx}]")
            traceback.print_exc()
            raise # Rilancia l'errore per fermare tutto   

# --- BLOCCO DI TEST ---
if __name__ == "__main__":
    print("Test avvio...")
    # Si assume che la cartella 'spqr_dataset' sia nella stessa directory di questo file .py
    TEST_CSV = "spqr_dataset/raw/bbx_annotations.csv" 
    TEST_IMG_DIR = "spqr_dataset/images"

    try:
        # Testiamo la modalità classificazione
        print(f"Sto cercando il CSV qui: {TEST_CSV}")
        print(f"Sto cercando le immagini qui: {TEST_IMG_DIR}")
        
        ds = SoccerDataset(csv_file=TEST_CSV, img_dir=TEST_IMG_DIR, task='classification')
        
        # Proviamo a caricare la prima immagine reale per essere sicuri
        img, label = ds[0] 
        print(f"SUCCESSO! Immagine caricata. Dimensioni tensore: {img.shape}")
        print(f"Label (classe numerica): {label}")
        
    except FileNotFoundError as e:
        print(f"\nERRORE: Non trovo i file. Controlla che il nome della cartella sia esattamente 'spqr_dataset'")
        print(f"Dettaglio errore: {e}")
    except Exception as e:
        print(f"\nErrore generico: {e}")       



    


                






    


