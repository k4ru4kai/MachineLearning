import pandas as pd

# 1. Carica il file (Assicurati che il nome sia esatto!)
# Se hai scaricato il file in una sottocartella, aggiungi il percorso (es. 'data/reacher...')
filename = 'reacher3_train_1.csv' 

try:
    df = pd.read_csv(filename)
    
    print("-" * 30)
    print(f"✅ FILE CARICATO: {filename}")
    print(f"Dimensioni: {df.shape}")
    print("-" * 30)
    
    print("NOMI DELLE COLONNE:")
    # Stampiamo l'elenco con l'indice per capire dove tagliare
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")
        
except FileNotFoundError:
    print(f"❌ ERRORE: Non trovo il file '{filename}' nella cartella.")
    print("Verifica di averlo scaricato e messo nella cartella ML_RobotControl")