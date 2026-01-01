import pandas as pd #per leggere i file CVS
import numpy as np  
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import StandardScaler #per normalizzare i dati
from sklearn.metrics import r2_score, mean_squared_error


#---CONFIGURAZIONE --
FILE_NAME = 'reacher3_train_1.csv'
IMAGE_NAME = 'linear_regression_results_eng.png'
print(f"Test Baseline Lineare su: {FILE_NAME}")

#1. Caricamento
df = pd.read_csv(FILE_NAME)
 
X = df.iloc[:,0:7].values #iloc sta per Integer Location, seleziona i dati basandosi sull'indice delle colonne
                          #prendiamo tutte le righe del file e le 7 colonne, con values prendiamo solo i dati numerici trasformandoli in una matrice, eliminando i nomi
y = df.iloc[:,7:10].values #con 7:10 prendiamo le colonne dalla 7 alla 0 (10 esclusa)

# 2. Split (Train vs Test)
#Usiamo il 20% dei dati per vedere se il modello generalizza, in questo modo evitiamo l'overfitting.  
#test_size=0.2 ci da il 20%, mentre random_state=42 stiamo dando un numero fisso di randomicità per mescolare i dati sempre allo stesso modo
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# 3. Normalizzazione 
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

X_test_scaled = scaler_X.transform(X_test)
#non scaliamo anche y perchè ci serve fare il confronto tra i valori reali  (gradi con gradi)

# 4. Addestramento (Fit)
#Proviamo con qualcosa di semplice come la regressione lineare, cercando di trovare una linea reta che approssima il movimento del robot
print("Training Linear Regression...")
model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)

# 5. Predizione e valutazione
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled) # "Denormalizziamo" le previsioni per vedere i valori reali

#Calcolo matrice
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("-" * 30)
print("RISULTATI BASELINE (LINEAR REGRESSION):")
print(f"MSE (Errore Quadratico Medio): {mse:.6f}")
print(f"R2 Score (Accuratezza): {r2:.4f}")
print("-" * 30)

if r2 < 0.5:
    print("INTERPRETAZIONE: Il punteggio è basso.")
    print("Il robot si muove seguendo curve (seni/coseni), non linee rette.")
    print("Questo GIUSTIFICA l'uso di una Rete Neurale!")
else:
    print("INTERPRETAZIONE: Sorprendente! Il modello lineare funziona decentemente.")

#RICORDA: In ML la y è l'obiettivo, cioè quello che vogliamo il modello impari a calcolare.
        # La X rappresenta invece tutte le info che abbiamo a disposizione per trovare la risposta (matematicamente, gli argomenti della funzione)
        #In questo caso sono le prime 7 colonne della tabella

#-------------SALVATAGGIO DATI -------------

# 5. IL GRAFICO (Visualizziamo solo il primo motore - Giunto 1)
plt.figure(figsize=(9, 8)) # Leggermente più largo per far stare il testo comodo

# --- INIZIO MODIFICHE ESTETICHE ---

# Disegniamo i punti (Label tradotta in Inglese)
plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.3, color='blue', label='Predictions')

# Disegniamo la "Linea della Perfezione" (Label tradotta in Inglese)
min_val = min(y_test[:, 0].min(), y_pred[:, 0].min())
max_val = max(y_test[:, 0].max(), y_pred[:, 0].max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal (Perfect)')

# Titoli e Assi in Inglese (Richiesto per report scientifico)
plt.title('Linear Regression: Actual vs. Predicted (Joint 1)', fontsize=14)
plt.xlabel('Actual Value (real dq)', fontsize=12)
plt.ylabel('Predicted Value (estimated dq)', fontsize=12)

# --- AGGIUNTA RIQUADRO CON I DATI ---
# Creiamo il testo da inserire
metrics_text = f"Model Performance:\n\nMSE: {mse:.4e}\n$R^2$ Score: {r2:.4f}"

# Definiamo lo stile del riquadro (bianco semitrasparente con bordi arrotondati)
bbox_props = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9)

# Inseriamo il testo nel grafico.
# (0.05, 0.95) sono coordinate relative: significa 5% da sinistra e 95% dal basso (angolo in alto a sx)
plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=bbox_props)
# --- FINE MODIFICHE ---

plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)

# Salviamo l'immagine
plt.savefig(IMAGE_NAME, dpi=300, bbox_inches='tight') # dpi=300 per alta qualità
print(f"🖼️ Grafico salvato con metriche come '{IMAGE_NAME}'")
plt.close() # Chiude la figura per liberare memoria
