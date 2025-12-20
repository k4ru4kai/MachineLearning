import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib

#---CONFIGURAZIONE--
FILE_NAME = 'reacher3_train_1.csv'
MODEL_NAME = 'model_reacher.h5' #file che contiene la rete neurale. E' un Hierachical Data Format v5, pensato per salvare grandi quantità di dati in modo strutturato
SCALER_X_NAME = 'scaler_x_reacher3.pkl' #pkl sta per Pickle, il modo standard di Python e trasformarlo in un file disco
SCALER_Y_NAME = 'scaler_y_reacher3.pkl' ##questi file contengono le regole di normalizzazione che ci aiutano a tradurre i dati in ingresso dal liguaggio del robot a quello della rete neurale

print("Avvio rete neurale su {FILE_NAME}")

# 1. CARICAMENTO E PREPARAZIONE
df = pd.read_csv(FILE_NAME)
X = df.iloc[:, 0:7].values # Input (7 features)
y = df.iloc[:, 7:10].values # Output (3 target)

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizzazione
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Fit
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

X_test_scaled = scaler_X.transform(X_test)

# Salvataggio degli scaler
joblib.dump(scaler_X, SCALER_X_NAME)
joblib.dump(scaler_y, SCALER_Y_NAME)
print("Dati processati e Scaler salvati")

# DEFININIZIONE DELLA RETE NEURALE (MPL)
# Definiamo l'architettura
model = tf.keras.Sequential([  #impiliamo i dati uno spr l'altro (comando tensorflow)
    
    # Input layer implicito (7 feature)-->Primo Hidden Layer (64 neuroni)
    tf.keras.layers.Dense(64, activation ='relu', input_shape=(7,)), #creiamo 64 neuroni ognuno dei quali collegato a tutti gli input (dense). 
    #Gli diciamo di aspettarsi in input pacchetti di 7 numeri alla volta con inputshape

    #Secondo Hiddel Layer (64 neuroni) - Aggiunge profondità per capire le curve
    tf.keras.layers.Dense(64, activation ='relu'), #lafunzione di attivazione è una ReLU, che spegne il valore se il valore è negativo e lo lascia passare se è positivo
    #APPROFONDIRE RELU

    #Output Layer (3 neuroni) - Attivazione Lineare per regressione pura
    tf.keras.layers.Dense(3, activation = 'linear') # E' la risposta finale del robot che ha 3 giunti
    #QUi non usasiamo la ReLU perchè i robot possono muoversi sia avanti che indietro e la relu azzera i valori negativi
])

# Compilazione 
#APPROFONDIRE OTTIMIZZATORE ADAM E MSE
model.compile(optimizer='adam', loss='mse', metrics=['mae']) 

# 4.ADDESTRAMENTO
print("\n Traing in corso...")
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.2,
    epochs=50, #Quante volte vede i dati
    batch_size=32, #Ogni quanti esempi aggiorna i pesi
    verbose=1
)

# VALUTAZIONE FINALE 
print("\n Valutazione sul Test Set...")
y_pred_scaled = model.predict(X_test_scaled) #La rete fa predizioni scalate
y_pred = scaler_y.inverse_transform(y_pred_scaled) #Le convertiamo in radianti veri

#Calcoliamo R2 Score
r2=r2_score(y_test, y_pred)

print("-" * 30)
print(f"R2 SCORE (RETE NEURALE): {r2:.4f}")
print("-" * 30)

# 6. SALVATAGGIO GRAFICO E MODELLO
model.save(MODEL_NAME) # [cite: 289]
print(f"Modello salvato in '{MODEL_NAME}'")

# Grafico della Loss (Training vs Validation) - Da mettere nel report 
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'Learning Curve - Reacher3 (R2 Finale: {r2:.2f})')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE Scalato)')
plt.legend()
plt.grid(True)
plt.savefig('learning_curve_reacher3.png') 
print("Grafico salvato come 'learning_curve_reacher3.png'")


