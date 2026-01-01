import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# --- CONFIGURAZIONE MASTER ---
# CAMBIA SOLO QUESTA RIGA PER PASSARE A REACHER 4 o 6
CURRENT_ROBOT = "Reacher4"  # Opzioni: "Reacher3", "Reacher4", "Reacher6"

# Dizionario con le specifiche di ogni robot
ROBOT_CONFIG = {
    "Reacher3": {
        "file": "reacher3_train_1.csv",
        "q_dims": 3,   # Numero giunti
        "x_dims": 2,   # Posizione (x,y)
        "input_cols": 7, # Totale input raw
        "output_cols": 3
    },
    "Reacher4": {
        "file": "reacher4_train_1.csv", 
        "q_dims": 4,
        "x_dims": 3,   # Reacher4 è 3D (x,y,z)
        "input_cols": 10, # x(3) + q(4) + dx(3) = 10
        "output_cols": 4
    },
    "Reacher6": {
        "file": "reacher6_train_1.csv",
        "q_dims": 6,
        "x_dims": 3,
        "input_cols": 12, # x(3) + q(6) + dx(3) = 12
        "output_cols": 6
    }
}

cfg = ROBOT_CONFIG[CURRENT_ROBOT]
FILE_NAME = cfg['file']
MODEL_NAME = f'model_{CURRENT_ROBOT}.h5'
SCALER_X_NAME = f'scaler_x_{CURRENT_ROBOT}.pkl'
SCALER_Y_NAME = f'scaler_y_{CURRENT_ROBOT}.pkl'

print(f"🤖 AVVIO TRAINING PER: {CURRENT_ROBOT}")
print(f"   File: {FILE_NAME}")
print(f"   Giunti: {cfg['q_dims']}, Spazio: {cfg['x_dims']}D")

# 1. CARICAMENTO DATI
try:
    df = pd.read_csv(FILE_NAME)
except FileNotFoundError:
    print(f"❌ ERRORE: Non trovo il file {FILE_NAME}. Controlla la cartella!")
    exit()

# Slicing dinamico basato sulla configurazione
X_raw = df.iloc[:, 0:cfg['input_cols']].values
y = df.iloc[:, cfg['input_cols']:].values

print(f"   Shape Originale X: {X_raw.shape}, y: {y.shape}")

# 2. FEATURE ENGINEERING (Sin/Cos)
# Dobbiamo capire dove sono le colonne 'q' per calcolarne seno e coseno
# Struttura dataset: [x...x] [q...q] [dx...dx]
start_q = cfg['x_dims']
end_q = start_q + cfg['q_dims']

q_values = X_raw[:, start_q:end_q] # Estraiamo solo le colonne q

# Creiamo le feature trigonometriche
sin_q = np.sin(q_values)
cos_q = np.cos(q_values)

# Ricostruiamo X: [x] + [sin_q] + [cos_q] + [dx]
X_smart = np.hstack([
    X_raw[:, 0:start_q],    # Posizione x
    sin_q,                  # Seno angoli
    cos_q,                  # Coseno angoli
    X_raw[:, end_q:]        # Spostamento dx
])

print(f"   Shape Smart X: {X_smart.shape} (Feature Engineering applicato)")

# 3. SPLIT E SCALING
X_train, X_test, y_train, y_test = train_test_split(X_smart, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)

# Salvataggio Scaler
joblib.dump(scaler_X, SCALER_X_NAME)
joblib.dump(scaler_y, SCALER_Y_NAME)

# 4. MODELLO DINAMICO
input_shape = X_smart.shape[1]
output_shape = cfg['output_cols']

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_shape,)),
    tf.keras.layers.Dense(512, activation='swish'),
    tf.keras.layers.Dense(512, activation='swish'),
    tf.keras.layers.Dense(256, activation='swish'),
    tf.keras.layers.Dense(128, activation='swish'),
    
    tf.keras.layers.Dense(output_shape, activation='linear')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')

# 5. TRAINING
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
]

print(f"\n🏋️ Inizio Training {CURRENT_ROBOT}...")
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.2,
    epochs=200, # Tante epoche, ci pensa l'early stopping
    batch_size=32,
    callbacks=callbacks,
    verbose=2,
    shuffle=True
)

# 6. VALUTAZIONE
y_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
r2 = r2_score(y_test, y_pred)

print("=" * 40)
print(f"🤖 RISULTATO {CURRENT_ROBOT}: R2 = {r2:.4f}")
print("=" * 40)

model.save(MODEL_NAME)

# Salvataggio Grafico con nome dinamico
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title(f'{CURRENT_ROBOT} Learning Curve (R2: {r2:.4f})')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.savefig(f'learning_curve_{CURRENT_ROBOT}.png')
print(f"🖼️ Grafico salvato: learning_curve_{CURRENT_ROBOT}.png")