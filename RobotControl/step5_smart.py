import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# --- CONFIGURAZIONE ---
FILE_NAME = 'reacher3_train_1.csv'
BEST_MODEL_NAME = 'smart_model_reacher3.h5'

print(f"🧠 Avvio Training con Feature Engineering (Sin/Cos) su: {FILE_NAME}")

# 1. CARICAMENTO BASE
df = pd.read_csv(FILE_NAME)
X_raw = df.iloc[:, 0:7].values # x(2), q(3), dx(2)
y = df.iloc[:, 7:10].values    # dq(3)

# 2. FEATURE ENGINEERING (Il trucco matematico)
# Estraniamo le colonne degli angoli q (indici 2, 3, 4)
q_values = X_raw[:, 2:5] 

# Calcoliamo Seno e Coseno
sin_q = np.sin(q_values)
cos_q = np.cos(q_values)

# Ricostruiamo la matrice X espansa:
# [x, y] + [sin_q] + [cos_q] + [dx, dy]
# 2 col + 3 col + 3 col + 2 col = 10 COLONNE TOTALI
X_smart = np.hstack([
    X_raw[:, 0:2],  # x, y
    sin_q,          # sin(q1, q2, q3)
    cos_q,          # cos(q1, q2, q3)
    X_raw[:, 5:7]   # dx, dy
])

print(f"   Dati trasformati! Nuova Input Shape: {X_smart.shape} (era 7, ora è 10)")

# 3. SPLIT E SCALING
X_train, X_test, y_train, y_test = train_test_split(X_smart, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)

# 4. MODELLO OTTIMIZZATO (Swish + Architettura Media)
# Usiamo 'swish' invece di 'relu' perché è più fluido per la regressione
activation_fn = 'swish' 

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)), # Ora l'input è 10!
    
    tf.keras.layers.Dense(256, activation=activation_fn),
    tf.keras.layers.Dense(256, activation=activation_fn),
    tf.keras.layers.Dense(128, activation=activation_fn),
    
    tf.keras.layers.Dense(3, activation='linear')
])

# Learning Rate un po' più basso per essere precisi
opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=opt, loss='mse', metrics=['mae'])

# 5. TRAINING
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
]

print("\n🏋️ Training 'Smart' in corso...")
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.2,
    epochs=300,
    batch_size=32,
    callbacks=callbacks,
    verbose=2,
    shuffle=True
)

# 6. VALUTAZIONE
print("\n🧪 Valutazione...")
y_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
r2 = r2_score(y_test, y_pred)

print("=" * 40)
print(f"🚀 R2 SCORE FINALE: {r2:.4f}")
print("=" * 40)

model.save(BEST_MODEL_NAME)