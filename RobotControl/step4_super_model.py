import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

# Setup pulito
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# --- CONFIGURAZIONE ---
FILE_NAME = 'reacher3_train_1.csv'
BEST_MODEL_NAME = 'super_model_reacher3.h5'

print(f"🚀 Avvio Training Avanzato (Deep + Callbacks) su: {FILE_NAME}")

# 1. CARICAMENTO DATI (Standard)
df = pd.read_csv(FILE_NAME)
X = df.iloc[:, 0:7].values
y = df.iloc[:, 7:10].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)

# 2. DEFINIZIONE DEL SUPER MODELLO
# 4 Strati nascosti da 256 neuroni (Deep & Wide)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(7,)), # Input esplicito
    
    # Blocco Deep Learning
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    
    # Output Layer
    tf.keras.layers.Dense(3, activation='linear')
])

# Optimizer Adam standard
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 3. I "CALLBACKS" (L'arma segreta)
callbacks_list = [
    # A. EarlyStopping: Se la 'val_loss' non migliora per 20 epoche, STOP.
    # restore_best_weights=True è CRUCIALE: alla fine torna indietro al momento migliore.
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        restore_best_weights=True,
        verbose=1
    ),
    
    # B. ReduceLROnPlateau: Se non migliori per 10 epoche, dimezza il learning rate.
    # Questo aiuta a scendere nel "minimo" con precisione millimetrica.
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=10, 
        min_lr=0.00001,
        verbose=1
    )
]

# 4. ADDESTRAMENTO LUNGO
print("\n🏋️ Inizio Training Intensivo (Max 300 epoche)...")
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.2,
    epochs=300,        # Mettiamo tante epoche, tanto ci pensa EarlyStopping a fermarsi
    batch_size=32,
    callbacks=callbacks_list, # Attiviamo i superpoteri
    verbose=2,         # Una riga per epoca
    shuffle=True
)

# 5. VALUTAZIONE
print("\n🧪 Valutazione Finale...")
y_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
r2 = r2_score(y_test, y_pred)

print("=" * 40)
print(f"🔥 R2 SCORE AVANZATO: {r2:.4f}")
print("=" * 40)

# Salviamo il modello
model.save(BEST_MODEL_NAME)

# Grafico Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title(f'Deep Learning Curve (R2: {r2:.4f})')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.savefig('super_learning_curve.png')
print("🖼️ Grafico salvato come 'super_learning_curve.png'")