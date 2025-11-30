import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
import joblib
from datetime import datetime  # <--- NUOVO: Per gestire data e ora

# Setup pulito
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# --- CONFIGURAZIONE COSTANTI ---
RESULTS_DIR = "results"  # Cartella dove finirà tutto
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    print(f"📂 Creata cartella '{RESULTS_DIR}' per i risultati.")

ROBOT_SPECS = {
    "Reacher3": {"file": "reacher3_train_1.csv", "q_dims": 3, "x_dims": 2, "in": 7},
    "Reacher4": {"file": "reacher4_train_1.csv", "q_dims": 4, "x_dims": 3, "in": 10},
    "Reacher6": {"file": "reacher6_train_1.csv", "q_dims": 6, "x_dims": 3, "in": 12}
}

print("="*60)
print("🧪 LABORATORIO SPERIMENTALE CON LOGGING AUTOMATICO")
print("="*60)

# --- 1. INPUT UTENTE ---

while True:
    robot_name = input("1. Robot (Reacher3, Reacher4, Reacher6): ").strip()
    if robot_name in ROBOT_SPECS: break
    print("❌ Nome non valido.")

print("\n2. Architettura (es. 256,256,128)")
layer_str = input("   Lista neuroni: ")
layers_structure = [int(x) for x in layer_str.split(',')]

use_dropout_str = input("\n3. Usa Dropout? (s/n): ").lower()
use_dropout = (use_dropout_str == 's' or use_dropout_str == 'y')

epochs_str = input("\n4. Epoche (invio per 300): ")
epochs = int(epochs_str) if epochs_str else 300

# Generiamo un ID unico per questo esperimento ADESSO
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_id = f"{robot_name}_{timestamp}"

print(f"\n🆔 ID ESPERIMENTO: {experiment_id}")
print("..." * 10)

# --- 2. PREPARAZIONE DATI ---
spec = ROBOT_SPECS[robot_name]
try:
    df = pd.read_csv(spec["file"])
except FileNotFoundError:
    print(f"❌ File {spec['file']} non trovato!")
    exit()

X_raw = df.iloc[:, 0:spec["in"]].values
y = df.iloc[:, spec["in"]:].values

# Feature Engineering
start_q = spec["x_dims"]
end_q = start_q + spec["q_dims"]
sin_q = np.sin(X_raw[:, start_q:end_q])
cos_q = np.cos(X_raw[:, start_q:end_q])
X_smart = np.hstack([X_raw[:, 0:start_q], sin_q, cos_q, X_raw[:, end_q:]])

# Split & Scale
X_train, X_test, y_train, y_test = train_test_split(X_smart, y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)

# Salvataggio Scaler (Questi li sovrascriviamo perché servono sempre gli ultimi per il robot)
joblib.dump(scaler_X, f'scaler_x_{robot_name}.pkl')
joblib.dump(scaler_y, f'scaler_y_{robot_name}.pkl')

# --- 3. TRAINING ---
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(X_smart.shape[1],)))

for neurons in layers_structure:
    model.add(tf.keras.layers.Dense(neurons, activation='swish'))
    if use_dropout:
        model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Dense(y.shape[1], activation='linear'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
]

print("🏋️ Training avviato...")
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.2,
    epochs=epochs,
    batch_size=64,
    callbacks=callbacks,
    verbose=2,
    shuffle=True
)

# --- 4. VALUTAZIONE E LOGGING ---
y_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*40)
print(f"🏆 RISULTATO: R2 = {r2:.4f}")
print("="*40)

# --- SALVATAGGIO FILE ---

# 1. Salva il Modello con ID unico
model_filename = os.path.join(RESULTS_DIR, f"model_{experiment_id}_R2_{r2:.3f}.h5")
model.save(model_filename)

# 2. Salva il Grafico con ID unico
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title(f'Exp: {experiment_id} | Layers: {layers_structure} | R2: {r2:.4f}')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plot_filename = os.path.join(RESULTS_DIR, f"plot_{experiment_id}.png")
plt.savefig(plot_filename)

# 3. Salva il Report Testuale (Log)
log_filename = os.path.join(RESULTS_DIR, f"log_{experiment_id}.txt")
log_content = f"""
========================================
REPORT ESPERIMENTO: {experiment_id}
DATA: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
========================================

ROBOT: {robot_name}
R2 SCORE: {r2:.6f}

CONFIGURAZIONE:
- Architettura: {layers_structure}
- Dropout: {use_dropout} (Rate 0.1)
- Epoche Max: {epochs}
- Batch Size: 64
- Learning Rate: 0.0005
- Input Features: {X_smart.shape[1]} (Feature Eng. applicato)

RISULTATI TRAINING:
- Epoche effettive: {len(history.history['loss'])}
- Loss Finale (Train): {history.history['loss'][-1]:.6f}
- Loss Finale (Val): {history.history['val_loss'][-1]:.6f}

FILE GENERATI:
- Modello: {model_filename}
- Grafico: {plot_filename}
========================================
"""

with open(log_filename, "w") as f:
    f.write(log_content)

print(f"\n✅ Salvato tutto nella cartella '{RESULTS_DIR}/':")
print(f"   📄 Log: log_{experiment_id}.txt")
print(f"   🖼️ Plot: plot_{experiment_id}.png")
print(f"   💾 Model: model_{experiment_id}_R2_{r2:.3f}.h5")