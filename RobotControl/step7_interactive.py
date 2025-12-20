import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
import joblib
import time
import glob

# Setup pulito
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# Cartella base
BASE_RESULTS_DIR = "results"
if not os.path.exists(BASE_RESULTS_DIR):
    os.makedirs(BASE_RESULTS_DIR)

ROBOT_DIMS = {
    "Reacher3": {"q_dims": 3, "x_dims": 2, "in": 7},
    "Reacher4": {"q_dims": 4, "x_dims": 3, "in": 10},
    "Reacher6": {"q_dims": 6, "x_dims": 3, "in": 12}
}

print("="*60)
print("🗂️  TRAINER ORGANIZZATO (Sottocartelle per ogni test)")
print("="*60)

# --- 1. INPUT ROBOT ---
while True:
    robot_name = input("1. Scegli il Robot (Reacher3, Reacher4, Reacher6): ").strip()
    if robot_name in ROBOT_DIMS: break
    print("❌ Nome non valido!")

# --- 2. SELEZIONE DATASET ---
search_pattern = f"{robot_name.lower()}*.csv"
found_files = sorted(glob.glob(search_pattern))

if not found_files:
    print(f"❌ ERRORE: Nessun file CSV trovato per {robot_name}!")
    exit()

print(f"\n2. Scegli il Dataset:")
for i, f in enumerate(found_files):
    print(f"   [{i}] {f}")

while True:
    try:
        choice = int(input("   Inserisci numero: "))
        if 0 <= choice < len(found_files):
            selected_file = found_files[choice]
            break
        print("❌ Numero non valido.")
    except ValueError:
        print("❌ Inserisci un numero.")

dataset_tag = os.path.splitext(selected_file)[0]

# --- 3. CONFIGURAZIONE ---
print("\n3. Architettura (es. 256,256,128)")
layer_input = input("   Lista neuroni (invio per default [256,128]): ")
if layer_input.strip():
    layers_structure = [int(x) for x in layer_input.split(',')]
else:
    layers_structure = [256, 128]

use_dropout_str = input("\n4. Usa Dropout? (s/n): ").lower()
use_dropout = (use_dropout_str == 's' or use_dropout_str == 'y')

epochs_str = input("\n5. Epoche (invio per 300): ")
epochs = int(epochs_str) if epochs_str else 300

# CREAZIONE ID E SOTTOCARTELLA
struct_name = "-".join(map(str, layers_structure))
drop_name = "dropYes" if use_dropout else "dropNo"
config_id = f"{robot_name}_{dataset_tag}_{struct_name}_{drop_name}_{epochs}ep"

# --- LA MODIFICA È QUI: Creiamo la sottocartella specifica ---
EXPERIMENT_DIR = os.path.join(BASE_RESULTS_DIR, config_id)
if not os.path.exists(EXPERIMENT_DIR):
    os.makedirs(EXPERIMENT_DIR)

print(f"\n📂 CARTELLA OUTPUT: {EXPERIMENT_DIR}")
print(f"🆔 ID SIMULAZIONE: {config_id}")

# --- 4. PREPARAZIONE DATI ---
spec = ROBOT_DIMS[robot_name]
df = pd.read_csv(selected_file)

X_raw = df.iloc[:, 0:spec["in"]].values
y = df.iloc[:, spec["in"]:].values

start_q = spec["x_dims"]
end_q = start_q + spec["q_dims"]
sin_q = np.sin(X_raw[:, start_q:end_q])
cos_q = np.cos(X_raw[:, start_q:end_q])
X_smart = np.hstack([X_raw[:, 0:start_q], sin_q, cos_q, X_raw[:, end_q:]])

X_train, X_test, y_train, y_test = train_test_split(X_smart, y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)

# Salvataggio Scaler NELLA SOTTOCARTELLA
joblib.dump(scaler_X, os.path.join(EXPERIMENT_DIR, 'scaler_x.pkl'))
joblib.dump(scaler_y, os.path.join(EXPERIMENT_DIR, 'scaler_y.pkl'))

# --- 5. TRAINING ---
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(X_smart.shape[1],)))

for neurons in layers_structure:
    model.add(tf.keras.layers.Dense(neurons, activation='swish'))
    if use_dropout:
        model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Dense(y.shape[1], activation='linear'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
]

print(f"\n⏱️  Start Training...")
start_time = time.time()

history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.2,
    epochs=epochs,
    batch_size=64,
    callbacks=callbacks,
    verbose=2,
    shuffle=True
)

end_time = time.time()
elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)
time_str = f"{int(minutes)}m {int(seconds)}s"

# --- 6. RISULTATI ---
y_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print(f"🏆 RISULTATO: {config_id}")
print(f"   R2 Score (Validation): {r2:.4f}")
print(f"   Tempo: {time_str}")
print("="*50)

# --- 7. SALVATAGGIO NELLA SOTTOCARTELLA ---
# Non serve più mettere l'ID nel nome del file, perché è già nel nome della cartella!
# Questo rende i nomi file più puliti (sempre "model.h5", "plot.png")

model.save(os.path.join(EXPERIMENT_DIR, "model.h5"))

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title(f'R2: {r2:.3f} | Time: {time_str} | Config: {layers_structure}')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(EXPERIMENT_DIR, "learning_curve.png"))

txt_path = os.path.join(EXPERIMENT_DIR, "report_summary.txt")
with open(txt_path, "w") as f:
    f.write(f"ROBOT: {robot_name}\n")
    f.write(f"DATASET: {selected_file}\n")
    f.write(f"CONFIG: {layers_structure} (Dropout: {use_dropout})\n")
    f.write(f"R2 SCORE: {r2:.5f}\n")
    f.write(f"TEMPO: {time_str}\n")

print(f"\n✅ Tutto salvato ordinatamente in:\n   {EXPERIMENT_DIR}/")