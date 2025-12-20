import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os

# Nasconde i messaggi di debug inutili, ma lascia gli errori
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FILE_NAME = 'reacher3_train_1.csv'
BEST_MODEL_NAME = 'best_model_reacher3.h5'

print(f"🔬 Avvio Grid Search (Versione Fix Keras 3) su: {FILE_NAME}")

# --- 1. DATI (Identico a prima) ---
df = pd.read_csv(FILE_NAME)
X = df.iloc[:, 0:7].values
y = df.iloc[:, 7:10].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)

# --- 2. GRID SEARCH ---
neurons_list = [64, 128]
epochs_list = [50, 100]

results = []
best_r2 = -np.inf
best_config = {}

print(f"\nInizio test di {len(neurons_list) * len(epochs_list)} combinazioni...")

for neurons in neurons_list:
    for epochs in epochs_list:
        print("\n" + "="*50)
        print(f"🔄 TEST: Neuroni={neurons}, Epoche={epochs}")
        
        # Reset totale della sessione
        tf.keras.backend.clear_session()
        tf.random.set_seed(42)
        np.random.seed(42)

        # COSTRUZIONE MODELLO 
        model = tf.keras.Sequential([
            # FIX: Input Layer esplicito per evitare problemi di inizializzazione
            tf.keras.layers.Input(shape=(7,)), 
            tf.keras.layers.Dense(neurons, activation='relu'),
            tf.keras.layers.Dense(neurons, activation='relu'),
            tf.keras.layers.Dense(3, activation='linear')
        ])
        
        # Optimizer con learning rate esplicito (per stabilità)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        
        # ADDESTRAMENTO
        # verbose=2 mostra una riga per epoca -> controlla se la 'loss' scende!
        print("   Addestramento in corso...")
        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_split=0.2,
            epochs=epochs,
            batch_size=32,
            verbose=0, # Rimettiamo 0 per pulizia, ma controlliamo la loss finale sotto
            shuffle=True
        )
        
        # DIAGNOSTICA: Controlliamo se ha imparato qualcosa
        final_loss = history.history['loss'][-1]
        print(f"   📉 Loss Finale Training: {final_loss:.5f}")
        
        if final_loss > 0.5:
            print("   ⚠️ ATTENZIONE: La Loss è alta! La rete non sta imparando.")

        # VALUTAZIONE
        y_pred_scaled = model.predict(X_test_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        score = r2_score(y_test, y_pred)
        
        print(f"   ⭐ R2 Score: {score:.4f}")
        
        results.append({'neurons': neurons, 'epochs': epochs, 'r2': score})
        
        if score > best_r2:
            print("   ✅ NUOVO RECORD!")
            best_r2 = score
            best_config = {'neurons': neurons, 'epochs': epochs}
            model.save(BEST_MODEL_NAME)

print("\n" + "="*50)
print(f"🏆 VINCITORE: {best_config}")
print(f"🏅 R2 Score: {best_r2:.4f}")

print("\n📋 Tabella Riassuntiva:")
print(f"{'Neuroni':<10} | {'Epoche':<10} | {'R2 Score':<10}")
print("-" * 35)
for res in results:
    print(f"{res['neurons']:<10} | {res['epochs']:<10} | {res['r2']:.4f}")

