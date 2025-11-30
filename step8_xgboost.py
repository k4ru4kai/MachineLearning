import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import time

# --- CONFIGURAZIONE ---
# Cambia qui per provare Reacher4 o Reacher6
ROBOT_NAME = "Reacher3" 

ROBOT_SPECS = {
    "Reacher3": {"file": "reacher3_train_1.csv", "in": 7, "x_dims": 2, "q_dims": 3},
    "Reacher4": {"file": "reacher4_train_1.csv", "in": 10, "x_dims": 3, "q_dims": 4},
    "Reacher6": {"file": "reacher6_train_1.csv", "in": 12, "x_dims": 3, "q_dims": 6}
}

spec = ROBOT_SPECS[ROBOT_NAME]
print(f"🌲 Avvio XGBoost (Decision Trees) su: {ROBOT_NAME}")

# 1. CARICAMENTO DATI
df = pd.read_csv(spec["file"])
X_raw = df.iloc[:, 0:spec["in"]].values
y = df.iloc[:, spec["in"]:].values

# Feature Engineering (Sempre utile anche per gli alberi!)
start_q = spec["x_dims"]
end_q = start_q + spec["q_dims"]
sin_q = np.sin(X_raw[:, start_q:end_q])
cos_q = np.cos(X_raw[:, start_q:end_q])
X_smart = np.hstack([X_raw[:, 0:start_q], sin_q, cos_q, X_raw[:, end_q:]])

# Split
X_train, X_test, y_train, y_test = train_test_split(X_smart, y, test_size=0.2, random_state=42)

# XGBoost non richiede scaling obbligatorio, ma lo usiamo per coerenza con la NN
scaler_X = StandardScaler()
scaler_y = StandardScaler() # Scaliamo anche y per aiutare la convergenza
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)

# 2. DEFINIZIONE DEL MODELLO XGBOOST
# Usiamo MultiOutputRegressor perché dobbiamo predire più giunti insieme
print("   Configurando XGBoost (questo è potente!)...")

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,    # Numero di alberi (più sono, meglio è, fino a un certo punto)
    learning_rate=0.05,   # Passo di apprendimento (basso = più preciso)
    max_depth=8,          # Profondità degli alberi (più profondo = capisce relazioni complesse)
    subsample=0.8,        # Usa solo l'80% dei dati per ogni albero (evita overfitting)
    colsample_bytree=0.8, # Usa solo l'80% delle feature per albero
    n_jobs=-1,            # Usa tutti i core della CPU
    random_state=42
)

model = MultiOutputRegressor(xgb_model)

# 3. TRAINING
print("🏋️ Training in corso (può volerci un minuto)...")
start_time = time.time()
model.fit(X_train_scaled, y_train_scaled)
print(f"   Finito in {time.time() - start_time:.2f} secondi.")

# 4. VALUTAZIONE
print("\n🧪 Valutazione...")
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

r2 = r2_score(y_test, y_pred)

print("=" * 40)
print(f"🌲 XGBOOST R2 SCORE ({ROBOT_NAME}): {r2:.4f}")
print("=" * 40)

# Salvataggio (Joblib è meglio per scikit-learn/xgboost)
joblib.dump(model, f'xgboost_model_{ROBOT_NAME}.pkl')
print("💾 Modello salvato come .pkl")

# Grafico veloce Predizione vs Realtà (Solo primo giunto)
plt.figure(figsize=(8, 8))
plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.2, s=10, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f'XGBoost: Realtà vs Predizione (Giunto 1) - R2: {r2:.3f}')
plt.xlabel('Reale')
plt.ylabel('Predetto')
plt.savefig(f'xgboost_scatter_{ROBOT_NAME}.png')