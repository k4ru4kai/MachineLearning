import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# --- CONFIGURAZIONE ---
FILE_NAME = 'reacher3_train_1.csv'
IMAGE_NAME = 'xgboost_results_eng.png'

print(f"🌲 Generazione Grafico XGBoost per: {FILE_NAME}")

# 1. CARICAMENTO E PREPARAZIONE
df = pd.read_csv(FILE_NAME)
X_raw = df.iloc[:, 0:7].values
y = df.iloc[:, 7:10].values

# Feature Engineering (Seno/Coseno) - Lo usiamo per essere onesti nel confronto
q_values = X_raw[:, 2:5]
sin_q = np.sin(q_values)
cos_q = np.cos(q_values)
X_smart = np.hstack([X_raw[:, 0:2], sin_q, cos_q, X_raw[:, 5:7]])

# Split
X_train, X_test, y_train, y_test = train_test_split(X_smart, y, test_size=0.2, random_state=42)

# Scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)

# 2. ADDESTRAMENTO XGBOOST
print("   Training XGBoost (Wait...)...")
xgb_estimator = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    n_jobs=-1,
    random_state=42
)
model = MultiOutputRegressor(xgb_estimator)
model.fit(X_train_scaled, y_train_scaled)

# 3. PREDIZIONE
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Calcolo metriche
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 4. GRAFICO (Visualizziamo Joint 1)
plt.figure(figsize=(9, 8))

# Punti (Scatter)
plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.3, color='green', label='XGBoost Predictions')

# Linea Ideale
min_val = min(y_test[:, 0].min(), y_pred[:, 0].min())
max_val = max(y_test[:, 0].max(), y_pred[:, 0].max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal (Perfect)')

# Etichette in Inglese
plt.title('XGBoost: Actual vs. Predicted (Joint 1)', fontsize=14)
plt.xlabel('Actual Value (real dq)', fontsize=12)
plt.ylabel('Predicted Value (estimated dq)', fontsize=12)

# Riquadro dati
metrics_text = f"XGBoost Performance:\n\nMSE: {mse:.4e}\n$R^2$ Score: {r2:.4f}"
bbox_props = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9)
plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=bbox_props)

plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)

plt.savefig(IMAGE_NAME, dpi=300, bbox_inches='tight')
print(f"🖼️ Grafico salvato: {IMAGE_NAME}")