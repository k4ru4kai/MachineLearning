import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# --- CONFIGURAZIONE ---
FILE_NAME = 'reacher3_train_1.csv'

# 1. Preparazione Dati (Identica a prima)
df = pd.read_csv(FILE_NAME)
X = df.iloc[:, 0:7].values
y = df.iloc[:, 7:10].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)

# 2. Addestramento Veloce
model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)

# 3. Predizione
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled) # Torniamo ai valori veri

# 4. IL GRAFICO (Visualizziamo solo il primo motore - Giunto 1)
plt.figure(figsize=(8, 8))

# Disegniamo i punti: X=Valore Vero, Y=Valore Predetto
# Alpha=0.5 li rende semitrasparenti così vedi dove si sovrappongono
plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.3, color='blue', label='Predizioni')

# Disegniamo la "Linea della Perfezione" (Diagonale Rossa)
min_val = min(y_test[:, 0].min(), y_pred[:, 0].min())
max_val = max(y_test[:, 0].max(), y_pred[:, 0].max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideale (Perfetto)')

plt.title('Linear Regression: Realtà vs Predizione (Giunto 1)')
plt.xlabel('Valore Vero (dq reale)')
plt.ylabel('Valore Predetto (dq stimato)')
plt.legend()
plt.grid(True)

# Salviamo l'immagine
plt.savefig('linear_regression_results.png')
print("🖼️ Grafico salvato come 'linear_regression_results.png'")