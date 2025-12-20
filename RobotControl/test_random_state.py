from sklearn.model_selection import train_test_split
import numpy as np

# Creiamo dati finti: numeri da 0 a 9
X = np.arange(10)
y = np.arange(10)

print("--- SENZA RANDOM STATE (Cambia ogni volta) ---")
# Se lo rilanci, questi numeri cambieranno
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2) 
print(f"X_test estratto: {X_test}")

print("\n--- CON RANDOM STATE=42 (Sempre uguale) ---")
# Se lo rilanci, questi numeri saranno SEMPRE [8, 1]
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_test estratto: {X_test}")

