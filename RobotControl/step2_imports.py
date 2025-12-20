# 1. PANDAS: Per leggere i file CSV 
import pandas as pd

# 2. NUMPY: Per calcoli matematici e matrici
import numpy as np

# 3. TENSORFLOW: Per costruire la Rete Neurale 
import tensorflow as tf

# 4. SKLEARN (Scikit-Learn): Per preprocessing e metriche
# Non si importa tutto 'sklearn', ma solo i pezzi che servono:
from sklearn.model_selection import train_test_split  # Per dividere Train/Test 
from sklearn.preprocessing import StandardScaler      # Per normalizzare i dati [cite: 320]
from sklearn.metrics import r2_score                  # La metrica di valutazione 

# 5. MATPLOTLIB: Per i grafici del report 
import matplotlib.pyplot as plt

print("-" * 40)
print("✅ CONTROLLO LIBRERIE:")
print(f"Pandas versione: {pd.__version__}")
print(f"NumPy versione: {np.__version__}")
print(f"TensorFlow versione: {tf.__version__}")
print("Tutte le librerie sono importate correttamente!")
print("-" * 40)