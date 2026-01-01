import matplotlib.pyplot as plt
import numpy as np

# Dati linearmente separabili (Due classi: +1 e -1)
# Classe +1 (Cerchi blu, alto a destra)
X_pos = np.array([[3, 4], [4, 3], [3.5, 3.5]])
# Classe -1 (Croci rosse, basso a sinistra)
X_neg = np.array([[1, 1], [2, 1], [1, 2]])

# Uniamo i dati
X = np.vstack((X_pos, X_neg))
y = np.array([1, 1, 1, -1, -1, -1])

# Funzione per disegnare lo stato corrente
def plot_perceptron_step(ax, weights, step_title):
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(step_title)
    
    # Disegna i dati
    ax.scatter(X_pos[:, 0], X_pos[:, 1], color='blue', marker='o', s=100, label='Class +1')
    ax.scatter(X_neg[:, 0], X_neg[:, 1], color='red', marker='x', s=100, label='Class -1')
    
    # Disegna l'iperpiano (retta) ortogonale ai pesi w
    # w0 + w1*x1 + w2*x2 = 0  => x2 = -(w1/w2)*x1 - (w0/w2)
    # Per semplicità visiva, assumiamo che la retta passi per l'origine o usiamo w bias
    w = weights
    x1_vals = np.linspace(-1, 6, 100)
    
    if w[1] != 0:
        x2_vals = -(w[0] * x1_vals) / w[1] # Semplificato senza bias esplicito per il grafico vettoriale
        ax.plot(x1_vals, x2_vals, 'k-', linewidth=2, label='Decision Boundary')
        
        # Disegna il vettore dei pesi w (freccia)
        # Parte dall'origine (o centro) e punta nella direzione positiva
        ax.arrow(2.5, 2.5, w[0]*0.5, w[1]*0.5, head_width=0.2, head_length=0.2, fc='green', ec='green', label='w vector')
    
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.6)

# --- Creazione dei 4 Grafici ---
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
plt.subplots_adjust(hspace=0.3)

# 1. Inizializzazione (Pesi casuali errati)
w_step1 = np.array([-1.0, 1.5]) # Punta in alto a sinistra (sbagliato)
plot_perceptron_step(axs[0, 0], w_step1, "Step 1: Inizializzazione (Casuale)")

# 2. Primo Aggiornamento (Rotazione verso la classe corretta)
# Immaginiamo di aver sbagliato un punto positivo, il vettore ruota verso destra
w_step2 = np.array([0.5, 1.5]) 
plot_perceptron_step(axs[0, 1], w_step2, "Step 2: Primo Aggiornamento")

# 3. Aggiornamento Intermedio (Aggiustamento fine)
# La retta separa quasi tutto ma tocca ancora un punto rosso
w_step3 = np.array([1.5, 1.0]) 
plot_perceptron_step(axs[1, 0], w_step3, "Step 3: Evoluzione Intermedia")

# 4. Soluzione Finale (Convergenza)
# Vettore punta perfettamente verso i blu (alto-destra), retta nel mezzo
w_step4 = np.array([1.0, 1.0]) 
plot_perceptron_step(axs[1, 1], w_step4, "Step 4: Soluzione Finale")

plt.show()