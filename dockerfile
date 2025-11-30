# Usa un'immagine Python leggera
FROM python:3.9-slim

# Imposta la cartella di lavoro
WORKDIR /app

# Installa dipendenze di sistema e librerie grafiche (necessarie per alcuni plot)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Installa le librerie Python (incluse quelle nuove)
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    tensorflow \
    matplotlib \
    jupyterlab \
    xgboost

# Espone la porta e avvia
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
