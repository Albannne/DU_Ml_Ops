FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier uniquement les fichiers et dossiers nécessaires
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/

# Variables d'environnement pour MLflow
ENV MLFLOW_TRACKING_URI=mlruns

# Créer un utilisateur non-root pour plus de sécurité
RUN useradd -m appuser
USER appuser

EXPOSE 8501

# Exécuter avec le chemin correct
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]