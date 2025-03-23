# src/mlflow_server.py
import os
import mlflow
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_mlflow_server():
    """Start MLflow tracking server with SQLite backend."""
    # Create directory for MLflow data
    os.makedirs("mlflow_runs", exist_ok=True)
    
    # Set tracking URI
    tracking_uri = "mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    
    logger.info(f"MLflow tracking URI set to: {tracking_uri}")
    logger.info("To view the MLflow UI, run: mlflow ui --port 5000")

if __name__ == "__main__":
    start_mlflow_server()
