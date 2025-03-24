# src/models/train_model.py

import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_processed_data(data_dir):
    """Load processed data for modeling."""
    logger.info(f"Loading processed data from {data_dir}")
    
    X_train = pd.read_csv(f"{data_dir}/X_train.csv").values
    X_test = pd.read_csv(f"{data_dir}/X_test.csv").values
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").values.ravel()
    
    logger.info(f"Data loaded successfully: X_train={X_train.shape}, y_train={y_train.shape}")
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test data."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    logger.info(f"Model evaluation results: {metrics}")
    
    return metrics, cm

def save_model(model, metrics, output_dir, model_name):
    """Save trained model and metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = f"{output_dir}/{model_name}.pkl"
    joblib.dump(model, model_path)
    
    # Save metrics
    metrics_path = f"{output_dir}/{model_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metrics saved to {metrics_path}")
    
    return model_path, metrics_path

def train_logistic_regression(X_train, y_train, X_test, y_test, model_dir):
    """Train a logistic regression model with MLflow tracking."""
    logger.info("Training Logistic Regression model")
    
    # Set MLflow experiment
    experiment_name = "loan_default_prediction_logistic_regression"
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"logistic_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Define hyperparameters
        params = {
            'C': 1.0,
            'solver': 'liblinear',
            'random_state': 42,
            'max_iter': 1000,
            'class_weight': 'balanced'
        }
        
        # Log parameters
        for param, value in params.items():
            mlflow.log_param(param, value)
        
        # Train model
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # Evaluate model
        metrics, cm = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log confusion matrix as a figure
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Logistic Regression')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save and log confusion matrix figure
        cm_path = f"{model_dir}/logistic_regression_cm.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        
        # Log importance of features
        feature_importance = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(X_train.shape[1])],
            'importance': np.abs(model.coef_[0])
        }).sort_values('importance', ascending=False)
        
        feature_imp_path = f"{model_dir}/logistic_regression_feature_importance.csv"
        feature_importance.to_csv(feature_imp_path, index=False)
        mlflow.log_artifact(feature_imp_path)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        model_path, metrics_path = save_model(model, metrics, model_dir, "logistic_regression")
        
        return model, metrics, model_path

def train_random_forest(X_train, y_train, X_test, y_test, model_dir):
    """Train a random forest model with MLflow tracking."""
    logger.info("Training Random Forest model")
    
    # Set MLflow experiment
    experiment_name = "loan_default_prediction_random_forest"
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"random_forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Define hyperparameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        
        # Log parameters
        for param, value in params.items():
            mlflow.log_param(param, value)
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluate model
        metrics, cm = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log confusion matrix as a figure
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Random Forest')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save and log confusion matrix figure
        cm_path = f"{model_dir}/random_forest_cm.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        
        # Log feature importances
        feature_importance = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(X_train.shape[1])],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_imp_path = f"{model_dir}/random_forest_feature_importance.csv"
        feature_importance.to_csv(feature_imp_path, index=False)
        mlflow.log_artifact(feature_imp_path)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        model_path, metrics_path = save_model(model, metrics, model_dir, "random_forest")
        
        return model, metrics, model_path

if __name__ == "__main__":
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("mlruns")
    
    # Load data
    data_dir = "data/processed"
    model_dir = "models"
    X_train, X_test, y_train, y_test = load_processed_data(data_dir)
    
    # Train models
    lr_model, lr_metrics, lr_path = train_logistic_regression(X_train, y_train, X_test, y_test, model_dir)
    rf_model, rf_metrics, rf_path = train_random_forest(X_train, y_train, X_test, y_test, model_dir)
    
    # Compare models
    logger.info("\nModel Comparison:")
    logger.info(f"Logistic Regression Metrics: {lr_metrics}")
    logger.info(f"Random Forest Metrics: {rf_metrics}")
    
    # Determine best model based on ROC AUC
    if lr_metrics['roc_auc'] > rf_metrics['roc_auc']:
        best_model = "logistic_regression"
        logger.info("Best model: Logistic Regression")
    else:
        best_model = "random_forest"
        logger.info("Best model: Random Forest")
    
    # Save best model info
    with open(f"{model_dir}/best_model.txt", 'w') as f:
        f.write(best_model)