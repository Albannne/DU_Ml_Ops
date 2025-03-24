# src/models/evaluate_models.py
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import json
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_run_data():
    """Load experiment data from MLflow."""
    client = mlflow.tracking.MlflowClient()
    
    # Define our experiments
    experiment_names = ["loan_default_prediction_logistic_regression", "loan_default_prediction_random_forest"]
    experiment_data = []
    
    for experiment_name in experiment_names:
        try:
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment:
                experiment_id = experiment.experiment_id
                runs = client.search_runs(experiment_ids=[experiment_id])
                
                for run in runs:
                    run_data = {
                        'experiment': experiment_name,
                        'run_id': run.info.run_id,
                        'start_time': run.info.start_time,
                    }
                    
                    # Add metrics
                    metrics = run.data.metrics
                    for key, value in metrics.items():
                        run_data[key] = value
                    
                    # Add parameters
                    params = run.data.params
                    for key, value in params.items():
                        run_data[f"param_{key}"] = value
                    
                    experiment_data.append(run_data)
        except Exception as e:
            logger.error(f"Error loading experiment {experiment_name}: {e}")
    
    return pd.DataFrame(experiment_data)

def compare_models(runs_df, output_dir):
    """Compare model performance and visualize results."""
    os.makedirs(output_dir, exist_ok=True)
    
    if runs_df.empty:
        logger.warning("No run data available to compare")
        return None
    
    # Group by experiment and get best run (by ROC AUC) for each
    best_runs = runs_df.groupby('experiment').apply(
        lambda x: x.loc[x['roc_auc'].idxmax()]
    ).reset_index(drop=True)
    
    # Create comparison visualizations
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    # Bar chart comparison
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    for i, (_, row) in enumerate(best_runs.iterrows()):
        plt.bar(x + width * (i - 0.5), [row[m] for m in metrics], width, label=row['experiment'].replace('loan_default_prediction_', ''))
    
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png")
    
    # Determine best model
    best_model = best_runs.loc[best_runs['roc_auc'].idxmax()]
    
    # Create comparison table
    comparison_table = best_runs[['experiment'] + metrics].copy()
    comparison_table['experiment'] = comparison_table['experiment'].apply(lambda x: x.replace('loan_default_prediction_', ''))
    comparison_table.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    
    # Save results
    best_model_info = {
        'experiment': best_model['experiment'].replace('loan_default_prediction_', ''),
        'run_id': best_model['run_id'],
        'metrics': {metric: best_model[metric] for metric in metrics}
    }
    
    with open(f"{output_dir}/best_model_info.json", 'w') as f:
        json.dump(best_model_info, f, indent=4)
    
    logger.info(f"Best model: {best_model_info['experiment']} with ROC AUC: {best_model_info['metrics']['roc_auc']}")
    
    return best_model_info

if __name__ == "__main__":
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("mlruns")
    
    # Load run data
    runs_df = load_run_data()
    
    # Compare models
    output_dir = "models/evaluation"
    best_model_info = compare_models(runs_df, output_dir)
    
    # Log best model info
    if best_model_info:
        model_name = best_model_info['experiment']
        # Write to the best_model.txt file needed by the streamlit app
        with open("models/best_model.txt", 'w') as f:
            f.write(model_name)