# src/data/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(filepath):
    """Load data from CSV file."""
    logger.info(f"Loading data from {filepath}")
    try:
        data = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully with shape {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def explore_data(data, output_dir):
    """Perform exploratory data analysis and save visualizations."""
    logger.info("Starting exploratory data analysis")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic statistics
    logger.info("Generating basic statistics")
    stats = data.describe().T
    stats.to_csv(f"{output_dir}/basic_statistics.csv")
    
    # Check target distribution
    logger.info("Analyzing target distribution")
    plt.figure(figsize=(8, 6))
    default_counts = data['default'].value_counts()
    sns.barplot(x=default_counts.index, y=default_counts.values)
    plt.title('Distribution of Default vs Non-Default')
    plt.xlabel('Default Status (1=Default, 0=Non-Default)')
    plt.ylabel('Count')
    plt.savefig(f"{output_dir}/target_distribution.png")
    
    # Correlation matrix
    logger.info("Generating correlation matrix")
    plt.figure(figsize=(12, 10))
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    corr = numeric_data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    
    # Feature distributions by default status
    logger.info("Analyzing feature distributions by default status")
    feature_cols = [col for col in data.columns if col not in ['customer_id', 'default']]
    
    for col in feature_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x=col, hue='default', kde=True, element='step')
        plt.title(f'Distribution of {col} by Default Status')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dist_{col}.png")
    
    # Box plots for outlier detection
    logger.info("Creating box plots for outlier detection")
    for col in feature_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data, x='default', y=col)
        plt.title(f'Box Plot of {col} by Default Status')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/boxplot_{col}.png")
    
    # Generate summary report
    logger.info("Generating data summary report")
    with open(f"{output_dir}/eda_summary.txt", 'w') as f:
        f.write(f"Dataset Shape: {data.shape}\n\n")
        f.write(f"Data Types:\n{data.dtypes}\n\n")
        f.write(f"Missing Values:\n{data.isnull().sum()}\n\n")
        f.write(f"Target Distribution:\n{data['default'].value_counts()}\n")
        f.write(f"Target Distribution (%):\n{data['default'].value_counts(normalize=True) * 100}\n\n")
        
    logger.info("EDA completed successfully")

if __name__ == "__main__":
    input_file = "data/raw/Loan_Data.csv"
    output_dir = "notebooks/eda_results"
    
    data = load_data(input_file)
    explore_data(data, output_dir)