# src/data/preprocess.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import os

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

def check_missing_values(data):
    """Check for missing values in the dataframe."""
    missing = data.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"Missing values found:\n{missing[missing > 0]}")
    else:
        logger.info("No missing values found")
    return missing

def handle_outliers(data, columns, method='iqr', threshold=1.5):
    """Handle outliers in specified columns."""
    data_clean = data.copy()
    
    for col in columns:
        if method == 'iqr':
            Q1 = data_clean[col].quantile(0.25)
            Q3 = data_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Log outliers
            outliers = data_clean[(data_clean[col] < lower_bound) | (data_clean[col] > upper_bound)]
            if not outliers.empty:
                logger.info(f"Found {len(outliers)} outliers in column {col}")
            
            # Cap outliers
            data_clean[col] = np.where(
                data_clean[col] < lower_bound,
                lower_bound,
                np.where(data_clean[col] > upper_bound, upper_bound, data_clean[col])
            )
    
    return data_clean

def preprocess_data(data, target_col='default', test_size=0.2, random_state=42):
    """Preprocess data for modeling."""
    logger.info("Starting data preprocessing")
    
    # Check for missing values
    check_missing_values(data)
    
    # Extract features and target
    X = data.drop(columns=[target_col, 'customer_id'])  # Assuming customer_id is not needed
    y = data[target_col]
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Handle outliers in numerical columns
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    X = handle_outliers(X, numerical_cols)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Data preprocessing completed")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_processed_data(X_train, X_test, y_train, y_test, scaler, output_dir):
    """Save processed data to disk."""
    logger.info(f"Saving processed data to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrames for saving
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    
    # Save data
    X_train_df.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test_df.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    
    logger.info("All processed data saved successfully")

if __name__ == "__main__":
    # Example usage
    input_file = "data/raw/Loan_Data.csv"
    output_dir = "data/processed"
    
    data = load_data(input_file)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    save_processed_data(X_train, X_test, y_train, y_test, scaler, output_dir)