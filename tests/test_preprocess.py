# tests/test_preprocess.py
import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(_file_), '..')))

from src.data.preprocess import check_missing_values, handle_outliers, preprocess_data

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'credit_lines_outstanding': [0, 5, 0, 0, 1, 2, 3, 4, 5, 6],
            'loan_amt_outstanding': [5000, 2000, 3000, 4800, 1300, 2500, 3000, 4000, 5000, 6000],
            'total_debt_outstanding': [4000, 8000, 2000, 2500, 1800, 3000, 4000, 5000, 6000, 7000],
            'income': [78000, 27000, 66000, 74000, 23000, 30000, 40000, 50000, 60000, 70000],
            'years_employed': [5, 2, 4, 5, 6, 3, 4, 5, 6, 7],
            'fico_score': [605, 572, 602, 612, 631, 620, 640, 650, 660, 670],
            'default': [0, 1, 0, 0, 0, 1, 0, 1, 0, 1]  # More samples for class 1
        })

    
    def test_check_missing_values(self):
        """Test missing values check function."""
        # Original data has no missing values
        missing = check_missing_values(self.data)
        self.assertEqual(missing.sum(), 0)
        
        # Add a missing value
        data_with_missing = self.data.copy()
        data_with_missing.loc[0, 'fico_score'] = np.nan
        missing = check_missing_values(data_with_missing)
        self.assertEqual(missing['fico_score'], 1)
    
    def test_handle_outliers(self):
        """Test outlier handling function."""
        # Create data with outliers
        data_with_outliers = self.data.copy()
        data_with_outliers.loc[0, 'income'] = 1000000  # Outlier
        
        # Handle outliers
        clean_data = handle_outliers(data_with_outliers, ['income'], method='iqr')
        
        # Check that outlier was capped
        self.assertLess(clean_data['income'].max(), 1000000)
    
    def test_preprocess_data(self):
        """Test the full preprocessing pipeline."""
        X_train, X_test, y_train, y_test, scaler = preprocess_data(self.data, test_size=0.3)
        # Check shapes
        self.assertEqual(X_train.shape[1], X_test.shape[1])  # Same number of features
        self.assertEqual(len(y_train) + len(y_test), len(self.data))  # All samples accounted for