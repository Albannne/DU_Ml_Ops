# tests/test_models.py
import unittest
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.train_model import evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class TestModels(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        # Create synthetic data
        np.random.seed(42)
        X = np.random.normal(size=(100, 6))
        y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale data
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
    
    def test_logistic_regression(self):
        """Test logistic regression model evaluation."""
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(self.X_train, self.y_train)
        
        # Evaluate model
        metrics, cm = evaluate_model(model, self.X_test, self.y_test)
        
        # Check that metrics exist
        self.assertIn('accuracy', metrics)
        self.assertIn('roc_auc', metrics)
    
    def test_random_forest(self):
        """Test random forest model evaluation."""
        # Train model
        model = RandomForestClassifier(random_state=42, n_estimators=10)
        model.fit(self.X_train, self.y_train)
        
        # Evaluate model
        metrics, cm = evaluate_model(model, self.X_test, self.y_test)
        
        # Check that metrics exist
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)