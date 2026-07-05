import os
import sys
import pytest
import warnings

# Suppress all FutureWarnings for the test environment
warnings.filterwarnings("ignore", category=FutureWarning)

# Add the project root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "app"))

from utils.model_loader import load_all, predict

def test_load_all():
    """Test if models, thresholds, scaler, and test_data can be loaded."""
    models, thresholds, scaler, test_data = load_all()
    
    # Check if models dictionary is populated
    assert isinstance(models, dict), "Models should be a dictionary"
    assert len(models) > 0, "No models were loaded"
    
    # Check if common models exist
    assert "RandomForest" in models, "RandomForest model missing"
    
    # Check if scaler is loaded
    assert scaler is not None, "Scaler is missing"
    
    # Check if test_data is a dict
    assert test_data is not None, "Test data is missing"
    assert isinstance(test_data, dict), "Test data should be a dictionary"
    assert "X_test" in test_data, "X_test not found in test_data"

def test_predict_function():
    """Test the predict wrapper function for a dummy sample."""
    models, thresholds, scaler, test_data = load_all()
    
    rf_model = models.get("RandomForest")
    if not rf_model:
        pytest.skip("RandomForest model not available for prediction test")
        
    X_test = test_data["X_test"]
    sample_X = X_test[:5]
    
    preds, scores = predict(rf_model, "RandomForest", sample_X, thresholds)
    
    assert len(preds) == 5, "Predictions count mismatch"
    assert len(scores) == 5, "Scores count mismatch"
