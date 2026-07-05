import os
import sys
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data_loader import TelemetryDataLoader

def test_loader_initialization():
    loader = TelemetryDataLoader("dummy/path")
    assert loader.data_path == "dummy/path"
    assert loader.data is None

def test_get_summary_error():
    loader = TelemetryDataLoader()
    with pytest.raises(ValueError):
        loader.get_summary()

def test_validate_data_empty():
    loader = TelemetryDataLoader()
    loader.data = pd.DataFrame()
    
    assert loader.validate_data() == False, "Empty dataframe should not be valid"
    
def test_validate_data_missing_cols():
    loader = TelemetryDataLoader()
    loader.data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    
    assert loader.validate_data() == True
    
    assert loader.validate_data(["A", "C"]) == False
