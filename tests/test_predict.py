import numpy as np
import sys
# import pytest
from predict import run_inference

def test_run_inference_valid_shape():
    # Assuming the updated app.py logic of 48 timesteps * 8 features = 384
    # Create a dummy input array of the exact shape expected by the model
    dummy_input = np.random.rand(1, 384) 
    
    # Run the function
    forecast = run_inference(dummy_input)
    
    # Assertions
    assert isinstance(forecast, np.ndarray), "Output should be a numpy array"
    assert forecast.shape == (1, 48), f"Expected shape (1, 48), got {forecast.shape}"
    
    # Ensure no NaN values were outputted
    assert not np.isnan(forecast).any(), "Forecast contains NaN values"