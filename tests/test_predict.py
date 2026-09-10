
# import pytest
# from unittest.mock import patch

import numpy as np

from predict import run_inference


# This decorator replaces 'model' inside 'predict.py' with a Mock object
# @patch('predict.model')
def test_run_inference_valid_shape(mock_model):
    # 1. Configure the fake model to return a dummy prediction shape
    mock_model.predict.return_value = np.random.rand(1, 48)
    
    # 2. Create the dummy input (384 features)
    dummy_input = np.random.rand(1, 384) 
    
    # 3. Run the function
    forecast = run_inference(dummy_input)
    
    # 4. Assertions
    assert isinstance(forecast, np.ndarray), "Output should be a numpy array"
    assert forecast.shape == (1, 48), f"Expected shape (1, 48), got {forecast.shape}"
    
    # Verify that the model.predict function was actually called with our data
    mock_model.predict.assert_called_once_with(dummy_input)