from unittest.mock import patch

# import pytest
import numpy as np
from fastapi.testclient import TestClient

from app import app

# Initialize the TestClient with your FastAPI app
client = TestClient(app)

def test_health_check():
    """Test the root endpoint to ensure the API is responsive."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "online", "message": "Energy Consumption Forecasting API is running. Visit /docs for the interactive UI."}

@patch('predict.model')
def test_predict_energy_valid_request(mock_model):
    """Test a perfectly formatted request."""
    # Configure the mock model to return a dummy prediction
    mock_model.predict.return_value = np.random.rand(1, 48)
    
    # 48 timesteps, each with 8 features
    valid_features = [[2**-0.5] * 8 for _ in range(48)]
    
    response = client.post("/predict", json={"features": valid_features})
    
    assert response.status_code == 200
    data = response.json()
    assert "forecast" in data
    assert len(data["forecast"]) == 48
    assert isinstance(data["forecast"][0], float)

@patch('predict.model')
def test_predict_energy_invalid_timestep_count(mock_model):
    """Test the strict validation for exactly 48 timesteps."""
    # Configure the mock model to return a dummy prediction
    mock_model.predict.return_value = np.random.rand(1, 48)
    
    # Only 47 timesteps provided
    invalid_features = [[2**-0.5] * 8 for _ in range(47)]
    
    response = client.post("/predict", json={"features": invalid_features})
    
    assert response.status_code in [400, 422]  # Depending on FastAPI's validation, it could be either
    assert "Expected 48 timesteps" in response.json()["detail"][0]["msg"]

@patch('predict.model')
def test_predict_energy_invalid_feature_count(mock_model):
    """Test the strict validation for exactly 8 features per timestep."""
    # Configure the mock model to return a dummy prediction
    mock_model.predict.return_value = np.random.rand(1, 48)
    
    # 48 timesteps, but only 7 features per timestep
    invalid_features = [[2**-0.5] * 7 for _ in range(48)]
    
    response = client.post("/predict", json={"features": invalid_features})
    
    assert response.status_code in [400, 422]  # Depending on FastAPI's validation, it could be either
    assert "must have exactly 8 features" in response.json()["detail"][0]["msg"]