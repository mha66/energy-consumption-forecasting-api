import logging
import math
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from predict import run_inference

app = FastAPI(
    title="Energy Consumption Forecasting API",
    description="MLOps API for forecasting 24-hour baseload energy consumption.",
    version="1.0.0"
)


# Configure the standard Python logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mlops_api")

@app.middleware("http")
async def log_system_metrics(request: Request, call_next):
    """
    SYSTEM OBSERVABILITY:
    Tracks pure software performance metrics regardless of the payload payload.
    """
    start_time = time.time()
    
    # Let the request pass through to the router
    response = await call_next(request)
    
    # Calculate latency
    process_time = time.time() - start_time
    
    # Log the system metrics
    logger.info(
        f"SYSTEM | Method: {request.method} | Path: {request.url.path} | "
        f"Status: {response.status_code} | Latency: {process_time:.4f}s"
    )
    
    return response

# Health Endpoint
@app.get("/health")
async def health_check():
    """Used by Docker/Load Balancers to verify the API is alive."""
    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "model_status": "loaded" # In a real app, you might verify the MLflow model is in memory
    }


# --- 1. Define the Data Contracts (DTOs) ---

class ForecastRequestDTO(BaseModel):
    features: list[list[float]] = Field(
        ..., 
        description="A 2D array of 48 timesteps. Each timestep must contain exactly 8 features: [Baseload_kW, consumption_change, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos]."
    )

    @field_validator('features')
    @classmethod
    def validate_features(cls, features: list[list[float]]) -> list[list[float]]:
        # 1. Validate total sequence length
        if len(features) != 48:
            raise ValueError(f"Expected 48 timesteps, but received {len(features)}.")
            
        for i, timestep in enumerate(features):
            # 2. Validate feature count per timestep
            if len(timestep) != 8:
                raise ValueError(f"Timestep {i} must have exactly 8 features.")
                
            baseload_kw = timestep[0]
            hour_sin, hour_cos = timestep[2], timestep[3]
            day_sin, day_cos = timestep[4], timestep[5]
            month_sin, month_cos = timestep[6], timestep[7]

            # 3. Physical Boundary Validation
            # Baseload clipped to 0 in data_prep.py, so it should never be negative
            if baseload_kw < 0:
                raise ValueError(f"Timestep {i}: Baseload_kW ({baseload_kw}) cannot be negative.")
                
            # 4. Cyclical Range Validation (-1.0 to 1.0)
            for val, name in zip(
                [hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos], 
                ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
            ):
                if not -1.0 <= val <= 1.0:
                    raise ValueError(f"Timestep {i}: {name} ({val}) must be between -1.0 and 1.0.")

            # 5. Unit Circle Geometric Validation
            # Ensure the sine and cosine pairs map to a valid point on the unit circle
            # using the Pythagorean identity
            if not math.isclose(hour_sin**2 + hour_cos**2, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                raise ValueError(f"Timestep {i}: hour_sin and hour_cos do not form a valid unit circle.")
                
        return features

class ForecastResponseDTO(BaseModel):
    # The output will be a list of 48 forecasted floats representing the next 24 hours
    forecast: list[float]


# --- 2. Define the Controller / Endpoint ---

@app.get("/")
async def root():
    return {"status": "online", "message": "Energy Consumption Forecasting API is running. Visit /docs for the interactive UI."}

@app.post("/predict", response_model=ForecastResponseDTO)
async def predict_energy(request: ForecastRequestDTO):
    # 1. Data Transformation
    features_array = np.array(request.features)
    
    # Extract ML Observability Metrics BEFORE flattening
    # Assuming Baseload_kW is at index 0 based on your DTO description
    baseload_mean = float(np.mean(features_array[:, 0]))
    baseload_max = float(np.max(features_array[:, 0]))
    
    # Flatten for the model
    input_array = features_array.flatten().reshape(1, -1)
    
    # 2. Model Inference
    try:
        prediction = run_inference(input_array)
        forecast_list = prediction[0].tolist()
        
        # Extract ML Observability Metrics from the OUTPUT
        forecast_mean = float(np.mean(forecast_list))
        forecast_max = float(np.max(forecast_list))
        # 3. Log the ML Metrics
        logger.info(
            f"ML_OPS | Inference Success | "
            f"Input Baseload (Mean: {baseload_mean:.2f}kW, Max: {baseload_max:.2f}kW) | "
            f"Forecast Mean: {forecast_mean:.2f}kW, Max: {forecast_max:.2f}kW"
        )
        
        return ForecastResponseDTO(forecast=forecast_list)
        
    except Exception:
        logger.exception("SYSTEM_ERROR | Model inference failed:")
        raise HTTPException(status_code=500, detail="Internal inference error.")

# --- 3. Server Startup ---
if __name__ == "__main__":
    import uvicorn
    # This runs the API on localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)