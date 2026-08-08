from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np

# Import your pre-loaded model function from predict.py
from predict import run_inference

app = FastAPI(
    title="Energy Consumption Forecasting API",
    description="MLOps API for forecasting 24-hour baseload energy consumption.",
    version="1.0.0"
)

# --- 1. Define the Data Contracts (DTOs) ---

class ForecastRequestDTO(BaseModel):
    # Based on your data_prep.py, the model expects 48 timesteps.
    # After dropping datetime, Aggregate_kW, and HVAC_kW, you have exactly 8 features left.
    features: List[List[float]] = Field(
        ..., 
        description="A 2D array of 48 timesteps. Each timestep must contain exactly 8 features: [Baseload_kW, consumption_change, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos]."
    )

class ForecastResponseDTO(BaseModel):
    # The output will be a list of 48 forecasted floats representing the next 24 hours
    forecast: List[float]


# --- 2. Define the Controller / Endpoint ---

@app.get("/")
async def root():
    return {"status": "online", "message": "Energy Consumption Forecasting API is running. Visit /docs for the interactive UI."}

@app.post("/predict", response_model=ForecastResponseDTO)
async def predict_energy(request: ForecastRequestDTO):
    """
    Accepts 48 timesteps of historical energy data and returns a 48-timestep forecast.
    """
    # 1. Strict Validation: Reject bad payloads immediately
    if len(request.features) != 48:
        raise HTTPException(status_code=400, detail=f"Expected 48 timesteps, received {len(request.features)}.")
    
    for i, timestep in enumerate(request.features):
        if len(timestep) != 8:
            raise HTTPException(status_code=400, detail=f"Timestep {i} must have exactly 8 features.")
            
    # 2. Data Transformation
    # Convert the 2D list into a numpy array, then flatten it to shape (1, 384)
    # 48 timesteps * 8 features = 384 flattened features expected by XGBoost
    input_array = np.array(request.features).flatten().reshape(1, -1)
    
    # 3. Model Inference
    try:
        prediction = run_inference(input_array)
        # prediction is a 2D numpy array: [[val1, val2, ... val48]]. Extract the inner list.
        return ForecastResponseDTO(forecast=prediction[0].tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

# --- 3. Server Startup ---
if __name__ == "__main__":
    import uvicorn
    # This runs the API on localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)