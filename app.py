from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List
import numpy as np
import math

# Import your pre-loaded model function from predict.py
from predict import run_inference

app = FastAPI(
    title="Energy Consumption Forecasting API",
    description="MLOps API for forecasting 24-hour baseload energy consumption.",
    version="1.0.0"
)

# --- 1. Define the Data Contracts (DTOs) ---

class ForecastRequestDTO(BaseModel):
    features: List[List[float]] = Field(
        ..., 
        description="A 2D array of 48 timesteps. Each timestep must contain exactly 8 features: [Baseload_kW, consumption_change, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos]."
    )

    @field_validator('features')
    @classmethod
    def validate_features(cls, features: List[List[float]]) -> List[List[float]]:
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
    # # 1. Strict Validation: Reject bad payloads immediately
    # if len(request.features) != 48:
    #     raise HTTPException(status_code=400, detail=f"Expected 48 timesteps, received {len(request.features)}.")
    
    # for i, timestep in enumerate(request.features):
    #     if len(timestep) != 8:
    #         raise HTTPException(status_code=400, detail=f"Timestep {i} must have exactly 8 features.")
            
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