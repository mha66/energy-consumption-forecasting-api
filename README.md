# Energy Consumption Forecasting API

An MLOps API for forecasting 24-hour baseload energy consumption. This project serves a machine learning model using FastAPI, complete with strict data contract validation, observability logging, and Docker containerization.

## Features

- **FastAPI Endpoint**: Real-time model inference through a robust `/predict` endpoint that processes 24 hours of data.
- **Data Validation**: Strict input validation using Pydantic, enforcing expected array shapes, physical boundaries (e.g., non-negative baseloads), and mathematical constraints (unit circle bounds for cyclical time features).
- **Machine Learning Architecture**: Multi-step time-series forecasting utilizing XGBoost wrapped in a `MultiOutputRegressor`, predicting 48 future timesteps simultaneously (24 hours at 30-minute intervals).
- **Advanced Feature Engineering**: Isolates pure baseload power consumption by subtracting HVAC/furnace power from the aggregate usage, and automatically generates cyclic unit-circle time features (Sine/Cosine mappings for hour, day, and month).
- **End-to-End Scikit-Learn Pipelines**: Safely integrates `StandardScaler` for inputs and a `TransformedTargetRegressor` for outputs directly into the model pipeline, preventing data leakage and simplifying production inference logic.
- **Experiment Tracking & MLOps**: Local tracking configured with MLflow for hyperparameter logging, model artifact versioning, and offline evaluation metrics tracking (RMSE, MAE, R²).
- **ML & System Observability**: Standardized API logging for system latency, endpoint health, and ML-specific metrics (monitoring input feature distributions and forecast output boundaries).
- **DVC Integration**: Production model weights (`deployed_model`) and datasets are strictly version-controlled using DVC (Data Version Control).
- **Dockerized Deployment**: Ready for production environments with a lightweight, optimized Python 3.11 Dockerfile.

## Quick Start

### Local Deployment (Python)

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/energy-consumption-forecasting-api.git
   cd energy-consumption-forecasting-api
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   
   pip install -r requirements.txt
   ```

3. Run the API locally:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
   The API will be available at `http://localhost:8000`. You can visit `http://localhost:8000/docs` to test the `/predict` endpoint via the interactive Swagger UI.

### Docker Deployment

To deploy this API on your machine using the pre-built Docker image, run the following command:

```bash
docker run -d -p 8000:8000 ghcr.io/mha66/energy-consumption-forecasting-api:latest
```

If you prefer to build the image locally:

```bash
docker build -t energy-consumption-forecasting-api .
docker run -d -p 8000:8000 energy-consumption-forecasting-api
```

## API Usage Example

**POST `/predict`**

Expects a 2D array of exactly 48 timesteps, where each timestep contains 8 features: `[Baseload_kW, consumption_change, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos]`.

Example payload:
```json
{
  "features": [
    [10.5, 0.1, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    ... // 47 more timesteps
  ]
}
```

Response:
```json
{
  "forecast": [11.2, 11.5, ... ] // 48 predictions
}
```
