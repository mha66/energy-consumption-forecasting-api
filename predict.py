import mlflow
import numpy as np

# 1. Connect to the local MLflow registry
# mlflow.set_tracking_uri("sqlite:///mlflow.db")

# # 2. Define the URI for the specific model you want to deploy
# # Replace this string with the actual Run ID you copied from the MLflow UI
# RUN_ID = "f8adfe37a65347aab9a4656a9342c617"
# model_uri = f"runs:/{RUN_ID}/model"

# # 3. Load the model into memory
# # We use mlflow.sklearn because you logged it using mlflow.sklearn.log_model
# print(f"Loading model from run: {RUN_ID}...")
# model = mlflow.sklearn.load_model(model_uri)
# print("✅ Model loaded successfully.")

MODEL_PATH = "deployed_model"

print(f"Loading model from local directory: {MODEL_PATH}...")
model = mlflow.sklearn.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

def run_inference(flattened_input_window: np.ndarray) -> np.ndarray:
    """
    Runs a prediction on new, incoming data.
    
    Expected Input: 
    A 2D numpy array of shape (Samples, TimeSteps * Features).
    For a single API request of 48 timesteps with 8 features, 
    the shape must be (1, 384).
    
    Returns:
    A 2D numpy array containing the forecasted output sequence.
    """
    # The loaded model pipeline automatically handles the StandardScaler 
    # transformations for both the input features and the target output.
    forecast = model.predict(flattened_input_window)
    
    return forecast

# --- Quick Local Test ---
if __name__ == "__main__":
    # Create a dummy array of the exact shape the model expects (1 sample, 528 flattened features)
    # This verifies the model is loaded and mathematically functional before you add FastAPI
    dummy_input = np.random.rand(1, 48 * 8) 
    
    print("\nRunning test inference...")
    dummy_forecast = run_inference(dummy_input)
    print("Test Forecast Shape:", dummy_forecast.shape)
    print("Test Forecast Values (First 5):", dummy_forecast[0][:5])