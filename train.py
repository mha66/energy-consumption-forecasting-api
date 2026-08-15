import os
import time

import mlflow
import pandas as pd
import xgboost as xgb
from mlflow.models import infer_signature
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_prep import data_preparation_ampds2

RAW_DATA_DIR = os.path.join('data', 'raw', 'ampds2')
PROCESSED_DATA_DIR = os.path.join('data', 'processed', 'ampds2', '30min')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
WHE_DIR = os.path.join(RAW_DATA_DIR, 'Electricity_WHE.csv') # Whole House Energy
FRE_DIR = os.path.join(RAW_DATA_DIR, 'Electricity_FRE.csv') # Furnace / HVAC
HPE_DIR = os.path.join(RAW_DATA_DIR, 'Electricity_HPE.csv') # Heat Pump Energy

TARGET_COL = 'Baseload_kW'
IN_SEQ_LEN = 48  # Input sequence length (48 timesteps = 24 hours)
OUT_SEQ_LEN = 48  # Output sequence length (48 timesteps = 24 hours)
TEST_SPLIT_RATIO = 0.2  # 20% of the data for testing

MLFLOW_TRUSTED_TYPES = ["xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"]

def create_multi_step_windows(df: pd.DataFrame, target_col: str, input_window=48, output_horizon=48):
    X, y = [], []
    values = df.values
    target_idx = df.columns.get_loc(target_col)

    # We need enough data for Input + Output
    total_window_size = input_window + output_horizon
    
    # 1. Create a View (Instantaneous)
    # Shape becomes: (Samples, total_window_size)
    windows = sliding_window_view(values, window_shape=total_window_size, axis=0)
    
    # 2. Slice X and y directly from the view
    # X = The first 24 columns of every row
    # y = The last 24 columns of every row
    X = windows[:, :, :input_window]
    y = windows[:, target_idx, input_window:]
    
    # 3. Optional: Copy to ensure memory continuity (safe for XGBoost)
    # XGBoost sometimes complains about non-contiguous memory views
    return X, y

def flatten_windows(X):
    """
    Flattens 3D windows for Random Forest / XGBoost
    Input: (Samples, TimeSteps, Features)
    Output: (Samples, TimeSteps * Features)
    """
    return X.reshape(X.shape[0], -1)

def create_ml_model_pipeline(PARAMS=None, multi_output=False):
    if PARAMS is None:
        PARAMS = {}
    model = xgb.XGBRegressor(
        n_estimators=PARAMS.get('n_estimators', 100),
        learning_rate=PARAMS.get('learning_rate', 0.1),
        max_depth=PARAMS.get('max_depth', 6),
        # NEW PARAMS
        subsample=PARAMS.get('subsample', 1.0),
        colsample_bytree=PARAMS.get('colsample_bytree', 1.0),
        gamma=PARAMS.get('gamma', 0),
        reg_lambda=PARAMS.get('reg_lambda', 1),
        n_jobs=-1, 
        random_state=42
    )
    if multi_output:
        # Wrap in MultiOutputRegressor for multi-step forecasting
        model = MultiOutputRegressor(model, n_jobs=-1)
    
    #model_type = PARAMS.get('model_type', DEFAULT_MODEL_TYPE)
    # Step A: The X-Scaler and the Model
    # This pipeline handles the Inputs (X)
    x_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Automatically scales input X
        ('model', model)
    ])

    # Step B: The Y-Scaler (TransformedTargetRegressor)
    # This wraps the X-Pipeline and handles the Target (y)
    # It scales y during .fit() and INVERSE scales prediction during .predict()
    final_model = TransformedTargetRegressor(
        regressor=x_pipeline,
        transformer=StandardScaler() # This scales/unscales y
    )
    return final_model

if __name__ == "__main__":
    print("Loading datasets...")
    # Load the datasets, reading only the timestamp and real power (P) in Watts
    cols_to_use = ['unix_ts', 'P']
    df_whe = pd.read_csv(WHE_DIR, usecols=cols_to_use).rename(columns={'P': 'P_WHE'})
    df_fre = pd.read_csv(FRE_DIR, usecols=cols_to_use).rename(columns={'P': 'P_FRE'})
    df_hpe = pd.read_csv(HPE_DIR, usecols=cols_to_use).rename(columns={'P': 'P_HPE'})

    # Call the data preparation function
    processed_data = data_preparation_ampds2(df_whe, df_fre, df_hpe)
    
    X, y = create_multi_step_windows(processed_data, TARGET_COL, IN_SEQ_LEN, OUT_SEQ_LEN)
    X_flat = flatten_windows(X)
    #test_size = int(len(processed_data) * TEST_SPLIT_RATIO)
    test_size = int(len(X_flat) * TEST_SPLIT_RATIO)
    X_train_real, X_test_real = X_flat[:-test_size], X_flat[-test_size:]
    y_train_real, y_test_real = y[:-test_size], y[-test_size:]
    
    os.makedirs("mlflow", exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Energy_Consumption_Forecasting_MLOps")
    
    PARAMS = {
        'model_type': 'XGBoost',
        'n_estimators': 100,
        'max_depth': 4,
        'subsample': 0.8, 
        'learning_rate': 0.05, 
        'min_child_weight': 5, 
        'gamma': 0, 
        'colsample_bytree': 0.8
    }
    
    print("\nTraining Model on 80% Data...")
    # Start a new run to store the FINAL artifact
    with mlflow.start_run(run_name="XGBoost_Validation"):
        # Log the PARAMS again for the production record
        mlflow.log_params(PARAMS)
        
        # Create Model using your Factory Function
        # We pass 'PARAMS' which now contains the correct ints/floats
        model_type = PARAMS.get('model_type', 'XGBoost') # Default if missing
        
        # TRAIN ON FULL DATA (Ensure X_train_flat is available in scope)
        # If running separately, load your data here first.
        final_model = create_ml_model_pipeline(PARAMS, multi_output=True)
        start_time = time.time()
        final_model.fit(X_train_real, y_train_real)
        train_duration = time.time() - start_time
        
        # Evaluate
        start_time = time.time()
        preds_real_test = final_model.predict(X_test_real)
        inference_duration = time.time() - start_time
        
        preds_real_train = final_model.predict(X_train_real)
        
        # Log Metrics
        mlflow.log_metrics({
            "rmse_train": root_mean_squared_error(y_train_real, preds_real_train),
            "mae_train": mean_absolute_error(y_train_real, preds_real_train),
            "r2_train": r2_score(y_train_real, preds_real_train),
            "rmse_test": root_mean_squared_error(y_test_real, preds_real_test),
            "mae_test": mean_absolute_error(y_test_real, preds_real_test),
            "r2_test": r2_score(y_test_real, preds_real_test),
        })

        signature = infer_signature(X_train_real, y_train_real)
        # Save the model object
        mlflow.sklearn.log_model(final_model, name="model", signature=signature, skops_trusted_types=MLFLOW_TRUSTED_TYPES)

        print("✅ Success! Production model saved to MLflow.")
        print(f"   Model Type: {model_type}")
        print("   Ready for deployment.")
    

