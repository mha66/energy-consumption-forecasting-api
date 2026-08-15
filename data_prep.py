import os

import numpy as np
import pandas as pd


def data_preparation_ampds2(df_whe: pd.DataFrame, df_fre: pd.DataFrame, df_hpe: pd.DataFrame, sample_rate: str = '30min', is_inference: bool = False) -> pd.DataFrame:
    print("Merging and converting timestamps...")
    # Merge the dataframes sequentially on the UNIX timestamp
    df = df_whe.merge(df_fre, on='unix_ts').merge(df_hpe, on='unix_ts')

    # Convert unix_ts to a proper pandas datetime object and set it as the index
    df['datetime'] = pd.to_datetime(df['unix_ts'], unit='s')
    df = df.set_index('datetime')

    print("Calculating power variables...")
    # AMPds2 records power in Watts. Convert to kilowatts (kW).
    df['Aggregate_kW'] = df['P_WHE'] / 1000.0
    df['HVAC_kW'] = (df['P_FRE'] + df['P_HPE']) / 1000.0
    df['Baseload_kW'] = df['Aggregate_kW'] - df['HVAC_kW']

    # Clip baseload to 0 to prevent negative values from minor metering drift
    df['Baseload_kW'] = df['Baseload_kW'].clip(lower=0)

    print("Resampling to 30-minute intervals...")
    # Resample to 30 minutes using the mean to represent average power over the window
    df_resampled = df[['Aggregate_kW', 'HVAC_kW', 'Baseload_kW']].resample(sample_rate).mean()

    print(df_resampled.shape)
    print("Engineering features...")
    # 1. Consumption Change (Difference from the previous 30-minute timestep)
    df_resampled['consumption_change'] = df_resampled['Aggregate_kW'].diff()

    # 2. Cyclical Time Features (Hour of day & Day of week)
    # Get fractional hours (e.g., 1:30 PM = 13.5)
    hours = df_resampled.index.hour + (df_resampled.index.minute / 60.0)
    # Day of week (Monday=0, Sunday=6)
    days_of_week = df_resampled.index.dayofweek
    months = df_resampled.index.month
    # Apply Sine and Cosine transformations
    df_resampled['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
    df_resampled['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
    df_resampled['day_sin'] = np.sin(2 * np.pi * days_of_week / 7.0)
    df_resampled['day_cos'] = np.cos(2 * np.pi * days_of_week / 7.0)
    df_resampled['month_sin'] = np.sin(2 * np.pi * (months - 1) / 12)
    df_resampled['month_cos'] = np.cos(2 * np.pi * (months - 1) / 12)

    # Reset index to turn 'datetime' back into a standard column
    df_final = df_resampled.reset_index()

    # Drop the first row since `.diff()` results in a NaN value for the first entry
    #df_final = df_final.dropna()

    # Filter and order the exact columns requested
    final_columns = [
        'datetime', 'Aggregate_kW', 'HVAC_kW', 'Baseload_kW', 
        'consumption_change', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
    ]

    print("Processing complete!")

    processed_data = df_final[final_columns]
    processed_data = processed_data.select_dtypes(include=[np.number])
    processed_data.drop(['Aggregate_kW', 'HVAC_kW'], inplace=True, axis=1)
    
    # Instead of dropping NaNs blindly during inference, handle the first row leakage
    if is_inference:
        # For inference, backfill the single NaN created by .diff() 
        # so you don't lose the critical first timestep of your 48-step window
        processed_data['consumption_change'] = processed_data['consumption_change'].bfill()
    else:
        # For training, it is perfectly fine to drop the initial NaN row
        processed_data.dropna(inplace=True)
    
    #processed_data.set_index('datetime', inplace=True)
    return processed_data


if __name__ == "__main__":
    RAW_DATA_DIR = os.path.join('data', 'raw', 'ampds2')
    WHE_DIR = os.path.join(RAW_DATA_DIR, 'Electricity_WHE.csv') # Whole House Energy
    FRE_DIR = os.path.join(RAW_DATA_DIR, 'Electricity_FRE.csv') # Furnace / HVAC
    HPE_DIR = os.path.join(RAW_DATA_DIR, 'Electricity_HPE.csv') # Heat Pump Energy
    cols_to_use = ['unix_ts', 'P']
    df_whe = pd.read_csv(WHE_DIR, usecols=cols_to_use).rename(columns={'P': 'P_WHE'})
    df_fre = pd.read_csv(FRE_DIR, usecols=cols_to_use).rename(columns={'P': 'P_FRE'})
    df_hpe = pd.read_csv(HPE_DIR, usecols=cols_to_use).rename(columns={'P': 'P_HPE'})

    # Call the data preparation function
    processed_data = data_preparation_ampds2(df_whe, df_fre, df_hpe)
    
    print(processed_data[34874:34922].values.tolist())