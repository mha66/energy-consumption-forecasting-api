import pandas as pd
import pytest

from data_prep import data_preparation_ampds2


@pytest.fixture
def mock_power_data():
    """Generates 3 hours of synthetic power data at 15-minute intervals."""
    # 1. Arrange: Create a time range and convert to UNIX timestamps
    timestamps = pd.date_range("2026-08-10 10:00:00", periods=12, freq="15min")
    unix_ts = timestamps.astype('int64') // 10**9
    
    # Create P_WHE (Whole House), P_FRE (Furnace), P_HPE (Heat Pump) in Watts
    df_whe = pd.DataFrame({'unix_ts': unix_ts, 'P_WHE': [2000, 2000, 1000, 2000, 3000, 3000, 2000, 2000, 1500, 1500, 1000, 1000]})
    df_fre = pd.DataFrame({'unix_ts': unix_ts, 'P_FRE': [500,  500,  800,  500,  500,  500,  500,  500,  0,    0,    0,    0]})
    df_hpe = pd.DataFrame({'unix_ts': unix_ts, 'P_HPE': [500,  500,  300,  500,  700,  700,  500,  500,  0,    0,    0,    0]})
    
    return df_whe, df_fre, df_hpe

@pytest.fixture
def mock_power_data_clipping():
    """Generates 3 hours of synthetic power data at 15-minute intervals."""
    # 1. Arrange: Create a time range and convert to UNIX timestamps
    timestamps = pd.date_range("2026-08-10 10:00:00", periods=12, freq="30min")
    unix_ts = timestamps.astype('int64') // 10**9
    
    # Create P_WHE (Whole House), P_FRE (Furnace), P_HPE (Heat Pump) in Watts
    df_whe = pd.DataFrame({'unix_ts': unix_ts, 'P_WHE': [2000, 2000, 1000, 2000, 3000, 3000, 2000, 2000, 1500, 1500, 1000, 1000]})
    df_fre = pd.DataFrame({'unix_ts': unix_ts, 'P_FRE': [500,  500,  800,  500,  500,  500,  500,  500,  0,    0,    0,    0]})
    df_hpe = pd.DataFrame({'unix_ts': unix_ts, 'P_HPE': [500,  500,  300,  500,  700,  700,  500,  500,  0,    0,    0,    0]})
    
    return df_whe, df_fre, df_hpe

def test_baseload_clipping_and_kw_conversion(mock_power_data_clipping):
    """Ensures Watts are converted to kW and negative baseloads are clipped to 0."""
    df_whe, df_fre, df_hpe = mock_power_data_clipping
    
    # 2. Act
    # Use the default training mode (is_inference=False)
    result_df = data_preparation_ampds2(df_whe, df_fre, df_hpe)
    
    # 3. Assert
    # Check that 'Aggregate_kW' and 'HVAC_kW' were dropped as requested
    assert 'Aggregate_kW' not in result_df.columns
    assert 'HVAC_kW' not in result_df.columns
    
    # Verify Baseload_kW bounds. It should never be negative.
    assert (result_df['Baseload_kW'] >= 0).all(), "Baseload_kW contains negative values!"
    # Verify the specific clipped value (Index 2 in raw data becomes Index 1 in resampled 30min data)
    # The raw data at 10:30 had P_WHE=1000, P_FRE=800, P_HPE=300 -> Baseload should be 0, not -0.1
    assert result_df.loc[2, 'Baseload_kW'] == 0.0

def test_cyclic_features_bounds(mock_power_data):
    """Verifies that sine and cosine transformations stay within the unit circle [-1, 1]."""
    df_whe, df_fre, df_hpe = mock_power_data
    result_df = data_preparation_ampds2(df_whe, df_fre, df_hpe)
    
    cyclic_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
    
    for col in cyclic_cols:
        assert (result_df[col] >= -1.0).all()
        assert (result_df[col] <= 1.0).all()
        
def test_inference_flag_drops_nan_on_training(mock_power_data):
    """When is_inference=False, the first row should be dropped due to .diff() NaNs."""
    df_whe, df_fre, df_hpe = mock_power_data
    
    result_df = data_preparation_ampds2(df_whe, df_fre, df_hpe, is_inference=False)
    
    # 3 hours of 15-min data resampled to 30-min = 6 rows. 
    # Dropping the first row leaves 5.
    assert len(result_df) == 5
    # Ensure no NaNs exist anywhere in the final dataframe
    assert not result_df.isna().any().any()

def test_inference_flag_bfills_nan_on_inference(mock_power_data):
    """When is_inference=True, the first row should be retained and backfilled."""
    df_whe, df_fre, df_hpe = mock_power_data
    
    result_df = data_preparation_ampds2(df_whe, df_fre, df_hpe, is_inference=True)
    
    # The first row should NOT be dropped, leaving all 6 rows.
    assert len(result_df) == 6
    # Ensure no NaNs exist (meaning the bfill worked)
    assert not result_df.isna().any().any()
    
    # Explicitly check that the first two rows of 'consumption_change' are identical due to bfill
    assert result_df.loc[0, 'consumption_change'] == result_df.loc[1, 'consumption_change']