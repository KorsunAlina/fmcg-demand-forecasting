import pandas as pd
from src.preprocessing import create_point_in_time_features

def test_no_data_leakage_in_lags():
    raw_df = pd.DataFrame({
        "week": [1, 2, 3],
        "sales": [100.0, 150.0, 200.0]
    })
    
    processed_df = create_point_in_time_features(raw_df)
    
    week_3_lag = processed_df.loc[processed_df["week"] == 3, "sales_lag_1"].values[0]
    week_2_actual = raw_df.loc[raw_df["week"] == 2, "sales"].values[0]
    
    assert week_3_lag == week_2_actual