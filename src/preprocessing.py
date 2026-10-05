import pandas as pd
import numpy as np

def generate_sample_data(n_weeks: int = 20) -> pd.DataFrame:
    np.random.seed(42)
    weeks = list(range(1, n_weeks + 1))
    sales = np.random.poisson(lam=50, size=n_weeks) + np.linspace(10, 30, n_weeks)
    return pd.DataFrame({"week": weeks, "sales": sales})

def create_point_in_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("week").copy()
    df["sales_lag_1"] = df["sales"].shift(1)
    return df.dropna().reset_index(drop=True)