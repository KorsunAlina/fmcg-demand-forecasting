import numpy as np
import pandas as pd


def generate_time_series_data(
    num_series: int = 50, num_weeks: int = 104, random_seed: int = 42
) -> pd.DataFrame:
    np.random.seed(random_seed)

    records = []

    for series_idx in range(1, num_series + 1):
        series_id = f"series_{series_idx:03d}"

        base_demand = np.random.uniform(50, 500)  
        trend_coef = np.random.uniform(-0.5, 1.5)  
        seasonality_amplitude = np.random.uniform(
            0.1, 0.4
        )  # impact of seasonality (10-40%)
        promo_sensitivity = np.random.uniform(0.1, 0.3)  # sensitivity to promotion

        for week in range(1, num_weeks + 1):
            trend = trend_coef * week
            seasonality = (np.sin(2 * np.pi * week / 52) * base_demand * seasonality_amplitude)

            is_promo = int(np.random.rand() < 0.2)  # promo in 20% of cases
            price = round(np.random.uniform(10, 100), 2)
            promo_effect = is_promo * promo_sensitivity * base_demand

            noise = np.random.normal(loc=0, scale=0.1 * base_demand)

            target_val = (base_demand + trend + seasonality + promo_effect + noise)
            target_val = max(0, target_val )  

            # Zero weeks / Out-of-stock
            # with a 5% probability weekly sales are 0
            if np.random.rand() < 0.05:
                target_val = 0.0

            records.append({
                "series_id": series_id,
                "week": int(week),
                "target": round(float(target_val), 2),
                "price": price,
                "is_promo": is_promo,
            })

    df = pd.DataFrame(records)

    # type casting for storage into a Parquet
    df["series_id"] = df["series_id"].astype("category")
    df["week"] = df["week"].astype("int16")
    df["target"] = df["target"].astype("float32")
    df["price"] = df["price"].astype("float32")
    df["is_promo"] = df["is_promo"].astype("int8")

    return df



# data generation (50 rows × 104 weeks = 5 200 rows)
df_series = generate_time_series_data(num_series=50, num_weeks=104)

output_filename = "dataset_104w_50s.parquet"
df_series.to_parquet(output_filename, engine="pyarrow", index=False)


print(df_series.head())