from typing import Generator, List, Tuple
import pandas as pd


def expanding_window_split(df: pd.DataFrame, cutoffs: List[int], horizon: int = 4, time_col: str = "week") -> Generator[Tuple[int, pd.DataFrame, pd.DataFrame], None, None]:
    
    sorted_cutoffs = sorted(cutoffs)
    min_week = df[time_col].min()
    max_week = df[time_col].max()

    for cutoff in sorted_cutoffs:
        test_end = cutoff + horizon

        # Validation: ensure cutoff and horizon are within available dataset limits
        if cutoff < min_week:
            raise ValueError(
                f"Cutoff {cutoff} is smaller than minimum week {min_week}."
            )
        if test_end > max_week:
            raise ValueError(
                f"Evaluation window [{cutoff + 1}..{test_end}] exceeds maximum week {max_week}."
            )

        # Expanding train split: all historical observations up to cutoff
        train_mask = df[time_col] <= cutoff

        # Out-of-time test split: strictly next H weeks
        test_mask = (df[time_col] > cutoff) & (df[time_col] <= test_end)

        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()

        yield cutoff, train_df, test_df