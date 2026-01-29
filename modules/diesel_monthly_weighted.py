import pandas as pd

def daily_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    monthly = (
        df.groupby(["year", "month"], as_index=False)
          .agg(avg_price=("price", "mean"))
    )

    return monthly
