import pandas as pd
import os

DATA_PATH = "data/diesel_monthly.csv"

def save_monthly_diesel(df_new: pd.DataFrame):
    os.makedirs("data", exist_ok=True)

    if os.path.exists(DATA_PATH):
        df_old = pd.read_csv(DATA_PATH)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["year", "month"], keep="last")
    else:
        df = df_new

    df = df.sort_values(["year", "month"])
    df.to_csv(DATA_PATH, index=False)
