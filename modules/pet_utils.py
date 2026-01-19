import pandas as pd

def add_pct_change(df, value_col):
    df = df.copy()
    df["% เปลี่ยนแปลง"] = df[value_col].pct_change() * 100
    return df
