import pandas as pd

def expand_wage_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    จาก effective date → ขยายเป็นค่าแรงรายเดือน
    """
    df = df.sort_values("date")

    start = df["date"].min()
    end = pd.Timestamp.today().replace(day=1)

    all_months = pd.date_range(start=start, end=end, freq="MS")

    monthly = pd.DataFrame({"date": all_months})
    monthly = pd.merge_asof(
        monthly,
        df[["date", "wage"]],
        on="date",
        direction="backward"
    )

    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month

    return monthly[["year", "month", "wage"]]
