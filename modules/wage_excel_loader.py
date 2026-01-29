import pandas as pd
from datetime import datetime

def load_wage_excel(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file)

    df = df.rename(columns={
        "วันที่": "date",
        "ค่าแรงขั้นต่ำ": "wage"
    })

    if not {"date", "wage"}.issubset(df.columns):
        raise ValueError("ไฟล์ต้องมีคอลัมน์ 'วันที่' และ 'ค่าแรงขั้นต่ำ'")

    def normalize_date(val):
        if isinstance(val, (pd.Timestamp, datetime)):
            y = val.year - 543 if val.year > 2400 else val.year
            return pd.Timestamp(year=y, month=val.month, day=1)

        s = str(val).strip()
        if " " in s:
            s = s.split(" ")[0]

        s = s.replace("-", "/")
        d, m, y = s.split("/")
        y = int(y) - 543 if int(y) > 2400 else int(y)

        return pd.Timestamp(year=y, month=int(m), day=1)

    df["date"] = df["date"].apply(normalize_date)
    df["wage"] = pd.to_numeric(df["wage"], errors="coerce")

    df = df.dropna().sort_values("date").reset_index(drop=True)

    return df
