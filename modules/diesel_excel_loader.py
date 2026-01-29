import pandas as pd
from datetime import datetime

def load_diesel_excel(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file)

    # Rename columns
    df = df.rename(columns={
        "วันที่": "date",
        "ไฮดีเซล": "price"
    })

    if not {"date", "price"}.issubset(df.columns):
        raise ValueError("ไฟล์ต้องมีคอลัมน์ 'วันที่' และ 'ไฮดีเซล'")

    def normalize_date(val):
        # ===== Case 1: Excel / pandas datetime =====
        if isinstance(val, (pd.Timestamp, datetime)):
            year = val.year
            if year > 2400:  # พ.ศ.
                year -= 543
            return pd.Timestamp(year=year, month=val.month, day=val.day)

        # ===== Case 2: string =====
        s = str(val).strip()

        # ตัดเวลาออกก่อน
        if " " in s:
            s = s.split(" ")[0]

        s = s.replace("-", "/")
        d, m, y = s.split("/")

        year = int(y)
        if year > 2400:
            year -= 543

        return pd.Timestamp(year=year, month=int(m), day=int(d))

    # Apply normalize
    df["date"] = df["date"].apply(normalize_date)

    # Price
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["date", "price"])
    df = df.sort_values("date").reset_index(drop=True)

    return df
