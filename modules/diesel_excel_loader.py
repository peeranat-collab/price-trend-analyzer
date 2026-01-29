import pandas as pd

def load_diesel_excel(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file)

    # --- Rename ให้เป็นมาตรฐาน ---
    df = df.rename(columns={
        "วันที่": "date",
        "ไฮดีเซล": "price"
    })

    # --- Validate ---
    required_cols = {"date", "price"}
    if not required_cols.issubset(df.columns):
        raise ValueError("ไฟล์ต้องมีคอลัมน์ 'วันที่' และ 'ไฮดีเซล'")

    # --- แปลงวันที่ (พ.ศ. → ค.ศ.) ---
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df["date"] = df["date"] - pd.DateOffset(years=543)

    # --- แปลงราคา ---
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["date", "price"])
    df = df.sort_values("date").reset_index(drop=True)

    return df
