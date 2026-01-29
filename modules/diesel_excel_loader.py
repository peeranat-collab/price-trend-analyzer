import pandas as pd

def load_diesel_excel(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file)

    # Rename columns
    df = df.rename(columns={
        "วันที่": "date",
        "ไฮดีเซล": "price"
    })

    required_cols = {"date", "price"}
    if not required_cols.issubset(df.columns):
        raise ValueError("ไฟล์ต้องมีคอลัมน์ 'วันที่' และ 'ไฮดีเซล'")

    # ===== FIX พ.ศ. → ค.ศ. (ปลอดภัย) =====
    df["date"] = df["date"].astype(str)

    def convert_be_to_ad(d):
        # รูปแบบ: 29/12/2566 หรือ 29-12-2566
        parts = d.replace("-", "/").split("/")
        day = int(parts[0])
        month = int(parts[1])
        year = int(parts[2]) - 543
        return pd.Timestamp(year=year, month=month, day=day)

    df["date"] = df["date"].apply(convert_be_to_ad)

    # Price
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["date", "price"])
    df = df.sort_values("date").reset_index(drop=True)

    return df
