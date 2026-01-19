import pandas as pd
from datetime import datetime

WEEKLY_FILE = "data/pet_weekly_raw.csv"

import os

def save_weekly_raw(df):
    os.makedirs(os.path.dirname(WEEKLY_FILE), exist_ok=True)  # ← เพิ่มบรรทัดนี้

    if os.path.exists(WEEKLY_FILE):
        old = pd.read_csv(WEEKLY_FILE)
        combined = pd.concat([old, df], ignore_index=True)
    else:
        combined = df.copy()

    combined.to_csv(WEEKLY_FILE, index=False, encoding="utf-8-sig")



def convert_monthly_to_main_schema(monthly_df, products):
    """
    แปลง monthly_df ให้เป็น schema หลักของระบบ
    """

    rows = []

    for _, r in monthly_df.iterrows():
        for product in products:
            rows.append({
                "สินค้า": product,
                "เดือน": int(r["เดือน"]),
                "ปี": int(r["ปี"]),
                "วัสดุ": "เม็ดพลาสติก PET",
                "ราคา/หน่วย": float(r["ราคาเฉลี่ย (บาท/กก.)"]),
                "ปริมาณ": 1,
                "ต้นทุน": float(r["ราคาเฉลี่ย (บาท/กก.)"]),
                "overhead_percent": 0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    return pd.DataFrame(rows)
