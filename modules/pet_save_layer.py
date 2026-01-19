import pandas as pd
from datetime import datetime

WEEKLY_FILE = "data/pet_weekly_raw.csv"

def save_weekly_raw(weekly_df):
    """
    บันทึก Raw รายสัปดาห์
    Replace ถ้าซ้ำ (ปี + สัปดาห์)
    """
    try:
        old_df = pd.read_csv(WEEKLY_FILE)
    except:
        old_df = pd.DataFrame(columns=weekly_df.columns)

    combined = pd.concat([old_df, weekly_df], ignore_index=True)

    # Replace duplicates
    combined = combined.sort_values(by=["ปี", "สัปดาห์"])
    combined = combined.drop_duplicates(subset=["ปี", "สัปดาห์"], keep="last")

    combined.to_csv(WEEKLY_FILE, index=False, encoding="utf-8-sig")

    return combined


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
