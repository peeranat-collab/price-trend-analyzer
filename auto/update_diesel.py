from scrapers.bangchak_diesel import get_monthly_average
import pandas as pd
from datetime import datetime
import os

DATA_FILE = "data.csv"

products = [
    "กระเป๋า Delivery ใบเล็ก",
    "กระเป๋า Delivery ใบใหญ่",
    "แจ็คเก็ต Delivery"
]

def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame(columns=[
            "สินค้า", "เดือน", "ปี", "วัสดุ",
            "ราคา/หน่วย", "ปริมาณ", "ต้นทุน",
            "overhead_percent", "timestamp"
        ])

def save_data(df):
    df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")

def main():
    today = datetime.today()
    year = today.year
    month = today.month

    result = get_monthly_average(year, month)

    if isinstance(result, dict):
        print("Fallback mode:", result)
        return

    price = result

    new_rows = []

    for product in products:
        new_rows.append({
            "สินค้า": product,
            "เดือน": month,
            "ปี": year,
            "วัสดุ": "ค่าขนส่ง (น้ำมันดีเซล)",
            "ราคา/หน่วย": price,
            "ปริมาณ": 1,
            "ต้นทุน": price,
            "overhead_percent": 0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    new_df = pd.DataFrame(new_rows)
    old_df = load_data()

    # ลบข้อมูลซ้ำ
    if len(old_df) > 0:
        old_df = old_df[
            ~(
                (old_df["วัสดุ"] == "ค่าขนส่ง (น้ำมันดีเซล)") &
                (old_df["เดือน"] == month) &
                (old_df["ปี"] == year)
            )
        ]

    final_df = pd.concat([old_df, new_df], ignore_index=True)
    save_data(final_df)

    print("Auto update completed:", price)

if __name__ == "__main__":
    main()
