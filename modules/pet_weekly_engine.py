import pandas as pd

def normalize_weekly_pet_data(df):
    """
    รับ DataFrame PET รายสัปดาห์
    คืนค่า DataFrame ที่:
    - มี column: ปี, สัปดาห์, ราคา, วันที่เริ่ม, วันที่สิ้นสุด
    - Replace ถ้าซ้ำ (ปี + สัปดาห์)
    """

    # ดึงปีจากวันที่เริ่ม
    df = df.copy()
    df["ปี"] = df["วันที่เริ่ม"].dt.year

    # เลือก column ที่ต้องใช้
    weekly_df = df[["ปี", "สัปดาห์", "ราคา", "วันที่เริ่ม", "วันที่สิ้นสุด"]].copy()

    # เรียงก่อน replace
    weekly_df = weekly_df.sort_values(by=["ปี", "สัปดาห์"])

    # Replace duplicates: keep last
    weekly_df = weekly_df.drop_duplicates(subset=["ปี", "สัปดาห์"], keep="last")

    weekly_df = weekly_df.reset_index(drop=True)

    return weekly_df
