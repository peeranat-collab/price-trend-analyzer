import pandas as pd
from datetime import timedelta

def split_week_to_months(row):
    """
    รับ 1 แถว: ปี, สัปดาห์, ราคา, วันที่เริ่ม, วันที่สิ้นสุด
    คืนค่า list ของ dict ที่แยกตามเดือนพร้อม weight
    """
    start = row["วันที่เริ่ม"]
    end = row["วันที่สิ้นสุด"]
    price = row["ราคา"]

    total_days = (end - start).days + 1
    results = []

    cur = start
    while cur <= end:
        month_start = cur.replace(day=1)
        next_month = (month_start + pd.offsets.MonthBegin(1)).to_pydatetime()

        month_end = min(end, next_month - timedelta(days=1))

        days_in_month_part = (month_end - cur).days + 1
        weight = days_in_month_part / total_days

        results.append({
            "ปี": cur.year,
            "เดือน": cur.month,
            "weighted_price": price * weight,
            "weight": weight
        })

        cur = month_end + timedelta(days=1)

    return results


def convert_weekly_to_monthly_weighted(weekly_df):
    """
    รับ weekly_df (จาก Part 1.2)
    คืนค่า monthly_df:
    - ปี
    - เดือน
    - ราคาเฉลี่ยถ่วงน้ำหนัก (บาท/กก.)
    """

    rows = []

    for _, row in weekly_df.iterrows():
        splits = split_week_to_months(row)
        rows.extend(splits)

    temp_df = pd.DataFrame(rows)

    # รวมตาม ปี + เดือน
    grouped = temp_df.groupby(["ปี", "เดือน"]).agg(
        total_weighted_price=("weighted_price", "sum"),
        total_weight=("weight", "sum")
    ).reset_index()

    grouped["ราคาเฉลี่ย (บาท/กก.)"] = grouped["total_weighted_price"] / grouped["total_weight"]

    monthly_df = grouped[["ปี", "เดือน", "ราคาเฉลี่ย (บาท/กก.)"]].copy()

    monthly_df = monthly_df.sort_values(by=["ปี", "เดือน"]).reset_index(drop=True)

    return monthly_df
