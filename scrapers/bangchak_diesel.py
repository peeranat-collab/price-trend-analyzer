import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import re

class BangchakScraperError(Exception):
    pass


def clean_price(text):
    """
    แปลง '31.24 บาท/ลิตร' -> 31.24
    """
    text = text.replace(",", "")
    numbers = re.findall(r"\d+\.?\d*", text)
    if not numbers:
        return None
    return float(numbers[0])


def fetch_bangchak_diesel_raw():
    """
    ดึงข้อมูลดิบจาก Bangchak
    คืนค่า: DataFrame(date, price)
    """
    url = "https://www.bangchak.co.th/th/oilprice/historical"
    res = requests.get(url, timeout=20)
    res.encoding = "utf-8"

    if res.status_code != 200:
        raise BangchakScraperError("ไม่สามารถเข้าถึงเว็บ Bangchak")

    soup = BeautifulSoup(res.text, "html.parser")

    tables = soup.find_all("table")
    if not tables:
        raise BangchakScraperError("ไม่พบตารางข้อมูล")

    target_table = None

    # พยายามหาตารางที่มีคำว่า 'ดีเซล'
    for t in tables:
        if "ดีเซล" in t.text:
            target_table = t
            break

    if target_table is None:
        # fallback: ใช้ table แรก
        target_table = tables[0]

    rows = target_table.find_all("tr")
    data = []

    for r in rows[1:]:
        cols = r.find_all("td")
        if len(cols) < 2:
            continue

        date_text = cols[0].text.strip()
        diesel_text = cols[1].text.strip()

        try:
            price = clean_price(diesel_text)
            if price is None:
                continue

            # ปรับ format วันที่ถ้าจำเป็น
            date_obj = datetime.strptime(date_text, "%d/%m/%Y")

            data.append({
                "date": date_obj,
                "price": price
            })
        except Exception:
            continue

    if len(data) == 0:
        raise BangchakScraperError("ไม่สามารถอ่านข้อมูลจากตารางได้")

    df = pd.DataFrame(data)
    return df


def forward_fill_missing_days(df):
    """
    ถ้ามีวันที่ขาด → ใช้วันก่อนหน้า
    """
    df = df.sort_values("date")
    df = df.set_index("date")

    all_days = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(all_days)
    df["price"] = df["price"].ffill()

    df = df.reset_index().rename(columns={"index": "date"})
    return df


def get_monthly_average(year, month):
    """
    คืนค่า:
    - float: ค่าเฉลี่ยราคาดีเซลรายเดือน
    - dict: fallback mode
    """
    try:
        df = fetch_bangchak_diesel_raw()
        df = forward_fill_missing_days(df)

        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        month_df = df[(df["year"] == year) & (df["month"] == month)]

        if len(month_df) == 0:
            raise BangchakScraperError("ไม่มีข้อมูลสำหรับเดือนนี้")

        return round(month_df["price"].mean(), 2)

    except BangchakScraperError as e:
        return {
            "status": "fallback",
            "reason": str(e)
        }
