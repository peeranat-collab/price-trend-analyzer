import requests
import pandas as pd
from bs4 import BeautifulSoup
import re

class BangchakBasicScraperError(Exception):
    pass

def clean_price(text):
    text = text.replace(",", "")
    nums = re.findall(r"\d+\.?\d*", text)
    return float(nums[0]) if nums else None

def fetch_bangchak_diesel_raw():
    url = "https://www.bangchak.co.th/th/oilprice/historical"
    res = requests.get(url, timeout=20)
    res.encoding = "utf-8"

    soup = BeautifulSoup(res.text, "html.parser")
    table = soup.find("table")
    if table is None:
        raise BangchakBasicScraperError("ไม่พบตารางข้อมูล")

    rows = table.find_all("tr")
    data = []

    for r in rows[1:]:
        cols = r.find_all("td")
        if len(cols) < 3:
            continue

        date_text = cols[0].get_text(strip=True)
        diesel_text = cols[2].get_text(strip=True)  # ปรับ index ตามเว็บ

        price = clean_price(diesel_text)
        if price:
            data.append({"date": date_text, "price": price})

    if not data:
        raise BangchakBasicScraperError("ไม่พบข้อมูลราคา")

    return pd.DataFrame(data)

def get_monthly_average_basic(year, month):
    try:
        df = fetch_bangchak_diesel_raw()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna()
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        filtered = df[(df["year"] == year) & (df["month"] == month)]
        if len(filtered) == 0:
            raise BangchakBasicScraperError("ไม่พบข้อมูลในเดือนนี้")

        return round(filtered["price"].mean(), 2)

    except Exception as e:
        raise BangchakBasicScraperError(str(e))
