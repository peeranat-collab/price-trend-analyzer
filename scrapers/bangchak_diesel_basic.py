import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
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

    if res.status_code != 200:
        raise BangchakBasicScraperError("ไม่สามารถเข้าถึงเว็บ Bangchak")

    soup = BeautifulSoup(res.text, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        raise BangchakBasicScraperError("ไม่พบตารางข้อมูล")

    table = tables[0]
    rows = table.find_all("tr")

    data = []

    for r in rows[1:]:
        cols = r.find_all("td")
        if len(cols) < 3:
            continue

        date_text = cols[0].text.strip()
        hi_diesel_text = cols[2].text.strip()  # Hi Diesel

        try:
            price = clean_price(hi_diesel_text)
            if price is None:
                continue

            try:
                d = datetime.strptime(date_text, "%d/%m/%Y")
            except:
                parts = date_text.split("/")
                if len(parts) == 3:
                    d = datetime(int(parts[2]) - 543, int(parts[1]), int(parts[0]))
                else:
                    continue

            data.append({"date": d, "price": price})
        except:
            continue

    if not data:
        raise BangchakBasicScraperError("ไม่สามารถอ่านข้อมูลจากตารางได้")

    return pd.DataFrame(data)


def forward_fill_missing_days(df):
    df = df.sort_values("date").set_index("date")
    all_days = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(all_days)
    df["price"] = df["price"].ffill()
    df = df.reset_index().rename(columns={"index": "date"})
    return df


def get_monthly_average_basic(year, month):
    try:
        df = fetch_bangchak_diesel_raw()
        df = forward_fill_missing_days(df)

        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        month_df = df[(df["year"] == year) & (df["month"] == month)]
        if len(month_df) == 0:
            raise BangchakBasicScraperError("ไม่มีข้อมูลสำหรับเดือนนี้")

        return round(month_df["price"].mean(), 2)

    except Exception as e:
        return {
            "status": "fallback",
            "reason": str(e)
        }
