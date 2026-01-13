import yfinance as yf
import pandas as pd
from datetime import datetime

class AluminumYahooError(Exception):
    pass


def _get_monthly_avg(symbol, start_date, end_date):
    """
    ดึงข้อมูลรายวันจาก Yahoo แล้วคำนวณค่าเฉลี่ยรายเดือน
    """
    df = yf.download(symbol, start=start_date, end=end_date, interval="1d", progress=False)

    if df.empty:
        raise AluminumYahooError(f"ไม่พบข้อมูลสำหรับ {symbol}")

    df = df.reset_index()
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month

    monthly = (
        df.groupby(["year", "month"])["Close"]
        .mean()
        .reset_index()
    )

    return monthly


def get_aluminum_monthly_avg_usd(year, month):
    """
    ดึงราคา Aluminum (USD/ton) เฉลี่ยรายเดือน
    """
    # Aluminum Futures
    symbol = "ALI=F"

    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-28"  # Yahoo ไม่ strict เรื่องวัน

    monthly = _get_monthly_avg(symbol, start_date, end_date)

    row = monthly[(monthly["year"] == year) & (monthly["month"] == month)]
    if row.empty:
        raise AluminumYahooError("ไม่พบข้อมูล Aluminum สำหรับเดือนนี้")

    return round(float(row["Close"].values[0]), 2)


def get_usd_thb_monthly_avg(year, month):
    """
    ดึงค่าเฉลี่ย USD/THB รายเดือน
    """
    symbol = "USDTHB=X"

    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-28"

    monthly = _get_monthly_avg(symbol, start_date, end_date)

    row = monthly[(monthly["year"] == year) & (monthly["month"] == month)]
    if row.empty:
        raise AluminumYahooError("ไม่พบข้อมูล USD/THB สำหรับเดือนนี้")

    return round(float(row["Close"].values[0]), 4)


def get_aluminum_monthly_avg_thb(year, month):
    """
    แปลง Aluminum จาก USD/ton → THB/ton
    โดยใช้ FX เฉลี่ยรายเดือน
    """
    aluminum_usd = get_aluminum_monthly_avg_usd(year, month)
    usd_thb = get_usd_thb_monthly_avg(year, month)

    aluminum_thb = aluminum_usd * usd_thb
    return round(aluminum_thb, 2)


def get_last_n_months(n=36):
    """
    คืน list ของ (year, month) ย้อนหลัง n เดือน
    """
    today = datetime.today()
    months = []

    y = today.year
    m = today.month

    for _ in range(n):
        months.append((y, m))
        m -= 1
        if m == 0:
            m = 12
            y -= 1

    return months
