import yfinance as yf
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

# =========================
# CONFIG
# =========================

# Aluminum Futures (COMEX)
AL_SYMBOL = "ALI=F"

# USD → THB (ค่าเฉลี่ยโดยประมาณ ถ้าคุณมี API FX ค่อยเปลี่ยน)
USD_TO_THB = 35.0

# =========================
# Helpers
# =========================

def get_last_n_months(n=36):
    """
    คืน list ของ (year, month) ย้อนหลัง n เดือน
    """
    results = []
    today = datetime.today().replace(day=1)

    for i in range(n):
        d = today - relativedelta(months=i)
        results.append((d.year, d.month))

    return results[::-1]


def _convert_usd_to_thb_per_ton(price_usd):
    """
    แปลง USD/ton → THB/ton
    """
    return round(price_usd * USD_TO_THB, 2)


# =========================
# Main Function
# =========================

def get_aluminum_monthly_avg_thb(year, month):
    """
    ดึงราคาอะลูมิเนียมจาก Yahoo Finance
    แล้วคำนวณค่าเฉลี่ยรายเดือน
    คืนค่าเป็น float (บาท/ตัน)
    """

    start_date = datetime(year, month, 1)
    end_date = start_date + relativedelta(months=1)

    df = yf.download(
        AL_SYMBOL,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        progress=False
    )

    if df is None or df.empty:
        raise ValueError("ไม่พบข้อมูลจาก Yahoo Finance")

    # ใช้ราคาปิด
    if "Close" not in df.columns:
        raise ValueError("ไม่พบ column Close")

    monthly_avg_usd = df["Close"].mean()

    if pd.isna(monthly_avg_usd):
        raise ValueError("ค่าเฉลี่ยเป็น NaN")

    thb_price = _convert_usd_to_thb_per_ton(monthly_avg_usd)

    return float(thb_price)
