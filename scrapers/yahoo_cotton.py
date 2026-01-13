import yfinance as yf
import pandas as pd
from datetime import datetime
import calendar

USD_TO_THB = 33  # fix ตามที่คุณเลือก
LB_TO_KG = 0.453592

class CottonScraperError(Exception):
    pass


def fetch_cotton_month(year: int, month: int):
    """
    ดึงราคาผ้าฝ้ายจาก Yahoo Finance (CT=F)
    คืนค่า: ค่าเฉลี่ยรายเดือน (บาท/กิโลกรัม)
    """
    try:
        symbol = "CT=F"
        ticker = yf.Ticker(symbol)

        start_date = datetime(year, month, 1)
        last_day = calendar.monthrange(year, month)[1]
        end_date = datetime(year, month, last_day)

        hist = ticker.history(start=start_date, end=end_date)

        if hist.empty:
            raise CottonScraperError("ไม่พบข้อมูลจาก Yahoo Finance")

        if "Close" not in hist.columns:
            raise CottonScraperError("ไม่พบ column Close")

        avg_cent_per_lb = hist["Close"].mean()

        if pd.isna(avg_cent_per_lb):
            raise CottonScraperError("ข้อมูลราคาไม่สมบูรณ์")

        # cent/lb → USD/lb
        avg_usd_per_lb = float(avg_cent_per_lb) / 100

        # USD/lb → USD/kg
        avg_usd_per_kg = avg_usd_per_lb / LB_TO_KG

        # USD/kg → THB/kg
        avg_thb_per_kg = avg_usd_per_kg * USD_TO_THB

        return round(avg_thb_per_kg, 4)

    except Exception as e:
        raise CottonScraperError(str(e))


def fetch_cotton_last_36_months():
    """
    ดึงข้อมูลย้อนหลัง 36 เดือน
    คืนค่า: dict { "YYYY-MM": price_thb_per_kg }
    """
    results = {}
    now = datetime.now()

    for i in range(36):
        y = now.year
        m = now.month - i

        while m <= 0:
            m += 12
            y -= 1

        try:
            price = fetch_cotton_month(y, m)
            key = f"{y}-{str(m).zfill(2)}"
            results[key] = price
        except:
            continue

    if len(results) == 0:
        raise CottonScraperError("ไม่สามารถดึงข้อมูลย้อนหลังได้เลย")

    return results


def get_cotton_with_priority(mode="current", year=None, month=None):
    """
    mode:
    - "current" = เดือนปัจจุบัน
    - "single" = ระบุปี/เดือน
    - "last36" = ย้อนหลัง 36 เดือน
    """

    try:
        if mode == "current":
            now = datetime.now()
            value = fetch_cotton_month(now.year, now.month)
            return {
                "status": "success",
                "mode": "current",
                "value": value
            }

        elif mode == "single":
            if not year or not month:
                raise CottonScraperError("ต้องระบุ year และ month")

            value = fetch_cotton_month(year, month)
            return {
                "status": "success",
                "mode": "single",
                "year": year,
                "month": month,
                "value": value
            }

        elif mode == "last36":
            values = fetch_cotton_last_36_months()
            return {
                "status": "success",
                "mode": "last36",
                "values": values
            }

        else:
            raise CottonScraperError("mode ไม่ถูกต้อง")

    except Exception as e:
        return {
            "status": "fallback",
            "reason": str(e)
        }
