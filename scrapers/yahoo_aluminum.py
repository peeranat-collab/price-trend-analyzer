import yfinance as yf
import pandas as pd
from datetime import datetime
import calendar

USD_TO_THB = 33  # ค่า fix ตามที่คุณเลือก

class AluminumScraperError(Exception):
    pass


def fetch_aluminum_month(year: int, month: int):
    """
    ดึงราคาของเดือนนั้นจาก Yahoo Finance
    ใช้ Symbol: ALI=F
    คืนค่า: ค่าเฉลี่ยรายเดือน (บาท/กก.)
    """
    try:
        symbol = "ALI=F"
        ticker = yf.Ticker(symbol)

        start_date = datetime(year, month, 1)
        last_day = calendar.monthrange(year, month)[1]
        end_date = datetime(year, month, last_day)

        hist = ticker.history(start=start_date, end=end_date)

        if hist.empty:
            raise AluminumScraperError("ไม่พบข้อมูลจาก Yahoo Finance")

        avg_usd = hist["Close"].mean()

        if pd.isna(avg_usd):
            raise AluminumScraperError("ข้อมูลราคาไม่สมบูรณ์")

        avg_thb_per_ton = float(avg_usd) * USD_TO_THB
        avg_thb_per_kt = avg_thb_per_ton / 1000 
        return round(avg_thb_per_kt, 2)


    except Exception as e:
        raise AluminumScraperError(str(e))


def fetch_aluminum_last_36_months():
    """
    ดึงข้อมูลย้อนหลัง 36 เดือน
    คืนค่า: dict { "YYYY-MM": price_thb }
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
            price = fetch_aluminum_month(y, m)
            key = f"{y}-{str(m).zfill(2)}"
            results[key] = price
        except:
            continue

    if len(results) == 0:
        raise AluminumScraperError("ไม่สามารถดึงข้อมูลย้อนหลังได้เลย")

    return results


def get_aluminum_with_priority(mode="current", year=None, month=None):
    """
    mode:
    - "current" = เดือนปัจจุบัน
    - "single" = ระบุปี/เดือน
    - "last36" = ย้อนหลัง 36 เดือน
    """

    try:
        if mode == "current":
            now = datetime.now()
            value = fetch_aluminum_month(now.year, now.month)
            return {
                "status": "success",
                "mode": "current",
                "value": value
            }

        elif mode == "single":
            if not year or not month:
                raise AluminumScraperError("ต้องระบุ year และ month")

            value = fetch_aluminum_month(year, month)
            return {
                "status": "success",
                "mode": "single",
                "year": year,
                "month": month,
                "value": value
            }

        elif mode == "last36":
            values = fetch_aluminum_last_36_months()
            return {
                "status": "success",
                "mode": "last36",
                "values": values
            }

        else:
            raise AluminumScraperError("mode ไม่ถูกต้อง")

    except Exception as e:
        return {
            "status": "fallback",
            "reason": str(e)
        }
