import yfinance as yf
import pandas as pd
from datetime import datetime

USD_TO_THB = 33  # fix rate (ปรับทีหลังได้)

def get_hrc_with_priority(mode="current"):
    """
    mode = current | last36
    """
    ticker = yf.Ticker("HRC=F")

    try:
        if mode == "current":
            hist = ticker.history(period="1mo")
            if hist.empty:
                raise ValueError("No data")

            avg_price_usd = hist["Close"].mean()
            price_thb = avg_price_usd * USD_TO_THB
            avg_thb_per_kt = price_thb *907 /1000000

            return round(avg_thb_per_kt, 2)


        elif mode == "last36":
            hist = ticker.history(period="36mo")
            if hist.empty:
                raise ValueError("No data")

            hist["month"] = hist.index.to_period("M")
            monthly = hist.groupby("month")["Close"].mean()

            result = {}
            for k, v in monthly.items():
                result[str(k)] = round(v * USD_TO_THB, 2)

            return {
                "mode": "last36",
                "values": result
            }

    except Exception as e:
        return {
            "status": "fallback",
            "reason": str(e)
        }
