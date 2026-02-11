import yfinance as yf
import pandas as pd

USD_TO_THB = 33
SHORT_TON_TO_GRAM = 907_185  # 907.185 kg

def get_hrc_with_priority(mode="current"):
    """
    mode = current | last36
    return = บาท / กรัม
    """

    ticker = yf.Ticker("HRC=F")

    try:
        # =========================
        # CURRENT MONTH
        # =========================
        if mode == "current":
            hist = ticker.history(period="1mo")

            if hist.empty:
                raise ValueError("No data")

            avg_price_usd = hist["Close"].mean()

            if pd.isna(avg_price_usd):
                raise ValueError("Invalid price")

            price_thb_per_short_ton = avg_price_usd * USD_TO_THB
            price_thb_per_gram = price_thb_per_short_ton / SHORT_TON_TO_GRAM

            return {
                "status": "success",
                "mode": "current",
                "value": round(price_thb_per_gram, 6)
            }

        # =========================
        # LAST 36 MONTHS
        # =========================
        elif mode == "last36":
            hist = ticker.history(period="36mo")

            if hist.empty:
                raise ValueError("No data")

            hist["month"] = hist.index.to_period("M")
            monthly = hist.groupby("month")["Close"].mean()

            result = {}

            for k, v in monthly.items():

                if pd.isna(v):
                    continue

                price_thb_per_short_ton = v * USD_TO_THB
                price_thb_per_gram = price_thb_per_short_ton / SHORT_TON_TO_GRAM

                result[str(k)] = round(price_thb_per_gram, 6)

            if len(result) == 0:
                raise ValueError("No valid monthly data")

            return {
                "status": "success",
                "mode": "last36",
                "values": result
            }

        else:
            raise ValueError("Invalid mode")

    except Exception as e:
        return {
            "status": "fallback",
            "reason": str(e)
        }
