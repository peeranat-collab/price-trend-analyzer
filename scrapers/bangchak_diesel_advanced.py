from playwright.sync_api import sync_playwright
import pandas as pd

class BangchakAdvancedScraperError(Exception):
    pass

def get_monthly_average_advanced(year, month):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("https://www.bangchak.co.th/th/oilprice/historical", timeout=60000)

            page.wait_for_timeout(5000)

            rows = page.query_selector_all("table tbody tr")
            if not rows:
                raise BangchakAdvancedScraperError("ไม่พบแถวข้อมูล")

            data = []

            for r in rows:
                cols = r.query_selector_all("td")
                if len(cols) < 3:
                    continue

                date_text = cols[0].inner_text().strip()
                price_text = cols[2].inner_text().strip()

                try:
                    price = float(price_text.replace(",", ""))
                except:
                    continue

                data.append({"date": date_text, "price": price})

            browser.close()

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna()
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        filtered = df[(df["year"] == year) & (df["month"] == month)]
        if len(filtered) == 0:
            raise BangchakAdvancedScraperError("ไม่พบข้อมูลเดือนนี้")

        return round(filtered["price"].mean(), 2)

    except Exception as e:
        raise BangchakAdvancedScraperError(str(e))
