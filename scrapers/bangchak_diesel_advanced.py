from playwright.sync_api import sync_playwright
from datetime import datetime
import re
import time

class BangchakAdvancedScraperError(Exception):
    pass


def _extract_floats(text):
    nums = re.findall(r"\d+\.?\d*", text.replace(",", ""))
    return [float(n) for n in nums]


def get_monthly_average_advanced(year, month, headless=True, timeout=30000):
    """
    ดึงราคาดีเซลด้วย Browser Automation
    คืนค่า: float (ค่าเฉลี่ยรายเดือน)
    """
    url = "https://www.bangchak.co.th/th/oilprice/historical"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()

        try:
            page.goto(url, timeout=timeout)
            page.wait_for_load_state("networkidle")
            time.sleep(5)

            # ---- NOTE ----
            # ตรงนี้ต้อง fine-tune selector จากหน้าเว็บจริง
            # ผมจะทำรอบถัดไปให้ตรง 100%
            # ตอนนี้เป็นโครง robust
            # ----------------

            tables = page.query_selector_all("table")
            if not tables:
                raise BangchakAdvancedScraperError("ไม่พบตารางข้อมูล (JS/iframe)")

            text = tables[0].inner_text()

            # TODO: Part 2.1 (รอบถัดไป)
            # - parse วันที่
            # - filter เฉพาะดีเซล
            # - แยกวัน
            # - map day → price
            # - filter year/month
            # - คำนวณ average

            # ตอนนี้ยังไม่รู้โครง HTML จริง → แจ้ง fallback
            raise BangchakAdvancedScraperError(
                "Advanced parser ต้อง fine-tune selector กับหน้าเว็บจริง"
            )

        except Exception as e:
            raise BangchakAdvancedScraperError(str(e))

        finally:
            browser.close()
