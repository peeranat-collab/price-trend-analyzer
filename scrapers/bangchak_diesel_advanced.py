from playwright.sync_api import sync_playwright
from datetime import datetime
import pandas as pd
import time

class BangchakAdvancedScraperError(Exception):
    pass


def get_monthly_average_advanced(year, month, headless=True, timeout=20000):
    """
    ดึงราคาดีเซลจาก Bangchak ด้วย Browser Automation (Playwright)
    คืนค่า: float (ค่าเฉลี่ยรายเดือน) หรือ raise error
    """

    url = "https://www.bangchak.co.th/th/oilprice/historical"

    prices = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()

        try:
            page.goto(url, timeout=timeout)
            page.wait_for_load_state("networkidle")

            # รอหน้าเว็บ render (เพราะใช้ JS)
            time.sleep(5)

            # ===== TODO: selector ตรงนี้จะ fine-tune ใน Part 2 =====
            # ตอนนี้เราจะ dump ตารางทั้งหมดก่อน
            tables = page.query_selector_all("table")

            if not tables:
                raise BangchakAdvancedScraperError("ไม่พบตารางข้อมูล")

            # ดึง text ทั้งหมดจากตาราง
            raw_text = tables[0].inner_text()

            # === Placeholder logic ===
            # ใน Part 2 ผมจะ:
            # - parse วันที่
            # - filter เฉพาะดีเซล
            # - แยกวัน
            # - แปลงเป็น float
            # - เลือกเฉพาะเดือน/ปี
            # - คำนวณค่าเฉลี่ยจริง

            raise BangchakAdvancedScraperError(
                "Advanced parser ยังไม่ถูก fine-tune (จะทำใน Part 2)"
            )

        except Exception as e:
            raise BangchakAdvancedScraperError(str(e))

        finally:
            browser.close()
