from playwright.sync_api import sync_playwright
from datetime import datetime
import re
import time

class BangchakAdvancedScraperError(Exception):
    pass


def _to_float(text):
    nums = re.findall(r"\d+\.?\d*", text.replace(",", ""))
    return float(nums[0]) if nums else None


def get_monthly_average_advanced(year, month, headless=True, timeout=45000):
    """
    ดึงราคาน้ำมัน 'ไฮดีเซล (Hi Diesel)' จาก Bangchak ด้วย Playwright
    แล้วคำนวณค่าเฉลี่ยทั้งเดือน (บาท/ลิตร)

    คืนค่า: float
    ถ้าพัง: raise BangchakAdvancedScraperError
    """

    url = "https://www.bangchak.co.th/th/oilprice/historical"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()

        try:
            page.goto(url, timeout=timeout)
            page.wait_for_load_state("networkidle")
            time.sleep(4)  # เผื่อ JS render

            # ====== จุดสำคัญ: จับตารางราคา ======
            # จาก DOM ที่คุณส่งมา: มี <table> จริง
            tables = page.query_selector_all("table")
            if not tables:
                raise BangchakAdvancedScraperError("ไม่พบตารางข้อมูล")

            # ใช้ตารางแรกเป็นหลัก (ถ้ามีหลายตาราง จะ fine-tune ต่อได้)
            table = tables[0]

            rows = table.query_selector_all("tbody tr")
            if not rows:
                raise BangchakAdvancedScraperError("ไม่พบแถวข้อมูลในตาราง")

            prices = []

            for r in rows:
                cols = r.query_selector_all("td")
                if len(cols) < 3:
                    continue

                # โครงสร้างที่พบบ่อย:
                # col[0] = วันที่
                # col[1] = ไฮพรีเมียม
                # col[2] = ไฮดีเซล (Hi Diesel)  ← เป้าหมายเรา
                date_text = cols[0].inner_text().strip()
                hi_diesel_text = cols[2].inner_text().strip()

                try:
                    # วันที่เว็บไทยมักเป็น พ.ศ. เช่น 09/01/2569
                    # แปลง พ.ศ. → ค.ศ.
                    d = datetime.strptime(date_text, "%d/%m/%Y")
                except:
                    # ถ้าเป็น พ.ศ. (เช่น 2569) ให้ลบ 543
                    try:
                        parts = date_text.split("/")
                        if len(parts) == 3:
                            d = datetime(int(parts[2]) - 543, int(parts[1]), int(parts[0]))
                        else:
                            continue
                    except:
                        continue

                if d.year == year and d.month == month:
                    price = _to_float(hi_diesel_text)
                    if price is not None:
                        prices.append(price)

            if not prices:
                raise BangchakAdvancedScraperError(
                    f"ไม่พบข้อมูลไฮดีเซลสำหรับ {month}/{year}"
                )

            avg = round(sum(prices) / len(prices), 2)
            return avg

        except Exception as e:
            raise BangchakAdvancedScraperError(str(e))

        finally:
            browser.close()
