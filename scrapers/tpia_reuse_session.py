from playwright.sync_api import sync_playwright
import os
import json

SESSION_FILE = "scrapers/tpia_session.json"
PET_URL = "https://www.tpia.org/plastic-price-report-member/"

def open_pet_page_with_session():
    if not os.path.exists(SESSION_FILE):
        raise Exception("❌ ไม่พบ session file — กรุณา login ก่อน (Phase P1.1)")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(storage_state=SESSION_FILE)
        page = context.new_page()

        print("♻️ Reusing saved session...")
        page.goto(PET_URL)

        print("⏳ เปิดหน้า PET แล้ว — ตรวจสอบว่าคุณเห็นข้อมูลหรือไม่")
        print("👉 ปิด browser เองเมื่อทดสอบเสร็จ")

        page.wait_for_timeout(120_000)  # เปิดทิ้งไว้ 2 นาที
