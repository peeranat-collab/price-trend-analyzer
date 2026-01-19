from playwright.sync_api import sync_playwright
import json
import os

SESSION_FILE = "scrapers/tpia_session.json"
LOGIN_URL = "https://www.tpia.org/member_login/"

def login_and_save_session():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        print("🔐 Opening login page...")
        page.goto(LOGIN_URL)

        print("👉 กรุณา Login ด้วยตัวเอง + ติ๊ก CAPTCHA")
        print("⏳ รอจนคุณ Login เสร็จ และเห็น Dashboard")

        # รอให้คุณ login เอง
        page.wait_for_timeout(120_000)  # 2 นาที

        # Save cookies
        cookies = context.cookies()
        storage = context.storage_state()

        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(storage, f, indent=2)

        print("✅ Session saved:", SESSION_FILE)

        browser.close()


def load_session_context(p):
    if not os.path.exists(SESSION_FILE):
        return None

    return p.chromium.new_context(storage_state=SESSION_FILE)
