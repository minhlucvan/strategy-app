from playwright.sync_api import sync_playwright, Playwright
import requests
import json
import subprocess

def __get_auth_token(self):
    """Fetches the authentication token by connecting to the browser and extracting data from local storage."""
    chromium = self.playwright.chromium
    browser = chromium.connect_over_cdp('http://localhost:1243')
    context = browser.contexts[0]
    page = context.new_page()

    try:
        page.goto('https://tcinvest.tcbs.com.vn/')
        user_info_str = page.evaluate(
            "() => localStorage.getItem('userInfo')")
        self.configure(json.loads(user_info_str))
        print(f"Auth token: {self.auth_token}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        page.close()

    return self.auth_token

def launch_native_browser(self):
    """Launches the native browser with remote debugging enabled."""
    subprocess.run(
        ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--remote-debugging-port=1243"])


def init_auth(self):
    with sync_playwright() as playwright:
        playwright = playwright
        __get_auth_token()