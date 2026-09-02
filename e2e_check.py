
from playwright.sync_api import sync_playwright
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True, executable_path="/home/ubuntu/workspace/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell", args=["--no-sandbox"])
    page=browser.new_page(viewport={"width":1440,"height":1000})
    page.on("pageerror", lambda e: errors.append("PAGEERROR: "+str(e)))
    page.goto("http://127.0.0.1:4173", wait_until="networkidle")
    assert page.title().startswith("Neural Complete")
    page.click("#startBtn")
    for name in ["Linear 2→1","Sigmoid","BCE Loss","SGD"]:
        page.get_by_text(name, exact=True).last.click()
    assert page.locator("#connectBtn").is_enabled()
    page.click("#connectBtn")
    assert "阶段 2" in page.locator("#stageLabel").inner_text()
    for _ in range(3):
        for sel in ["#forwardBtn","#lossBtn","#backwardBtn","#updateBtn"]:
            page.click(sel)
    assert page.locator("#autoBtn").is_enabled()
    for _ in range(3):
        page.click("#autoBtn")
    result={
      "acc":page.locator("#accValue").inner_text(),
      "loss":page.locator("#lossValue").inner_text(),
      "stage":page.locator("#stageLabel").inner_text(),
      "victory":page.locator("#victoryOverlay").is_visible(),
      "errors":errors,
    }
    page.screenshot(path="/home/ubuntu/workspace/neural-complete/first-level-validated.png", full_page=True)
    print(result)
    assert result["victory"] is True
    assert not errors
    browser.close()
