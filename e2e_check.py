from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:4173"
BROWSER = "/home/ubuntu/workspace/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell"

def reset(page, demo):
    page.evaluate("(d)=>localStorage.removeItem(\"nc90:\"+d)", demo)
    page.evaluate(f"window.__NC90__.openDemo('{demo}')")
    page.wait_for_timeout(100)

def main():
    errors = []
    results = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=BROWSER, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("pageerror", lambda e: errors.append("PAGEERROR: " + str(e)))
        page.on("console", lambda m: errors.append("CONSOLE: " + m.text) if m.type == "error" else None)
        page.goto(URL, wait_until="networkidle")
        page.evaluate("localStorage.clear()")
        page.reload(wait_until="networkidle")

        assert page.locator(".launch-card").count() == 5

        # XOR — visible UI only: build x1*x2 -> logistic output.
        reset(page, "xor-lab")
        page.locator('.part-brick[data-part="1"]').dblclick()
        op = page.locator(".xor-node.op").get_attribute("data-node")
        page.locator('[data-node="x1"] .port.out').click()
        page.locator(f'[data-node="{op}"] .port.in').click()
        page.locator('[data-node="x2"] .port.out').click()
        page.locator(f'[data-node="{op}"] .port.in').click()
        page.locator(f'[data-node="{op}"] .port.out').click()
        page.locator('[data-node="out"] .port.in').click()
        page.locator("#xor-train").click()
        page.locator("#xor-exam").click()
        assert "PASS" in page.locator("#xor-hidden-score").inner_text()
        results["xor"] = page.locator("#xor-hidden-score").inner_text()

        # Persistence: the player-built graph survives reload.
        page.reload(wait_until="networkidle")
        assert page.locator(".xor-node.op").count() == 1
        assert page.locator(".xor-wire").count() == 3

        # Feature Foundry — visible UI only: forge x1^2 + x2^2.
        reset(page, "feature-foundry")
        page.locator('.machine[data-op="square"]').click()
        f1 = page.locator("#ff-shelf .feature-card").nth(0)
        page.locator('#ff-raw .feature-card[data-feature="raw-x2"]').drag_to(page.locator('.forge-slot[data-slot="A"]'))
        page.locator('.machine[data-op="square"]').click()
        f2 = page.locator("#ff-shelf .feature-card").nth(1)
        f1.drag_to(page.locator('.forge-slot[data-slot="A"]'))
        f2.drag_to(page.locator('.forge-slot[data-slot="B"]'))
        page.locator('.machine[data-op="add"]').click()
        page.locator("#ff-shelf .feature-card").nth(2).drag_to(page.locator('.classifier-dock[data-dock="0"]'))
        for _ in range(5):
            page.locator("#ff-train").click()
        page.locator("#ff-exam").click()
        assert float(page.locator("#ff-hidden").inner_text().rstrip("%")) >= 94
        results["feature"] = page.locator("#ff-hidden").inner_text()

        # Vision Forge — visible UI only: install a local filter and learn its pixels.
        reset(page, "vision-forge")
        page.locator('.parts-drawer button[data-size="3"]').drag_to(page.locator("#vf-drop"))
        for _ in range(4):
            page.locator("#vf-train").click()
        page.locator("#vf-exam").click()
        assert float(page.locator("#vf-hidden").inner_text().rstrip("%")) >= 95
        assert page.locator('[id^="kernel-"]').count() == 1
        assert page.locator('[id^="map-"]').count() == 1
        results["vision"] = page.locator("#vf-hidden").inner_text()

        # Latent Cartographer — visible UI only: create and wire two latent channels.
        reset(page, "latent-cartographer")
        page.locator("#lc-add").click()
        page.locator("#lc-all").click()
        page.locator("#lc-add").click()
        page.locator("#lc-all").click()
        for _ in range(12):
            page.locator("#lc-train").click()
        page.locator("#lc-exam").click()
        hidden_mse = float(page.locator("#lc-hidden").inner_text())
        assert hidden_mse <= 0.003
        results["latent"] = hidden_mse

        # Policy Garden — visible UI only: build a state representation, then learn.
        reset(page, "policy-garden")
        page.locator('.sensor-pack[data-sensor="row"]').drag_to(page.locator('.brain-slot[data-slot="0"]'))
        page.locator('.sensor-pack[data-sensor="col"]').drag_to(page.locator('.brain-slot[data-slot="1"]'))
        for _ in range(4):
            page.locator("#pg-train").click()
        page.locator("#pg-eval").click()
        success = float(page.locator("#pg-success").inner_text().rstrip("%"))
        steps = float(page.locator("#pg-steps").inner_text())
        assert success >= 90 and steps <= 18
        results["policy"] = {"success": success, "steps": steps}

        # Completion cards survive navigation/reload.
        page.evaluate("window.__NC90__.home()")
        assert sum("SOLVED" in x for x in page.locator(".card-num").all_inner_texts()) == 5
        page.reload(wait_until="networkidle")
        assert sum("SOLVED" in x for x in page.locator(".card-num").all_inner_texts()) == 5

        # Narrow desktop smoke.
        for width, height in [(1280, 820), (1024, 760), (900, 760)]:
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_timeout(80)
            assert page.locator(".launcher").is_visible()

        print({"results": results, "errors": errors})
        assert not errors, errors
        browser.close()

if __name__ == "__main__":
    main()
