from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:4173"
BROWSER = "/home/ubuntu/workspace/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell"

def set_range(page, selector, value):
    page.locator(selector).evaluate(
        "(e,v)=>{e.value=String(v);e.dispatchEvent(new Event('input',{bubbles:true}))}",
        value,
    )

def main():
    errors = []
    results = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=BROWSER, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda e: errors.append("PAGEERROR: " + str(e)))
        page.on("console", lambda m: errors.append("CONSOLE: " + m.text) if m.type == "error" else None)

        page.goto(URL, wait_until="networkidle")
        page.evaluate("localStorage.clear()")
        page.reload(wait_until="networkidle")
        assert page.locator(".demo-card").count() == 5
        page.screenshot(path="/home/ubuntu/workspace/neural-complete/validation-launcher.png", full_page=True)

        # 01 Boundary Foundry: weak representation fails, radial representation + enough data passes.
        page.evaluate("window.__NC5__.openDemo('boundary')")
        page.evaluate("window.__NC5_BOUNDARY__.train(1000)")
        page.evaluate("window.__NC5_BOUNDARY__.exam()")
        weak = page.evaluate("window.__NC5_BOUNDARY__.getState().hiddenAcc")
        assert weak is not None and weak < .95, weak
        page.select_option("#bf-mode", "radial")
        set_range(page, "#bf-samples", 120)
        page.evaluate("window.__NC5_BOUNDARY__.train(1400)")
        page.evaluate("window.__NC5_BOUNDARY__.exam()")
        strong = page.evaluate("window.__NC5_BOUNDARY__.getState().hiddenAcc")
        assert strong >= .95, strong
        assert page.locator("#bf-field").is_visible()
        results["boundary"] = {"weak_hidden": weak, "strong_hidden": strong}

        # 02 XOR Workshop: a linear hidden stack cannot solve XOR; nonlinear MLP can.
        page.evaluate("window.__NC5__.openDemo('xor')")
        set_range(page, "#xor-hidden", 4)
        page.select_option("#xor-act", "linear")
        page.evaluate("window.__NC5_XOR__.train(3000)")
        page.evaluate("window.__NC5_XOR__.exam()")
        linear = page.evaluate("window.__NC5_XOR__.getState().hiddenAcc")
        assert linear < .94, linear
        set_range(page, "#xor-hidden", 3)
        page.select_option("#xor-act", "tanh")
        page.evaluate("window.__NC5_XOR__.train(4000)")
        page.evaluate("window.__NC5_XOR__.exam()")
        nonlinear = page.evaluate("window.__NC5_XOR__.getState().hiddenAcc")
        assert nonlinear >= .94, nonlinear
        assert page.locator("#xor-hidden-bars .control-row").count() == 3
        results["xor"] = {"linear_hidden": linear, "nonlinear_hidden": nonlinear}

        # 03 Conv Forge: random filters fail; learned kernels generalize to unseen shifts/noise.
        page.evaluate("window.__NC5__.openDemo('conv')")
        page.evaluate("window.__NC5_CONV__.exam()")
        untrained_conv = page.evaluate("window.__NC5_CONV__.getState().hiddenAcc")
        assert untrained_conv < .95, untrained_conv
        set_range(page, "#conv-filters", 2)
        page.select_option("#conv-pool", "max")
        page.evaluate("window.__NC5_CONV__.train(1200)")
        page.evaluate("window.__NC5_CONV__.exam()")
        trained_conv = page.evaluate("window.__NC5_CONV__.getState().hiddenAcc")
        assert trained_conv >= .95, trained_conv
        assert page.locator('[id^="conv-kernel-"]').count() == 2
        assert page.locator('[id^="conv-map-"]').count() == 2
        results["conv"] = {"untrained_hidden": untrained_conv, "trained_hidden": trained_conv}

        # 04 Latent Vault: one-dimensional bottleneck underfits; 2D autoencoder passes compression target.
        page.evaluate("window.__NC5__.openDemo('latent')")
        set_range(page, "#ae-latent", 1)
        set_range(page, "#ae-lr", .08)
        page.evaluate("window.__NC5_LATENT__.train(6000)")
        page.evaluate("window.__NC5_LATENT__.exam()")
        one_d = page.evaluate("window.__NC5_LATENT__.getState().hiddenMse")
        one_solved = page.evaluate("window.__NC5_LATENT__.getState().solved")
        assert not one_solved
        assert one_d > .0025, one_d
        set_range(page, "#ae-latent", 2)
        page.evaluate("window.__NC5_LATENT__.train(6000)")
        page.evaluate("window.__NC5_LATENT__.exam()")
        two_d = page.evaluate("window.__NC5_LATENT__.getState().hiddenMse")
        two_solved = page.evaluate("window.__NC5_LATENT__.getState().solved")
        assert two_solved and two_d <= .0025, two_d
        results["latent"] = {"1d_hidden_mse": one_d, "2d_hidden_mse": two_d}

        # 05 Q-Lab: zero Q-table has no useful full-map policy; experience produces a compact greedy policy.
        page.evaluate("window.__NC5__.openDemo('qlearn')")
        page.evaluate("window.__NC5_QLEARN__.evaluate()")
        initial_q = page.evaluate("window.__NC5_QLEARN__.getState().eval")
        assert initial_q["success"] < .9 or initial_q["avgSteps"] > 18
        page.evaluate("window.__NC5_QLEARN__.train(5000)")
        page.evaluate("window.__NC5_QLEARN__.evaluate()")
        learned_q = page.evaluate("window.__NC5_QLEARN__.getState().eval")
        assert learned_q["success"] >= .9 and learned_q["avgSteps"] <= 18, learned_q
        assert page.locator("#q-grid .grid-cell").count() == 36
        page.screenshot(path="/home/ubuntu/workspace/neural-complete/validation-qlearn.png", full_page=True)
        results["qlearn"] = {"initial": initial_q, "learned": learned_q}

        # Completion/persistence: all five are recorded and survive reload.
        page.evaluate("window.__NC5__.home()")
        solved_cards = page.locator(".demo-card .index").all_inner_texts()
        assert sum("SOLVED" in x for x in solved_cards) == 5, solved_cards
        page.reload(wait_until="networkidle")
        solved_cards = page.locator(".demo-card .index").all_inner_texts()
        assert sum("SOLVED" in x for x in solved_cards) == 5

        # Responsive smoke checks.
        for width, height in [(1180, 820), (980, 760), (700, 900)]:
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_timeout(100)
            assert page.locator(".topbar").is_visible()
            assert page.locator(".demo-card").count() == 5

        print({"results": results, "errors": errors})
        assert not errors, errors
        browser.close()

if __name__ == "__main__":
    main()
