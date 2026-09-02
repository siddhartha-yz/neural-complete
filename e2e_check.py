from pathlib import Path
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:4173"
BROWSER = "/home/ubuntu/workspace/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell"


def wire(page, source_node, source_port, target_node, target_port):
    page.locator(f'.port[data-node="{source_node}"][data-port="{source_port}"][data-kind="output"]').click()
    page.locator(f'.port[data-node="{target_node}"][data-port="{target_port}"][data-kind="input"]').click()


def set_const(page, node_id, value):
    page.locator(f'.node[data-node="{node_id}"]').click()
    page.locator('.tab[data-tab="inspect"]').click()
    field = page.locator("#constInput")
    field.fill(str(value))
    field.press("Enter")


def main():
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            executable_path=BROWSER,
            args=["--no-sandbox"],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda e: errors.append(f"PAGEERROR: {e}"))
        page.on(
            "console",
            lambda m: errors.append(f"CONSOLE {m.type}: {m.text}")
            if m.type == "error"
            else None,
        )
        page.goto(URL, wait_until="networkidle")
        page.evaluate("localStorage.clear()")
        page.reload(wait_until="networkidle")

        assert page.locator("#intro").is_visible()
        page.locator("#enterBtn").click()
        assert page.locator(".node").count() == 3
        assert page.locator(".part").count() == 4
        assert page.locator("#verifyBtn").is_disabled()

        # Failure-path check: an empty board must fail and eventually unlock the hint.
        page.locator("#runAllBtn").click()
        page.wait_for_function("!document.querySelector('#runAllBtn').disabled")
        assert "输出" in page.locator("#simMessage").inner_text()
        page.locator("#runAllBtn").click()
        page.wait_for_function("!document.querySelector('#runAllBtn').disabled")
        assert page.locator("#hintBtn").is_enabled()

        board = page.locator("#board")
        placements = [
            ("const", 220, 120), ("const", 220, 300), ("const", 390, 410),
            ("mul", 390, 100), ("mul", 390, 270),
            ("add", 550, 180), ("add", 650, 350),
            ("step", 760, 300),
        ]
        for part_type, x, y in placements:
            page.locator(f'.part[data-type="{part_type}"]').drag_to(
                board, target_position={"x": x, "y": y}
            )

        ids = page.evaluate("""() => {
          const s = window.__NC__.state.nodes;
          const by = t => s.filter(n => n.type === t).map(n => n.id);
          return {consts:by('const'), muls:by('mul'), adds:by('add'), steps:by('step')};
        }""")
        c1, c2, c3 = ids["consts"]
        m1, m2 = ids["muls"]
        a1, a2 = ids["adds"]
        step = ids["steps"][0]

        set_const(page, c1, 1.5)
        set_const(page, c2, -1)
        set_const(page, c3, -0.2)
        page.locator('.tab[data-tab="tests"]').click()

        wire(page, "input-x1", "out", m1, "a")
        wire(page, c1, "out", m1, "b")
        wire(page, "input-x2", "out", m2, "a")
        wire(page, c2, "out", m2, "b")
        wire(page, m1, "out", a1, "a")
        wire(page, m2, "out", a1, "b")
        wire(page, a1, "out", a2, "a")
        wire(page, c3, "out", a2, "b")
        wire(page, a2, "out", step, "x")
        wire(page, step, "out", "output-y", "in")

        assert page.locator(".wire").count() == 10

        # Inspect a live case: values should propagate onto nodes and wires.
        page.locator('.test-row[data-case="5"]').click()
        assert page.locator("#ioY").inner_text() == "1"
        assert "PASS" in page.locator("#simMessage").inner_text()
        assert page.locator(".wire.signal-pos, .wire.signal-neg, .wire.signal-zero").count() > 0

        page.locator("#runAllBtn").click()
        page.wait_for_function("!document.querySelector('#runAllBtn').disabled")
        assert page.locator("#testSummary").inner_text() == "6 / 6"
        assert page.locator("#verifyBtn").is_enabled()

        page.locator("#verifyBtn").click()
        page.locator("#victory").wait_for(state="visible", timeout=5000)
        assert "4 / 4" in page.locator("#verifyTitle").inner_text()

        page.locator("#compileBtn").click()
        assert page.locator("#compiledShelf").is_visible()
        assert "NEURON_01" in page.locator("#compiledShelf").inner_text()
        page.screenshot(
            path="/home/ubuntu/workspace/neural-complete/rebuild-solved.png",
            full_page=True,
        )

        # Progressive-abstraction check: clear the low-level circuit, drag the compiled
        # component back in, and solve the public contract with only three wires.
        page.once("dialog", lambda d: d.accept())
        page.locator("#resetBtn").click()
        assert page.locator(".node:not(.fixed)").count() == 0
        assert page.locator("#compiledShelf").is_visible()

        page.locator(".compiled-chip").drag_to(
            board, target_position={"x": 480, "y": 260}
        )
        macro_id = page.evaluate(
            "() => window.__NC__.state.nodes.find(n => n.type === 'neuron').id"
        )
        wire(page, "input-x1", "out", macro_id, "x1")
        wire(page, "input-x2", "out", macro_id, "x2")
        wire(page, macro_id, "out", "output-y", "in")

        page.locator("#runAllBtn").click()
        page.wait_for_function("!document.querySelector('#runAllBtn').disabled")
        assert page.locator("#testSummary").inner_text() == "6 / 6"
        assert page.locator(".node:not(.fixed)").count() == 1
        assert page.locator(".wire").count() == 3
        assert page.locator("#verifyBtn").is_enabled()

        page.locator("#verifyBtn").click()
        page.locator("#victory").wait_for(state="visible", timeout=5000)
        assert "4 / 4" in page.locator("#verifyTitle").inner_text()
        page.locator("#stayBtn").click()
        page.screenshot(
            path="/home/ubuntu/workspace/neural-complete/rebuild-macro-reuse.png",
            full_page=True,
        )

        # Save-state check: the compiled definition, macro instance, and wiring survive reload.
        page.reload(wait_until="networkidle")
        assert not page.locator("#intro").is_visible()
        assert page.locator("#compiledShelf").is_visible()
        assert page.locator('.node[data-node^="neuron-"]').count() == 1
        assert page.locator(".wire").count() == 3
        page.locator("#runAllBtn").click()
        page.wait_for_function("!document.querySelector('#runAllBtn').disabled")
        assert page.locator("#testSummary").inner_text() == "6 / 6"

        result = {
            "visible": page.locator("#testSummary").inner_text(),
            "verify": "4 / 4 macro reuse",
            "components": page.locator(".node:not(.fixed)").count(),
            "wires": page.locator(".wire").count(),
            "compiled": page.locator("#compiledShelf").is_visible(),
            "macro_reuse": True,
            "persistence": True,
            "errors": errors,
        }
        print(result)
        assert not errors, errors
        browser.close()


if __name__ == "__main__":
    main()
