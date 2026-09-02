from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:4173"
BROWSER = "/home/ubuntu/workspace/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell"

def js(page, code, arg=None):
    return page.evaluate(code, arg) if arg is not None else page.evaluate(code)

def main():
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True, executable_path=BROWSER, args=["--no-sandbox"])
        page=browser.new_page(viewport={"width":1280,"height":820})
        page.on("pageerror", lambda e: errors.append(f"PAGEERROR: {e}"))
        page.on("console", lambda m: errors.append(f"CONSOLE {m.type}: {m.text}") if m.type=="error" else None)
        page.goto(URL, wait_until="networkidle")
        page.evaluate("localStorage.clear()")
        page.reload(wait_until="networkidle")
        page.locator("#enterBtn").click()

        # Cycle diagnostics: construct an intentionally invalid feed-forward graph.
        ids=page.evaluate("""() => {
          const api=window.__NC__;
          const c=api.addPart('const',180,120).id;
          const a1=api.addPart('add',360,120).id;
          const a2=api.addPart('add',540,220).id;
          api.connect({node:c,port:'out'},{node:a1,port:'b'});
          api.connect({node:c,port:'out'},{node:a2,port:'b'});
          api.connect({node:a1,port:'out'},{node:a2,port:'a'});
          api.connect({node:a2,port:'out'},{node:a1,port:'a'});
          api.connect({node:a1,port:'out'},{node:'output-y',port:'in'});
          return {c,a1,a2};
        }""")
        page.locator('.test-row[data-case="0"]').click()
        msg=page.locator("#simMessage").inner_text()
        assert "回路" in msg, msg

        # Rewiring must replace an existing incoming wire rather than fan two sources into one input.
        page.evaluate("""() => window.__NC__.resetBoard(false)""")
        rewire=page.evaluate("""() => {
          const api=window.__NC__;
          const c1=api.addPart('const',220,120).id;
          const c2=api.addPart('const',220,260).id;
          const a=api.addPart('add',460,180).id;
          api.connect({node:c1,port:'out'},{node:a,port:'a'});
          const before=api.state.wires.length;
          api.connect({node:c2,port:'out'},{node:a,port:'a'});
          const incoming=api.state.wires.filter(w=>w.to.node===a&&w.to.port==='a');
          return {c1,c2,a,before,after:api.state.wires.length,incoming};
        }""")
        assert rewire["before"]==1
        assert rewire["after"]==1
        assert len(rewire["incoming"])==1 and rewire["incoming"][0]["from"]["node"]==rewire["c2"]

        # Esc cancels a partially-started wire.
        page.locator(f'.port[data-node="{rewire["c1"]}"][data-port="out"][data-kind="output"]').click()
        assert page.locator("#wireCancelBtn").is_visible()
        page.keyboard.press("Escape")
        assert not page.locator("#wireCancelBtn").is_visible()

        # Delete selected wire then Ctrl+Z must restore it and persistence must track the restored state.
        wire_id=page.evaluate("() => window.__NC__.state.wires[0].id")
        page.locator(f'.wire[data-wire="{wire_id}"]').click()
        page.keyboard.press("Delete")
        assert page.locator(".wire").count()==0
        page.keyboard.press("Control+z")
        assert page.locator(".wire").count()==1
        page.reload(wait_until="networkidle")
        assert page.locator(".wire").count()==1

        # Responsive fixed terminals stay inside board after viewport changes.
        for width,height in [(1440,1000),(1180,820),(1100,780)]:
            page.set_viewport_size({"width":width,"height":height})
            page.wait_for_timeout(80)
            board=page.locator("#board").bounding_box()
            assert board and board["width"]>0 and board["height"]>0
            for node_id in ["input-x1","input-x2","output-y"]:
                box=page.locator(f'.node[data-node="{node_id}"]').bounding_box()
                assert box, node_id
                assert box["x"] >= board["x"]-1
                assert box["y"] >= board["y"]-1
                assert box["x"]+box["width"] <= board["x"]+board["width"]+1, (width,node_id,box,board)
                assert box["y"]+box["height"] <= board["y"]+board["height"]+1, (height,node_id,box,board)

        print({
            "cycle_diagnostic": True,
            "rewire_replace": True,
            "escape_cancel": True,
            "undo_persistent": True,
            "responsive": True,
            "errors": errors,
        })
        assert not errors, errors
        browser.close()

if __name__=="__main__":
    main()
