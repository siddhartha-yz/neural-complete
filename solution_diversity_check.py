from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:4173"
BROWSER = "/home/ubuntu/workspace/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell"

def clean(page, key, demo):
    page.evaluate(f"localStorage.removeItem('nc90:{key}')")
    page.evaluate(f"window.__NC90__.openDemo('{demo}')")

def check_xor(page):
    out = {}
    for name, kind in [("multiply", "mul"), ("tanh2", "tanh"), ("relu3", "relu")]:
        clean(page, "xor-lab", "xor-lab")
        if kind == "mul":
            mid = page.evaluate("window.__NC90_XOR__.addPart({type:'op',op:'mul',label:'MULTIPLY'},350,220).id")
            page.evaluate("(id)=>{const a=window.__NC90_XOR__;a.connect('x1',id);a.connect('x2',id);a.connect(id,'out');a.graph.reinitialize(4);a.train(600);a.exam()}", mid)
        else:
            units, epochs = (2, 1200) if kind == "tanh" else (3, 1800)
            ids = page.evaluate("(u)=>Array.from({length:u},(_,i)=>window.__NC90_XOR__.addPart({type:'neuron',label:'NEURON'},320,100+i*120).id)", units)
            page.evaluate("""([ids,act,epochs])=>{const a=window.__NC90_XOR__;for(const id of ids){a.graph.node(id).activation=act;a.connect('x1',id);a.connect('x2',id);a.connect(id,'out')}a.graph.reinitialize(4);a.train(epochs);a.exam()}""", [ids, kind, epochs])
        score = page.evaluate("window.__NC90_XOR__.getState().state.hidden")
        assert score >= .94, (name, score)
        out[name] = score
    return out

def feature_recipe(page, recipe):
    clean(page, "feature-foundry", "feature-foundry")
    def forge(a,b,op):
        page.evaluate("([a,b])=>window.__NC90_FEATURE__.setForge(a,b)", [a,b])
        page.evaluate("(op)=>window.__NC90_FEATURE__.forge(op)", op)
        return page.evaluate("window.__NC90_FEATURE__.state.features.at(-1).id")
    if recipe == "radius":
        a=forge("raw-x1","raw-x2","square"); b=forge("raw-x2","raw-x1","square"); c=forge(a,b,"add"); ids=[c]
    elif recipe == "separate":
        a=forge("raw-x1","raw-x2","square"); b=forge("raw-x2","raw-x1","square"); ids=[a,b]
    else:
        a=forge("raw-x1","raw-x2","abs"); b=forge("raw-x2","raw-x1","abs"); c=forge(a,b,"add"); ids=[c]
    for i,fid in enumerate(ids): page.evaluate("([i,id])=>window.__NC90_FEATURE__.dock(i,id)", [i,fid])
    page.evaluate("window.__NC90_FEATURE__.train(1200);window.__NC90_FEATURE__.exam()")
    score=page.evaluate("window.__NC90_FEATURE__.state.hidden")
    assert score >= .94, (recipe, score)
    return score

def vision_arch(page, name):
    clean(page, "vision-forge", "vision-forge")
    configs={
        "3max":[(3,"max")],
        "3top2":[(3,"top2")],
        "3max+5max":[(3,"max"),(5,"max")],
    }
    epochs={"3max":1800,"3top2":4000,"3max+5max":2500}[name]
    for size,red in configs[name]:
        fid=page.evaluate("(s)=>window.__NC90_VISION__.addFilter(s).id", size)
        page.evaluate("([id,r])=>window.__NC90_VISION__.setReducer(id,r)", [fid,red])
    page.evaluate("(n)=>window.__NC90_VISION__.train(n)", epochs)
    page.evaluate("window.__NC90_VISION__.exam()")
    score=page.evaluate("window.__NC90_VISION__.getState().hidden")
    assert score >= .95, (name, score)
    return score

def latent_mask(page, name):
    clean(page, "latent-cartographer", "latent-cartographer")
    all_mask=[True]*36
    checker=[(i//6+i%6)%2==0 for i in range(36)]
    top=[i//6<3 for i in range(36)]
    configs={
        "dense":[all_mask,all_mask],
        "checker":[checker,[not x for x in checker]],
        "topbottom":[top,[not x for x in top]],
    }
    ids=[page.evaluate("window.__NC90_LATENT__.addChannel().id") for _ in range(2)]
    for cid,mask in zip(ids,configs[name]):
        page.evaluate("([id,m])=>window.__NC90_LATENT__.setMasks(id,m)", [cid,mask])
    epochs=6000 if name=="dense" else 9000
    page.evaluate("(n)=>window.__NC90_LATENT__.train(n)", epochs)
    page.evaluate("window.__NC90_LATENT__.exam()")
    mse=page.evaluate("window.__NC90_LATENT__.getState().hiddenMse")
    assert mse <= .003, (name,mse)
    return mse

def policy_repr(page, sensors):
    clean(page, "policy-garden", "policy-garden")
    for i,s in enumerate(sensors):
        page.evaluate("([i,s])=>window.__NC90_POLICY__.installSensor(i,s)", [i,s])
    page.evaluate("window.__NC90_POLICY__.train(5000)")
    ev=page.evaluate("window.__NC90_POLICY__.evaluate()")
    assert ev["success"] >= .9 and ev["avgSteps"] <= 18, (sensors,ev)
    return ev

def main():
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True, executable_path=BROWSER, args=["--no-sandbox"])
        page=browser.new_page(viewport={"width":1280,"height":820})
        page.on("pageerror", lambda e: errors.append("PAGEERROR: "+str(e)))
        page.on("console", lambda m: errors.append("CONSOLE: "+m.text) if m.type=="error" else None)
        page.goto(URL, wait_until="networkidle")
        page.evaluate("localStorage.clear()")
        results={
            "xor":check_xor(page),
            "feature":{r:feature_recipe(page,r) for r in ["radius","separate","l1"]},
            "vision":{r:vision_arch(page,r) for r in ["3max","3top2","3max+5max"]},
            "latent":{r:latent_mask(page,r) for r in ["dense","checker","topbottom"]},
            "policy":{
                "row+col":policy_repr(page,["row","col"]),
                "dx+dy":policy_repr(page,["dx","dy"]),
                "dir+walls+region":policy_repr(page,["goalDir","walls","region"]),
            },
        }
        # Known weak structures must really fail.
        clean(page,"policy-garden","policy-garden")
        blind=page.evaluate("window.__NC90_POLICY__.evaluate()")
        assert blind["success"] < .9 or blind["avgSteps"] > 18
        clean(page,"vision-forge","vision-forge")
        empty=page.evaluate("window.__NC90_VISION__.metrics()")
        assert empty["acc"] < .95
        print({"diversity":results,"errors":errors})
        assert not errors, errors
        browser.close()

if __name__=="__main__":
    main()
