import {LearningGraph} from "../engine.js";
import {seeded,randn,clamp,pct,fmt,save,load,uid,drawLine,canvasBox,resizeWatch,logPush,logHTML,deepCopy} from "../core.js";

const META={
  id:"xor-lab",theme:"violet",verb:"BUILD / WIRE / DEBUG",title:"异或构造实验室",en:"XOR Construction Lab",
  card:"从两个输入端开始，自己搭神经元、非线性和特征算子。系统只负责对你画出的 DAG 自动 forward / backward。",
  tags:["free graph","backprop","probe"]
};
function makeData(seed,n,noise=.17){
  const rng=seeded(seed),centers=[[-.72,-.72,0],[-.72,.72,1],[.72,-.72,1],[.72,.72,0]],out=[];
  for(let i=0;i<n;i++){const c=centers[i%4];out.push({x1:clamp(c[0]+randn(rng)*noise,-.98,.98),x2:clamp(c[1]+randn(rng)*noise,-.98,.98),label:c[2]})}
  return out;
}
const TRAIN=makeData(32117,160,.16),HIDDEN=makeData(98173,260,.2);
const PARTS=[
  {type:"neuron",label:"NEURON",symbol:"N",note:"trainable weighted sum"},
  {type:"op",op:"mul",label:"MULTIPLY",symbol:"×",note:"engineered interaction"},
  {type:"op",op:"add",label:"ADD",symbol:"+",note:"merge signals"},
  {type:"op",op:"square",label:"SQUARE",symbol:"x²",note:"nonlinear feature"},
  {type:"op",op:"abs",label:"ABS",symbol:"|x|",note:"fold a signal"}
];
function fresh(){
  return {graph:new LearningGraph().toJSON(),nodeSeq:0,edgeSeq:0,epoch:0,history:[],hidden:null,selected:"out",pending:null,probe:{x1:.72,x2:-.72,label:1},logs:[],solved:false};
}
function restored(){
  const s=load(META.id)||fresh();return{...fresh(),...s,graph:s.graph||fresh().graph};
}
function nodeLabel(n){
  if(n.id==="x1")return"x₁";if(n.id==="x2")return"x₂";if(n.id==="out")return"OUTPUT";
  if(n.type==="neuron")return(n.activation||"tanh").toUpperCase()+" NEURON";
  return(n.op||"op").toUpperCase();
}
function symbol(n){if(n.id==="x1")return"x₁";if(n.id==="x2")return"x₂";if(n.id==="out")return"σ";if(n.type==="neuron")return"●";return{mul:"×",add:"+",square:"²",abs:"| |"}[n.op]||"◇"}

export default {
  ...META,
  mount(root,ctx){
    let state=restored(),graph=new LearningGraph(state.graph),destroyResize=()=>{},dragging=null;
    root.innerHTML=`
      <section class="xor-screen">
        <header class="xor-top">
          <button class="back-chip" id="xor-back">← LABS</button>
          <div><span>EXPERIMENT 01 · STRUCTURE FIRST</span><h1>异或构造实验室</h1></div>
          <div class="xor-target"><small>HIDDEN TARGET</small><b id="xor-hidden-score">NOT TESTED</b></div>
          <button class="reset-chip" id="xor-reset">RESET</button>
        </header>
        <div class="xor-body">
          <aside class="xor-parts">
            <span class="rail-label">PARTS CRATE</span>
            <p>拖入画布。正确网络没有被预先列成选项。</p>
            <div id="xor-palette"></div>
            <div class="xor-help"><b>接线</b><span>点源节点右侧端口 → 点目标节点左侧端口</span><b>调试</b><span>点节点查看 activation / gradient，Del 删除。</span></div>
          </aside>
          <div class="xor-canvas-wrap">
            <div class="xor-board" id="xor-board" tabindex="0">
              <svg id="xor-wires"></svg>
              <div id="xor-nodes"></div>
              <div class="xor-board-note"><b>BUILD A MODEL</b><span>输入与输出已经固定。中间结构由你决定。</span></div>
              <div class="xor-probe-field"><canvas id="xor-field"></canvas><span>click to move probe</span></div>
            </div>
          </div>
          <aside class="xor-inspector">
            <span class="rail-label">OSCILLOSCOPE</span>
            <div id="xor-inspect"></div>
          </aside>
        </div>
        <footer class="xor-console">
          <div class="xor-run">
            <button id="xor-step">STEP 1 EPOCH</button>
            <button class="hot" id="xor-train">RUN 600 EPOCHS</button>
            <button id="xor-exam">HIDDEN EVALUATION</button>
          </div>
          <div class="xor-meters">
            <div><span>epoch</span><b id="xor-epoch">0</b></div>
            <div><span>train loss</span><b id="xor-loss-v">—</b></div>
            <div><span>train acc</span><b id="xor-acc">—</b></div>
            <div><span>parameters</span><b id="xor-params">—</b></div>
          </div>
          <canvas id="xor-curve"></canvas>
          <div class="xor-log" id="xor-log"></div>
        </footer>
      </section>`;

    const board=document.getElementById("xor-board"),nodesEl=document.getElementById("xor-nodes"),svg=document.getElementById("xor-wires"),field=document.getElementById("xor-field"),curve=document.getElementById("xor-curve");

    function persist(){state.graph=graph.toJSON();save(META.id,state)}
    function invalidate(message="structure changed"){
      state.hidden=null;state.solved=false;logPush(state.logs,message,"warn");persist();
    }
    function addPart(part,x=360,y=220){
      const n={id:(part.type==="neuron"?"n":"op")+(++state.nodeSeq),type:part.type,label:part.label,x,y,fixed:false};
      if(part.type==="neuron"){n.activation="tanh";n.bias=0;n.weights={};n.frozen=false;n.muted=false}
      else n.op=part.op;
      graph.addNode(n);state.selected=n.id;invalidate("added "+nodeLabel(n));render();return n;
    }
    function removeSelected(){
      if(!state.selected)return;
      const n=graph.node(state.selected);if(n&&!n.fixed){graph.removeNode(n.id);state.selected=null;invalidate("node removed");render()}
    }
    function connect(from,to){
      const target=graph.node(to);if(!target||target.type==="input")return;
      const max=(target.type==="op"&&(target.op==="square"||target.op==="abs"))?1:(target.type==="op"?2:4);
      const inc=graph.incoming(to);
      if(inc.length>=max){ctx.toast(nodeLabel(target)+" 输入已满","bad");return}
      const e=graph.addEdge(from,to,"e"+(++state.edgeSeq));if(e){state.pending=null;state.selected=to;invalidate("wire added");render()}
    }
    function renderPalette(){
      document.getElementById("xor-palette").innerHTML=PARTS.map((p,i)=>`<button class="part-brick" draggable="true" data-part="${i}"><b>${p.symbol}</b><span>${p.label}</span><small>${p.note}</small></button>`).join("");
      document.querySelectorAll(".part-brick").forEach(b=>{
        b.ondragstart=e=>e.dataTransfer.setData("text/x-part",b.dataset.part);
        b.ondblclick=()=>addPart(PARTS[+b.dataset.part],300+Math.random()*180,150+Math.random()*200);
      });
    }
    function portCenter(id,kind){
      const el=document.querySelector(`[data-node="${id}"] .port.${kind}`);if(!el)return null;const r=el.getBoundingClientRect(),b=board.getBoundingClientRect();return{x:r.left-b.left+r.width/2,y:r.top-b.top+r.height/2};
    }
    function wirePath(a,b){const dx=Math.max(35,Math.abs(b.x-a.x)*.42);return`M${a.x},${a.y} C${a.x+dx},${a.y} ${b.x-dx},${b.y} ${b.x},${b.y}`}
    function renderWires(){
      svg.innerHTML="";
      const probe=graph.gradients(state.probe);
      for(const e of graph.edges){
        const a=portCenter(e.from,"out"),b=portCenter(e.to,"in");if(!a||!b)continue;
        const path=document.createElementNS("http://www.w3.org/2000/svg","path");path.setAttribute("d",wirePath(a,b));path.classList.add("xor-wire");
        const target=graph.node(e.to),w=target?.weights?.[e.id];if(Number.isFinite(w))path.classList.add(w>=0?"positive":"negative");
        path.dataset.edge=e.id;path.onclick=ev=>{ev.stopPropagation();graph.removeEdge(e.id);invalidate("wire removed");render()};svg.appendChild(path);
      }
      if(state.pending){const a=portCenter(state.pending,"out");if(a){const p=document.createElementNS("http://www.w3.org/2000/svg","path");p.id="xor-ghost";p.setAttribute("d",wirePath(a,a));p.classList.add("xor-wire","ghost");svg.appendChild(p)}}
    }
    function renderNodes(){
      nodesEl.innerHTML="";
      const probe=graph.gradients(state.probe),diag=graph.diagnostics(TRAIN);
      for(const n of graph.nodes){
        const el=document.createElement("div");el.className="xor-node "+n.type+(state.selected===n.id?" selected":"")+(n.muted?" muted":"");
        el.dataset.node=n.id;el.style.left=n.x+"px";el.style.top=n.y+"px";
        const pv=probe.ok?probe.values[n.id]:null,dg=diag[n.id]?.gradient||0;
        el.innerHTML=`<button class="port in" title="input"></button><div class="node-head"><span>${nodeLabel(n)}</span><b>${symbol(n)}</b></div><div class="node-scope"><strong>${fmt(pv,3)}</strong><small>∇ ${fmt(dg,3)}</small></div><button class="port out" title="output"></button>`;
        el.onclick=e=>{if(e.target.classList.contains("port"))return;state.selected=n.id;render()};
        const out=el.querySelector(".port.out");out.onclick=e=>{e.stopPropagation();state.pending=n.id;render()};
        const inp=el.querySelector(".port.in");inp.onclick=e=>{e.stopPropagation();if(state.pending&&state.pending!==n.id)connect(state.pending,n.id)};
        el.querySelector(".node-head").onpointerdown=e=>startDrag(e,n);
        nodesEl.appendChild(el);
      }
      document.querySelector(".xor-board-note").classList.toggle("hidden",graph.nodes.length>3);
    }
    function startDrag(e,n){
      if(n.fixed)return;e.preventDefault();const br=board.getBoundingClientRect(),sx=e.clientX,sy=e.clientY,ox=n.x,oy=n.y;
      const move=ev=>{n.x=clamp(ox+ev.clientX-sx,10,br.width-130);n.y=clamp(oy+ev.clientY-sy,10,br.height-90);const el=document.querySelector(`[data-node="${n.id}"]`);if(el){el.style.left=n.x+"px";el.style.top=n.y+"px"}renderWires()};
      const up=()=>{window.removeEventListener("pointermove",move);window.removeEventListener("pointerup",up);persist()};
      window.addEventListener("pointermove",move);window.addEventListener("pointerup",up);
    }
    function inspect(){
      const box=document.getElementById("xor-inspect"),n=graph.node(state.selected||"out"),g=graph.gradients(state.probe),diag=graph.diagnostics(TRAIN),valid=graph.validate();
      if(!n){box.innerHTML='<div class="scope-empty">SELECT A NODE</div>';return}
      const inc=graph.incoming(n.id),d=diag[n.id]||{};
      let controls="";
      if(n.type==="neuron")controls=`<label>activation<select id="node-act"><option value="tanh">tanh</option><option value="relu">ReLU</option><option value="linear">linear</option><option value="sigmoid">sigmoid</option></select></label>
        <div class="toggle-row"><button id="node-freeze" class="${n.frozen?"on":""}">❄ FREEZE</button><button id="node-mute" class="${n.muted?"on":""}">⊘ MUTE</button></div>`;
      const weights=(n.type==="neuron"||n.type==="output")?inc.map(e=>`<div class="weight-row"><span>${nodeLabel(graph.node(e.from))}</span><b>${fmt(n.weights[e.id],4)}</b></div>`).join(""):"";
      box.innerHTML=`<div class="scope-title"><span>${nodeLabel(n)}</span><b>${symbol(n)}</b></div>
        <div class="scope-grid"><div><span>probe output</span><b>${g.ok?fmt(g.values[n.id],4):"ERR"}</b></div><div><span>|gradient|</span><b>${fmt(d.gradient,5)}</b></div><div><span>|activation| avg</span><b>${fmt(d.activation,4)}</b></div><div><span>dead fraction</span><b>${pct(d.deadFraction||0)}</b></div></div>
        ${controls}${weights?'<div class="scope-weights"><small>INCOMING WEIGHTS</small>'+weights+"</div>":""}
        <div class="scope-probe"><small>PROBE SAMPLE</small><b>x₁=${fmt(state.probe.x1,2)} · x₂=${fmt(state.probe.x2,2)} · y=${state.probe.label}</b><span>${g.ok?"p="+fmt(g.p,4)+" · loss="+fmt(g.loss,4):g.error}</span></div>
        <div class="scope-diagnosis ${valid.ok?"ok":"bad"}">${valid.ok?diagnose(n,d):valid.message}</div>
        ${!n.fixed?'<button class="delete-node" id="delete-node">DELETE NODE</button>':""}`;
      if(n.type==="neuron"){const sel=document.getElementById("node-act");sel.value=n.activation||"tanh";sel.onchange=()=>{n.activation=sel.value;invalidate("activation changed to "+sel.value);render()};document.getElementById("node-freeze").onclick=()=>{n.frozen=!n.frozen;persist();render()};document.getElementById("node-mute").onclick=()=>{n.muted=!n.muted;state.hidden=null;persist();render()}}
      if(!n.fixed)document.getElementById("delete-node").onclick=removeSelected;
    }
    function diagnose(n,d){
      if(n.type==="neuron"&&n.activation==="relu"&&d.deadFraction>.97)return"⚠ 这个 ReLU 在几乎所有训练样本上都为 0；上游梯度会被切断。";
      if((d.gradient||0)<1e-5&&state.epoch>20&&n.type!=="input")return"⚠ 这里的梯度几乎消失。继续训练可能不会改变这个节点。";
      if(n.muted)return"节点已静音。你正在做结构消融实验。";
      return"信号与梯度正常。改变结构后再比较这里的读数。";
    }
    function drawField(){
      const {ctx:wctx,w,h}=canvasBox(field),pad=12,res=24;wctx.fillStyle="#090c10";wctx.fillRect(0,0,w,h);
      for(let iy=0;iy<res;iy++)for(let ix=0;ix<res;ix++){
        const x=-1+2*(ix+.5)/res,y=1-2*(iy+.5)/res,f=graph.forward({x1:x,x2:y,label:0}),p=f.ok?f.p:.5;
        wctx.fillStyle=`rgba(${Math.round(70+120*p)},${Math.round(72+40*(1-p))},${Math.round(110+105*p)},.9)`;wctx.fillRect(ix*w/res,iy*h/res,w/res+1,h/res+1)
      }
      wctx.strokeStyle="#f4f0e9";wctx.lineWidth=1.2;const px=(state.probe.x1+1)/2*w,py=(1-state.probe.x2)/2*h;wctx.beginPath();wctx.arc(px,py,4.5,0,Math.PI*2);wctx.stroke();
    }
    function metrics(){return graph.metrics(TRAIN)}
    function train(n){
      const valid=graph.validate();if(!valid.ok){ctx.toast(valid.message,"bad");logPush(state.logs,valid.message,"err");render();return}
      let last;for(let i=0;i<n;i++){last=graph.trainEpoch(TRAIN,.12);if(!last.ok){ctx.toast(last.error,"bad");break}state.epoch++;if(state.epoch===1||state.epoch%5===0)state.history.push(last.loss)}
      if(state.history.length>280)state.history.splice(0,state.history.length-280);state.hidden=null;
      logPush(state.logs,`trained ${n} epochs · loss ${fmt(last?.loss,4)} · acc ${pct(last?.acc)}`,last?.acc>.93?"ok":"");persist();render()
    }
    function exam(){
      const m=graph.metrics(HIDDEN);if(!m.ok){ctx.toast(m.error,"bad");return}
      state.hidden=m.acc;const pass=m.acc>=.94;
      logPush(state.logs,`hidden noisy XOR ${pct(m.acc)} · ${pass?"PASS":"FAIL"}`,pass?"ok":"err");
      if(pass&&!state.solved){state.solved=true;ctx.complete()}persist();render()
    }
    function render(){
      graph.ensureParameters();renderNodes();requestAnimationFrame(renderWires);inspect();drawField();drawLine(curve,state.history,{color:"#b8a0e3",label:"BCE"});
      const m=metrics(),st=graph.structure();document.getElementById("xor-epoch").textContent=state.epoch;document.getElementById("xor-loss-v").textContent=m.ok?fmt(m.loss,4):"INVALID";document.getElementById("xor-acc").textContent=m.ok?pct(m.acc):"—";document.getElementById("xor-params").textContent=st.params;
      document.getElementById("xor-hidden-score").textContent=state.hidden===null?"NOT TESTED":pct(state.hidden)+(state.hidden>=.94?" · PASS":" · FAIL");document.getElementById("xor-hidden-score").className=state.hidden!==null&&state.hidden>=.94?"pass":"";
      document.getElementById("xor-log").innerHTML=logHTML(state.logs);
    }

    board.ondragover=e=>{if(e.dataTransfer.types.includes("text/x-part"))e.preventDefault()};
    board.ondrop=e=>{const i=e.dataTransfer.getData("text/x-part");if(i==="")return;e.preventDefault();const r=board.getBoundingClientRect();addPart(PARTS[+i],e.clientX-r.left-55,e.clientY-r.top-34)};
    board.onpointermove=e=>{const p=document.getElementById("xor-ghost");if(!p||!state.pending)return;const a=portCenter(state.pending,"out"),r=board.getBoundingClientRect();if(a)p.setAttribute("d",wirePath(a,{x:e.clientX-r.left,y:e.clientY-r.top}))};
    board.onclick=e=>{if(e.target===board){state.pending=null;state.selected=null;render()}};
    board.onkeydown=e=>{if(e.key==="Delete"||e.key==="Backspace")removeSelected();if(e.key==="Escape"){state.pending=null;render()}};
    field.onclick=e=>{e.stopPropagation();const r=field.getBoundingClientRect();state.probe.x1=clamp((e.clientX-r.left)/r.width*2-1,-1,1);state.probe.x2=clamp(1-(e.clientY-r.top)/r.height*2,-1,1);state.probe.label=(state.probe.x1*state.probe.x2<0)?1:0;persist();render()};
    document.getElementById("xor-step").onclick=()=>train(1);document.getElementById("xor-train").onclick=()=>train(600);document.getElementById("xor-exam").onclick=exam;document.getElementById("xor-back").onclick=ctx.home;document.getElementById("xor-reset").onclick=ctx.reset;
    renderPalette();render();destroyResize=resizeWatch(board,()=>{renderWires();drawField();drawLine(curve,state.history,{color:"#b8a0e3"})});
    window.__NC90_XOR__={graph,state,addPart,connect,train,exam,metrics,reset:ctx.reset,getState:()=>({state:deepCopy(state),graph:graph.toJSON()})};
    return()=>{destroyResize();delete window.__NC90_XOR__}
  }
};
