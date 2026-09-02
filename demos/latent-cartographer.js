import {seeded,randn,mean,mse,clamp,pct,fmt,save,load,uid,drawMatrix,drawLine,canvasBox,resizeWatch,logPush,logHTML,deepCopy} from "../core.js";

const META={
  id:"latent-cartographer",theme:"green",verb:"PAINT / COMPRESS / MAP",title:"潜空间制图室",en:"Latent Cartographer",
  card:"亲手画每个 latent channel 能“听见”哪些像素、能“画回”哪些像素。训练会自己学习连接权重，但信息瓶颈由你布线。",
  tags:["masked autoencoder","latent map","collapse"]
};
function bases(){
  const v=[],h=[];for(let r=0;r<6;r++)for(let c=0;c<6;c++){v.push(c%2===1?.34:.035);h.push(r%2===1?.34:.035)}return[v,h]
}
const [BV,BH]=bases();
function dataset(seed,n){
  const rng=seeded(seed),out=[];for(let i=0;i<n;i++){const a=.08+.88*rng(),b=.08+.88*rng(),x=[];for(let j=0;j<36;j++)x.push(clamp(.03+a*BV[j]+b*BH[j]+randn(rng)*.006,0,1));out.push({x,a,b})}return out
}
const TRAIN=dataset(81021,84),HIDDEN=dataset(92777,150);
const allMask=()=>Array(36).fill(true),emptyMask=()=>Array(36).fill(false);
function newChannel(index){
  const rng=seeded(7001+index*107);return{id:uid("z"),activation:"linear",encMask:emptyMask(),decMask:allMask(),encW:Array.from({length:36},()=>randn(rng)*.05),encB:0,decW:Array.from({length:36},()=>randn(rng)*.05),frozen:false};
}
function fresh(){return{channels:[],outB:Array(36).fill(.12),epoch:0,history:[],hiddenMse:null,selected:null,paintMode:"listen",brush:"toggle",sampleA:5,sampleB:61,t:.5,logs:[],solved:false}}
function restored(){return{...fresh(),...(load(META.id)||{})}}
function act(kind,x){return kind==="tanh"?Math.tanh(x):x}
function dact(kind,z){return kind==="tanh"?1-Math.tanh(z)**2:1}

export default {
  ...META,
  mount(root,ctx){
    let state=restored(),clean=()=>{},painting=false,paintValue=true;
    root.innerHTML=`
      <section class="latent-screen">
        <header class="latent-head">
          <button id="lc-back">← LABS</button>
          <div><span>EXPERIMENT 04 · INFORMATION BOTTLENECK</span><h1>LATENT CARTOGRAPHER</h1></div>
          <div class="latent-contract"><span>CONTRACT</span><b>exactly 2 channels · hidden MSE ≤ .003</b></div>
          <button id="lc-reset">ERASE MAP</button>
        </header>
        <div class="latent-world">
          <aside class="artifact-wall">
            <span class="latent-label">ARTIFACTS</span>
            <div class="artifact-pair"><div><b>SOURCE A</b><canvas id="lc-a"></canvas></div><div><b>RECON A</b><canvas id="lc-ra"></canvas></div></div>
            <div class="artifact-pair"><div><b>SOURCE B</b><canvas id="lc-b"></canvas></div><div><b>LATENT MIX</b><canvas id="lc-mix"></canvas></div></div>
            <label class="latent-slider">INTERPOLATE A → B<input id="lc-t" type="range" min="0" max="1" step=".05"><b id="lc-tv">.50</b></label>
            <div class="artifact-nav"><button id="lc-prev">‹</button><span id="lc-sample">A#5 · B#61</span><button id="lc-next">›</button></div>
          </aside>
          <main class="atlas">
            <div class="atlas-top"><div><span class="latent-label">LEARNED ATLAS</span><p>每个点的位置来自 encoder，不是人工标签坐标。</p></div><button id="lc-add">＋ ADD LATENT CHANNEL</button></div>
            <canvas id="lc-map"></canvas>
            <div class="channel-orbit" id="lc-channels"></div>
            <div class="atlas-actions"><button id="lc-step">1 EPOCH</button><button class="hot" id="lc-train">TRAIN 600</button><button id="lc-exam">HIDDEN RECONSTRUCTION</button></div>
            <div class="atlas-metrics"><div><span>epoch</span><b id="lc-epoch">0</b></div><div><span>train MSE</span><b id="lc-mse">—</b></div><div><span>hidden MSE</span><b id="lc-hidden">—</b></div><div><span>connections</span><b id="lc-connections">0</b></div></div>
          </main>
          <aside class="mask-studio">
            <span class="latent-label">CONNECTIVITY PAINTER</span>
            <div id="lc-mask-ui"></div>
          </aside>
        </div>
        <footer class="latent-footer">
          <section><span class="latent-label">RECONSTRUCTION LOSS</span><canvas id="lc-curve"></canvas></section>
          <section><span class="latent-label">LATENT DIAGNOSTICS</span><div id="lc-diagnosis"></div></section>
          <section><span class="latent-label">TRAINING LOG</span><div class="latent-log" id="lc-log"></div></section>
        </footer>
      </section>`;
    const ca=document.getElementById("lc-a"),cra=document.getElementById("lc-ra"),cb=document.getElementById("lc-b"),cmix=document.getElementById("lc-mix"),map=document.getElementById("lc-map"),curve=document.getElementById("lc-curve");

    function persist(){save(META.id,state)}
    function addChannel(){
      if(state.channels.length>=3){ctx.toast("最多 3 个 latent channels","bad");return}
      const c=newChannel(state.channels.length);state.channels.push(c);state.selected=c.id;state.hiddenMse=null;state.solved=false;logPush(state.logs,"added latent channel z"+state.channels.length,"ok");persist();render();return c
    }
    function channel(id){return state.channels.find(c=>c.id===id)}
    function removeChannel(id){state.channels=state.channels.filter(c=>c.id!==id);if(state.selected===id)state.selected=state.channels[0]?.id||null;state.hiddenMse=null;state.solved=false;logPush(state.logs,"latent channel removed","warn");persist();render()}
    function encode(x){
      const z=[],pre=[];for(const c of state.channels){let s=c.encB;for(let j=0;j<36;j++)if(c.encMask[j])s+=c.encW[j]*x[j];pre.push(s);z.push(act(c.activation,s))}return{z,pre}
    }
    function decode(z){
      const y=Array(36);for(let j=0;j<36;j++){let s=state.outB[j];for(let k=0;k<state.channels.length;k++)if(state.channels[k].decMask[j])s+=state.channels[k].decW[j]*z[k];y[j]=s}return y
    }
    function forward(x){const e=encode(x);return{...e,y:decode(e.z)}}
    function metrics(data=TRAIN){if(!state.channels.length)return{mse:.04};return{mse:mean(data.map(ex=>mse(forward(ex.x).y,ex.x)))}}
    function trainEpoch(){
      const K=state.channels.length;if(!K)return metrics();
      const gEncW=state.channels.map(()=>Array(36).fill(0)),gEncB=Array(K).fill(0),gDecW=state.channels.map(()=>Array(36).fill(0)),gOut=Array(36).fill(0);const n=TRAIN.length;
      for(const ex of TRAIN){
        const o=forward(ex.x),dy=o.y.map((v,j)=>2*(v-ex.x[j])/36),dz=Array(K).fill(0);
        for(let j=0;j<36;j++){gOut[j]+=dy[j]/n;for(let k=0;k<K;k++)if(state.channels[k].decMask[j]){gDecW[k][j]+=dy[j]*o.z[k]/n;dz[k]+=dy[j]*state.channels[k].decW[j]}}
        for(let k=0;k<K;k++){const c=state.channels[k],dp=dz[k]*dact(c.activation,o.pre[k]);gEncB[k]+=dp/n;for(let j=0;j<36;j++)if(c.encMask[j])gEncW[k][j]+=dp*ex.x[j]/n}
      }
      const lr=.085;for(let j=0;j<36;j++)state.outB[j]-=lr*gOut[j];
      for(let k=0;k<K;k++){const c=state.channels[k];if(c.frozen)continue;c.encB-=lr*gEncB[k];for(let j=0;j<36;j++){if(c.encMask[j])c.encW[j]-=lr*gEncW[k][j];if(c.decMask[j])c.decW[j]-=lr*gDecW[k][j]}}
      state.epoch++;return metrics()
    }
    function train(n){
      if(!state.channels.length){ctx.toast("先添加 latent channel","bad");return}
      if(state.channels.some(c=>!c.encMask.some(Boolean))){ctx.toast("至少有一个 channel 完全听不见输入","bad")}
      let m;for(let i=0;i<n;i++){m=trainEpoch();if(state.epoch===1||state.epoch%4===0)state.history.push(m.mse)}
      if(state.history.length>260)state.history.splice(0,state.history.length-260);state.hiddenMse=null;logPush(state.logs,`trained ${n} epochs · MSE ${fmt(m.mse,5)}`,m.mse<.004?"ok":"");persist();render()
    }
    function exam(){
      const m=metrics(HIDDEN);state.hiddenMse=m.mse;const pass=state.channels.length===2&&m.mse<=.003;
      logPush(state.logs,`hidden reconstruction ${fmt(m.mse,5)} · ${pass?"PASS":"FAIL"}`,pass?"ok":"err");if(pass&&!state.solved){state.solved=true;ctx.complete()}persist();render()
    }
    function latentStats(){
      const zs=TRAIN.map(ex=>encode(ex.x).z),stats=[];for(let k=0;k<state.channels.length;k++){const vals=zs.map(z=>z[k]),m=mean(vals),v=mean(vals.map(x=>(x-m)**2));stats.push({mean:m,var:v})}
      let corr=0;if(state.channels.length>=2){const a=zs.map(z=>z[0]),b=zs.map(z=>z[1]),ma=mean(a),mb=mean(b),sa=Math.sqrt(mean(a.map(x=>(x-ma)**2))),sb=Math.sqrt(mean(b.map(x=>(x-mb)**2)));corr=sa*sb?mean(a.map((x,i)=>(x-ma)*(b[i]-mb)))/(sa*sb):0}
      return{stats,corr}
    }
    function connections(){return state.channels.reduce((s,c)=>s+c.encMask.filter(Boolean).length+c.decMask.filter(Boolean).length,0)}
    function matrix(v){const m=[];for(let r=0;r<6;r++)m.push(v.slice(r*6,r*6+6).map(x=>clamp(x,0,1)));return m}
    function drawArtifacts(){
      const A=TRAIN[state.sampleA],B=TRAIN[state.sampleB],oa=forward(A.x),eb=encode(B.x),t=state.t,z=oa.z.map((v,k)=>v*(1-t)+(eb.z[k]??v)*t),mix=decode(z);
      drawMatrix(ca,matrix(A.x),{signed:false});drawMatrix(cra,matrix(oa.y),{signed:false});drawMatrix(cb,matrix(B.x),{signed:false});drawMatrix(cmix,matrix(mix),{signed:false});document.getElementById("lc-sample").textContent=`A#${state.sampleA} · B#${state.sampleB}`
    }
    function drawAtlas(){
      const {ctx:g,w,h}=canvasBox(map);g.fillStyle="#09110d";g.fillRect(0,0,w,h);g.strokeStyle="#21342a";for(let i=1;i<5;i++){g.beginPath();g.moveTo(w*i/5,0);g.lineTo(w*i/5,h);g.moveTo(0,h*i/5);g.lineTo(w,h*i/5);g.stroke()}
      if(!state.channels.length){g.fillStyle="#60746a";g.font="11px ui-monospace";g.textAlign="center";g.fillText("NO LATENT COORDINATES",w/2,h/2);return}
      const pts=TRAIN.map(ex=>({z:encode(ex.x).z,a:ex.a,b:ex.b})),xs=pts.map(p=>p.z[0]),ys=state.channels.length>1?pts.map(p=>p.z[1]):pts.map(p=>p.a);let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);if(xmax-xmin<1e-8)xmax=xmin+1;if(ymax-ymin<1e-8)ymax=ymin+1;
      for(const p of pts){const x=12+(p.z[0]-xmin)/(xmax-xmin)*(w-24),yv=state.channels.length>1?p.z[1]:p.a,y=h-12-(yv-ymin)/(ymax-ymin)*(h-24);g.fillStyle=`rgb(${Math.round(70+160*p.a)},${Math.round(85+135*p.b)},120)`;g.beginPath();g.arc(x,y,3,0,Math.PI*2);g.fill()}
      g.fillStyle="#70857a";g.font="8px ui-monospace";g.fillText(state.channels.length>1?"z₁ ↔ z₂":"z₁ ↔ diagnostic factor",8,11)
    }
    function renderChannels(){
      const stats=latentStats();document.getElementById("lc-channels").innerHTML=state.channels.length?state.channels.map((c,i)=>`<button class="latent-beacon ${state.selected===c.id?"selected":""}" data-z="${c.id}"><i></i><b>z${i+1}</b><span>var ${fmt(stats.stats[i]?.var,4)}</span><small>${c.encMask.filter(Boolean).length} listen · ${c.decMask.filter(Boolean).length} paint</small></button>`).join(""):'<div class="no-beacons">ADD CHANNELS TO CREATE A BOTTLENECK</div>';
      document.querySelectorAll(".latent-beacon").forEach(b=>b.onclick=()=>{state.selected=b.dataset.z;renderMask();renderChannels()})
    }
    function cellGrid(mask,kind){
      return`<div class="mask-grid ${kind}" data-mask="${kind}">${mask.map((on,i)=>`<button data-cell="${i}" class="${on?"on":""}" title="pixel ${Math.floor(i/6)},${i%6}"></button>`).join("")}</div>`
    }
    function renderMask(){
      const box=document.getElementById("lc-mask-ui"),c=channel(state.selected);if(!c){box.innerHTML='<div class="mask-empty"><b>NO CHANNEL SELECTED</b><span>添加一个 latent channel，然后画它的连接。</span></div>';return}
      const idx=state.channels.indexOf(c),st=latentStats().stats[idx]||{};
      box.innerHTML=`<div class="mask-title"><b>z${idx+1}</b><span>${c.activation} channel</span></div>
        <label>ACTIVATION<select id="lc-act"><option value="linear">linear</option><option value="tanh">tanh</option></select></label>
        <div class="mask-tabs"><button data-mode="listen" class="${state.paintMode==="listen"?"on":""}">LISTEN MASK</button><button data-mode="paint" class="${state.paintMode==="paint"?"on":""}">PAINT MASK</button></div>
        <p>${state.paintMode==="listen"?"亮格 = 这个 latent 能从该输入像素接收可训练权重。":"亮格 = 这个 latent 能向该重建像素输出可训练权重。"}</p>
        ${cellGrid(state.paintMode==="listen"?c.encMask:c.decMask,state.paintMode)}
        <div class="mask-tools"><button id="lc-all">ALL</button><button id="lc-clear">CLEAR</button><button id="lc-invert">INVERT</button></div>
        <div class="mask-stats"><div><span>variance</span><b>${fmt(st.var,5)}</b></div><div><span>encoder bias</span><b>${fmt(c.encB,4)}</b></div></div>
        <div class="mask-actions"><button id="lc-freeze" class="${c.frozen?"on":""}">❄ FREEZE</button><button id="lc-remove">REMOVE CHANNEL</button></div>`;
      const sel=document.getElementById("lc-act");sel.value=c.activation;sel.onchange=()=>{c.activation=sel.value;state.hiddenMse=null;persist();render()};
      document.querySelectorAll("[data-mode]").forEach(b=>b.onclick=()=>{state.paintMode=b.dataset.mode;persist();renderMask()});
      const mask=state.paintMode==="listen"?c.encMask:c.decMask;
      document.querySelectorAll(".mask-grid button").forEach(b=>{
        const toggle=()=>{const i=+b.dataset.cell;mask[i]=!mask[i];state.hiddenMse=null;state.solved=false;persist();renderMask();renderChannels()};
        b.onpointerdown=e=>{e.preventDefault();painting=true;paintValue=!mask[+b.dataset.cell];mask[+b.dataset.cell]=paintValue;state.hiddenMse=null;persist();renderMask();renderChannels()};
        b.onpointerenter=()=>{if(painting){mask[+b.dataset.cell]=paintValue;persist();b.classList.toggle("on",paintValue)}};
      });
      const apply=fn=>{for(let i=0;i<36;i++)mask[i]=fn(mask[i]);state.hiddenMse=null;state.solved=false;persist();renderMask();renderChannels()};
      document.getElementById("lc-all").onclick=()=>apply(()=>true);document.getElementById("lc-clear").onclick=()=>apply(()=>false);document.getElementById("lc-invert").onclick=()=>apply(v=>!v);
      document.getElementById("lc-freeze").onclick=()=>{c.frozen=!c.frozen;persist();renderMask()};document.getElementById("lc-remove").onclick=()=>removeChannel(c.id)
    }
    function diagnosis(){
      const m=metrics(),st=latentStats();if(!state.channels.length)return"没有 bottleneck：decoder 只能学习平均图像。";
      if(state.channels.some(c=>!c.encMask.some(Boolean)))return"至少一个 latent channel 完全没有输入连接，它的 variance 会塌到 0。";
      if(state.channels.length===1&&state.epoch>150)return"只有一个连续标量必须同时编码两个独立生成因素。观察 reconstruction error 的残余条纹。";
      if(state.channels.length>=2&&Math.abs(st.corr)>.96&&state.epoch>120)return"两个 latent 坐标高度相关：它们可能在重复编码同一因素，出现 representation collapse。";
      if(state.channels.length>2)return"重构可能很容易，但关卡要求恰好 2D。你需要压缩，而不是只增加容量。";
      if(m.mse<.003)return"低重构误差已经出现。现在检查 hidden combinations，确认你的连接结构不是只记住训练组合。";
      return"用 mask 改变信息能流向哪里；训练只负责在你允许的连接上学习权重。";
    }
    function render(){
      const m=metrics();drawArtifacts();drawAtlas();renderChannels();renderMask();drawLine(curve,state.history,{color:"#75d6a1",label:"MSE"});document.getElementById("lc-epoch").textContent=state.epoch;document.getElementById("lc-mse").textContent=fmt(m.mse,5);document.getElementById("lc-hidden").textContent=state.hiddenMse===null?"—":fmt(state.hiddenMse,5);document.getElementById("lc-connections").textContent=connections();document.getElementById("lc-diagnosis").innerHTML=`<b>${diagnosis()}</b><span>latent corr: ${fmt(latentStats().corr,3)}</span>`;document.getElementById("lc-log").innerHTML=logHTML(state.logs);document.getElementById("lc-t").value=state.t;document.getElementById("lc-tv").textContent=fmt(state.t,2)
    }
    window.addEventListener("pointerup",()=>painting=false);
    document.getElementById("lc-add").onclick=addChannel;document.getElementById("lc-step").onclick=()=>train(1);document.getElementById("lc-train").onclick=()=>train(600);document.getElementById("lc-exam").onclick=exam;document.getElementById("lc-back").onclick=ctx.home;document.getElementById("lc-reset").onclick=ctx.reset;
    document.getElementById("lc-prev").onclick=()=>{state.sampleA=(state.sampleA-1+TRAIN.length)%TRAIN.length;state.sampleB=(state.sampleB-1+TRAIN.length)%TRAIN.length;persist();render()};document.getElementById("lc-next").onclick=()=>{state.sampleA=(state.sampleA+1)%TRAIN.length;state.sampleB=(state.sampleB+1)%TRAIN.length;persist();render()};
    document.getElementById("lc-t").oninput=e=>{state.t=+e.target.value;persist();drawArtifacts();document.getElementById("lc-tv").textContent=fmt(state.t,2)};
    render();clean=resizeWatch(map,render);
    window.__NC90_LATENT__={state,addChannel,train,exam,metrics,encode,forward,setMasks:(id,enc,dec=null)=>{const c=channel(id);if(c){c.encMask=[...enc];if(dec)c.decMask=[...dec];persist();render()}},getState:()=>deepCopy(state)};
    return()=>{clean();delete window.__NC90_LATENT__}
  }
};
