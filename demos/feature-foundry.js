import {seeded,randn,sigmoid,bce,mean,clamp,pct,fmt,save,load,uid,canvasBox,drawLine,resizeWatch,logPush,logHTML,deepCopy} from "../core.js";

const META={
  id:"feature-foundry",theme:"amber",verb:"FORGE / ROUTE / COMPARE",title:"特征铸造厂",en:"Feature Foundry",
  card:"把原始坐标真的加工成新特征。Square、Abs、Add、Multiply 是机器，不是下拉选项；你决定数据经过哪条生产线。",
  tags:["feature graph","logistic","geometry"]
};
function generate(seed,n){
  const rng=seeded(seed),out=[];
  while(out.length<n){
    const x=rng()*1.9-.95,y=rng()*1.9-.95,r=Math.hypot(x,y),l1=Math.abs(x)+Math.abs(y),label=r>.59?1:0;
    if(Math.abs(r-.59)<.09)continue;
    if(label&&l1<.84)continue;
    if(!label&&l1>.76)continue;
    out.push({x1:x,x2:y,label});
  }
  return out;
}
const TRAIN=generate(14141,150),HIDDEN=generate(99173,240);
const OPS=[
  {id:"square",label:"SQUARE",symbol:"x²",arity:1},
  {id:"abs",label:"ABS",symbol:"|x|",arity:1},
  {id:"add",label:"ADD",symbol:"+",arity:2},
  {id:"mul",label:"MULTIPLY",symbol:"×",arity:2},
  {id:"sub",label:"SUBTRACT",symbol:"−",arity:2}
];
const RAW=[
  {id:"raw-x1",expr:{op:"x1"},formula:"x₁",fixed:true},
  {id:"raw-x2",expr:{op:"x2"},formula:"x₂",fixed:true}
];
function evalExpr(e,p){
  if(e.op==="x1")return p.x1;if(e.op==="x2")return p.x2;
  const a=evalExpr(e.a,p);
  if(e.op==="square")return a*a;if(e.op==="abs")return Math.abs(a);
  const b=evalExpr(e.b,p);if(e.op==="add")return a+b;if(e.op==="mul")return a*b;if(e.op==="sub")return a-b;return 0
}
function formula(e){
  if(e.op==="x1")return"x₁";if(e.op==="x2")return"x₂";
  const a=formula(e.a);if(e.op==="square")return"("+a+")²";if(e.op==="abs")return"|"+a+"|";
  const b=formula(e.b),s={add:"+",mul:"×",sub:"−"}[e.op];return"("+a+" "+s+" "+b+")";
}
function fresh(){
  return{features:[],forgeA:"raw-x1",forgeB:"raw-x2",selected:"raw-x1",docks:[null,null,null],weights:[0,0,0,0],epoch:0,history:[],hidden:null,logs:[],solved:false};
}
function restored(){return{...fresh(),...(load(META.id)||{})}}
function allFeatures(state){return[...RAW,...state.features]}
function getFeature(state,id){return allFeatures(state).find(f=>f.id===id)}
function featureValues(state,id,data=TRAIN){const f=getFeature(state,id);return f?data.map(p=>evalExpr(f.expr,p)):[]}
function corr(a,b){const ma=mean(a),mb=mean(b),da=Math.sqrt(mean(a.map(x=>(x-ma)**2))),db=Math.sqrt(mean(b.map(x=>(x-mb)**2)));return da*db?mean(a.map((x,i)=>(x-ma)*(b[i]-mb)))/(da*db):0}
function aucLike(vals,data){
  const a=[],b=[];vals.forEach((v,i)=>(data[i].label?b:a).push(v));if(!a.length||!b.length)return 0;
  const ma=mean(a),mb=mean(b),sa=Math.sqrt(mean(a.map(x=>(x-ma)**2))),sb=Math.sqrt(mean(b.map(x=>(x-mb)**2)));return Math.abs(mb-ma)/(sa+sb+1e-6)
}

export default {
  ...META,
  mount(root,ctx){
    let state=restored(),clean=()=>{};
    root.innerHTML=`
      <section class="feature-screen">
        <header class="factory-head">
          <button id="ff-back">← LABS</button>
          <div class="factory-brand"><span>EXPERIMENT 02</span><h1>FEATURE FOUNDRY</h1><small>把几何结构加工成模型能线性利用的特征</small></div>
          <div class="factory-order"><span>ORDER</span><b>hidden accuracy ≥ 94%</b></div>
          <button id="ff-reset">CLEAR FACTORY</button>
        </header>
        <div class="factory-floor">
          <section class="raw-bin">
            <div class="floor-label">RAW MATERIAL</div>
            <div id="ff-raw"></div>
            <div class="raw-scatter"><canvas id="ff-scatter"></canvas><span>decision field / training points</span></div>
          </section>
          <section class="forge-bay">
            <div class="floor-label">TRANSFORM PRESS</div>
            <div class="forge-inputs">
              <div class="forge-slot" data-slot="A"><span>INPUT A</span><b id="ff-slot-a">drop feature</b></div>
              <div class="forge-slot" data-slot="B"><span>INPUT B</span><b id="ff-slot-b">drop feature</b></div>
            </div>
            <div id="ff-machines" class="machines"></div>
            <div class="forge-arrow">↓</div>
            <div class="output-cradle"><span>NEW FEATURE</span><b>运行一台机器会把结果送进右侧货架</b></div>
            <div class="factory-tip">同一原料可以反复加工。系统不会告诉你“Radial feature”这个答案；你只能看到每道工序产生的分布。</div>
          </section>
          <section class="feature-rack">
            <div class="floor-label">FEATURE SHELF</div>
            <div id="ff-shelf" class="shelf"></div>
          </section>
          <section class="classifier-rack">
            <div class="floor-label">CLASSIFIER LOADING DOCK</div>
            <p>把最多 3 张特征卡拖进线性分类器。权重由梯度下降学习，不需要你手调。</p>
            <div id="ff-docks" class="dock-list"></div>
            <div class="classifier-core"><div><span>LOGISTIC HEAD</span><b id="ff-prob">σ(w·f+b)</b></div><i></i></div>
            <div class="factory-actions"><button id="ff-step">1 EPOCH</button><button class="hot" id="ff-train">TRAIN 250</button><button id="ff-exam">SHIP / HIDDEN TEST</button></div>
          </section>
        </div>
        <footer class="factory-lab">
          <section><span class="floor-label">FEATURE MICROSCOPE</span><div id="ff-inspect"></div></section>
          <section><span class="floor-label">TRAINING TRACE</span><canvas id="ff-curve"></canvas></section>
          <section><span class="floor-label">DIAGNOSTIC LOG</span><div class="factory-log" id="ff-log"></div></section>
          <section class="factory-score"><div><span>epoch</span><b id="ff-epoch">0</b></div><div><span>train</span><b id="ff-train-acc">—</b></div><div><span>hidden</span><b id="ff-hidden">—</b></div></section>
        </footer>
      </section>`;
    const scatter=document.getElementById("ff-scatter"),curve=document.getElementById("ff-curve");

    function persist(){save(META.id,state)}
    function resetModel(reason="classifier structure changed"){
      state.weights=[0,...state.docks.map(()=>0)].slice(0,1+state.docks.filter(Boolean).length);state.epoch=0;state.history=[];state.hidden=null;state.solved=false;logPush(state.logs,reason,"warn");persist()
    }
    function classifierFeatures(p){return state.docks.filter(Boolean).map(id=>evalExpr(getFeature(state,id).expr,p))}
    function predict(p){
      const fs=classifierFeatures(p);let z=state.weights[0]||0;for(let i=0;i<fs.length;i++)z+=(state.weights[i+1]||0)*fs[i];return sigmoid(z)
    }
    function metrics(data=TRAIN){
      const ids=state.docks.filter(Boolean);if(!ids.length)return{loss:.693,acc:.5,errors:data};
      let loss=0,c=0,errors=[];for(const p of data){const pr=predict(p);loss+=bce(pr,p.label);const good=(pr>=.5)==p.label;c+=good;if(!good)errors.push({...p,p:pr})}return{loss:loss/data.length,acc:c/data.length,errors}
    }
    function trainEpoch(){
      const ids=state.docks.filter(Boolean);if(!ids.length)return metrics();
      const g=Array(ids.length+1).fill(0);
      for(const p of TRAIN){const fs=classifierFeatures(p),pr=predict(p),e=pr-p.label;g[0]+=e/TRAIN.length;for(let i=0;i<fs.length;i++)g[i+1]+=e*fs[i]/TRAIN.length}
      const lr=.16;for(let i=0;i<g.length;i++)state.weights[i]=(state.weights[i]||0)-lr*g[i];state.epoch++;return metrics()
    }
    function train(n){
      if(!state.docks.some(Boolean)){ctx.toast("先把特征送进 classifier dock","bad");return}
      let m;for(let i=0;i<n;i++){m=trainEpoch();if(state.epoch===1||state.epoch%4===0)state.history.push(m.loss)}
      state.hidden=null;logPush(state.logs,`classifier trained ${n} epochs · loss ${fmt(m.loss,4)} · acc ${pct(m.acc)}`,m.acc>.93?"ok":"");persist();render()
    }
    function exam(){
      const m=metrics(HIDDEN);state.hidden=m.acc;const pass=m.acc>=.94;logPush(state.logs,`shipment hidden test ${pct(m.acc)} · ${pass?"PASS":"FAIL"}`,pass?"ok":"err");
      if(pass&&!state.solved){state.solved=true;ctx.complete()}persist();render()
    }
    function forge(op){
      const a=getFeature(state,state.forgeA),b=getFeature(state,state.forgeB);if(!a){ctx.toast("INPUT A 为空","bad");return}
      const def=OPS.find(x=>x.id===op);if(def.arity===2&&!b){ctx.toast("这台机器需要两个输入","bad");return}
      const expr=def.arity===1?{op,a:deepCopy(a.expr)}:{op,a:deepCopy(a.expr),b:deepCopy(b.expr)};
      const f={id:uid("feat"),expr,formula:formula(expr)};state.features.push(f);state.selected=f.id;logPush(state.logs,"forged "+f.formula,"ok");persist();render()
    }
    function deleteFeature(id){
      state.features=state.features.filter(f=>f.id!==id);state.docks=state.docks.map(x=>x===id?null:x);if(state.forgeA===id)state.forgeA="raw-x1";if(state.forgeB===id)state.forgeB="raw-x2";state.selected="raw-x1";resetModel("derived feature removed");render()
    }
    function featureCard(f){
      const vals=featureValues(state,f.id),sep=aucLike(vals,TRAIN);
      return `<article class="feature-card ${state.selected===f.id?"selected":""}" draggable="true" data-feature="${f.id}"><div><span>${f.fixed?"RAW":"DERIVED"}</span>${!f.fixed?'<button data-del="'+f.id+'">×</button>':""}</div><b>${f.formula}</b><small>class separation ${fmt(sep,2)}</small><i style="width:${Math.min(100,sep*34)}%"></i></article>`
    }
    function renderCards(){
      document.getElementById("ff-raw").innerHTML=RAW.map(featureCard).join("");
      document.getElementById("ff-shelf").innerHTML=state.features.length?state.features.map(featureCard).join(""):'<div class="empty-shelf">铸造出的特征会出现在这里</div>';
      document.querySelectorAll(".feature-card").forEach(card=>{
        card.ondragstart=e=>e.dataTransfer.setData("text/x-feature",card.dataset.feature);
        card.onclick=e=>{if(e.target.dataset.del)return;state.selected=card.dataset.feature;renderInspect()};
      });
      document.querySelectorAll("[data-del]").forEach(b=>b.onclick=e=>{e.stopPropagation();deleteFeature(b.dataset.del)});
    }
    function renderDocks(){
      document.getElementById("ff-docks").innerHTML=state.docks.map((id,i)=>`<div class="classifier-dock ${id?"loaded":""}" data-dock="${i}"><span>F${i+1}</span><b>${id?getFeature(state,id)?.formula:"DROP FEATURE"}</b>${id?'<button data-undock="'+i+'">×</button>':""}</div>`).join("");
      document.querySelectorAll(".classifier-dock").forEach(d=>{d.ondragover=e=>e.preventDefault();d.ondrop=e=>{e.preventDefault();const id=e.dataTransfer.getData("text/x-feature");if(!getFeature(state,id))return;state.docks[+d.dataset.dock]=id;resetModel("classifier feature set changed");render()}});
      document.querySelectorAll("[data-undock]").forEach(b=>b.onclick=()=>{state.docks[+b.dataset.undock]=null;resetModel("feature undocked");render()});
    }
    function renderForge(){
      document.getElementById("ff-slot-a").textContent=getFeature(state,state.forgeA)?.formula||"drop feature";document.getElementById("ff-slot-b").textContent=getFeature(state,state.forgeB)?.formula||"drop feature";
      document.getElementById("ff-machines").innerHTML=OPS.map(o=>`<button data-op="${o.id}" class="machine ${o.arity===1?"unary":""}"><b>${o.symbol}</b><span>${o.label}</span><small>${o.arity} input</small></button>`).join("");
      document.querySelectorAll(".machine").forEach(m=>m.onclick=()=>forge(m.dataset.op));
      document.querySelectorAll(".forge-slot").forEach(s=>{s.ondragover=e=>e.preventDefault();s.ondrop=e=>{e.preventDefault();const id=e.dataTransfer.getData("text/x-feature");if(getFeature(state,id)){if(s.dataset.slot==="A")state.forgeA=id;else state.forgeB=id;persist();renderForge()}}});
    }
    function renderInspect(){
      const f=getFeature(state,state.selected),box=document.getElementById("ff-inspect");if(!f){box.innerHTML="select a feature";return}
      const vals=featureValues(state,f.id),a=vals.filter((_,i)=>!TRAIN[i].label),b=vals.filter((_,i)=>TRAIN[i].label),sep=aucLike(vals,TRAIN);
      const docked=state.docks.filter(Boolean),redundant=docked.filter(id=>id!==f.id).map(id=>Math.abs(corr(vals,featureValues(state,id)))).reduce((m,x)=>Math.max(m,x),0);
      box.innerHTML=`<div class="microscope-formula">${f.formula}</div><div class="micro-stats"><div><span>class 0 mean</span><b>${fmt(mean(a),3)}</b></div><div><span>class 1 mean</span><b>${fmt(mean(b),3)}</b></div><div><span>separation</span><b>${fmt(sep,3)}</b></div><div><span>max redundancy</span><b>${fmt(redundant,3)}</b></div></div><div class="micro-bars"><i style="left:${clamp((mean(a)+2)/4*100,0,100)}%"></i><b style="left:${clamp((mean(b)+2)/4*100,0,100)}%"></b></div><p>${sep<.35?"两类在这个特征上大量重叠；线性 head 很难只靠它分开。":redundant>.96?"它和已装载特征高度重复，可能只是增加参数而没有增加信息。":"这个特征改变了两类样本在一维轴上的相对位置。"}</p>`
    }
    function drawScatter(){
      const {ctx:g,w,h}=canvasBox(scatter),pad=18,res=26;g.fillStyle="#0a0e10";g.fillRect(0,0,w,h);
      if(state.docks.some(Boolean))for(let iy=0;iy<res;iy++)for(let ix=0;ix<res;ix++){const x=-1+2*(ix+.5)/res,y=1-2*(iy+.5)/res,p=predict({x1:x,x2:y,label:0});g.fillStyle=`rgba(${Math.round(40+185*p)},${Math.round(85+75*(1-p))},${Math.round(110-45*p)},.45)`;g.fillRect(pad+ix*(w-2*pad)/res,pad+iy*(h-2*pad)/res,(w-2*pad)/res+1,(h-2*pad)/res+1)}
      g.strokeStyle="#2f3a3f";g.strokeRect(pad,pad,w-2*pad,h-2*pad);
      for(const p of TRAIN){const x=pad+(p.x1+1)/2*(w-2*pad),y=pad+(1-p.x2)/2*(h-2*pad);g.beginPath();g.arc(x,y,2.4,0,Math.PI*2);g.fillStyle=p.label?"#ffbb5e":"#65d3d7";g.fill()}
    }
    function diagnosis(m){
      const ids=state.docks.filter(Boolean);if(!ids.length)return"没有输入特征：classifier 只能输出一个常数。";
      if(m.acc<.8&&state.epoch>120){
        const seps=ids.map(id=>aucLike(featureValues(state,id),TRAIN));if(Math.max(...seps)<.5)return"训练已经很久，但每个已装载特征都让两类高度重叠。问题更像表示能力，而不是优化速度。";
      }
      if(ids.length>1){let maxc=0;for(let i=0;i<ids.length;i++)for(let j=i+1;j<ids.length;j++)maxc=Math.max(maxc,Math.abs(corr(featureValues(state,ids[i]),featureValues(state,ids[j]))));if(maxc>.98)return"两个装载特征几乎完全冗余。尝试让生产线提供不同几何信息。"}
      return"观察散点背景如何随着你加工出的 feature 改变；这是结构 → 表示 → 边界的直接因果链。";
    }
    function render(){
      renderCards();renderForge();renderDocks();renderInspect();drawScatter();drawLine(curve,state.history,{color:"#ffb347",label:"BCE"});
      const m=metrics();document.getElementById("ff-epoch").textContent=state.epoch;document.getElementById("ff-train-acc").textContent=pct(m.acc);document.getElementById("ff-hidden").textContent=state.hidden===null?"—":pct(state.hidden);document.getElementById("ff-log").innerHTML=logHTML(state.logs)+`<div class="diagnose">${diagnosis(m)}</div>`;
      document.getElementById("ff-prob").textContent=state.docks.filter(Boolean).length?state.docks.filter(Boolean).map((id,i)=>"w"+(i+1)+"·"+getFeature(state,id).formula).join(" + "):"σ(bias only)";
    }
    document.getElementById("ff-step").onclick=()=>train(1);document.getElementById("ff-train").onclick=()=>train(250);document.getElementById("ff-exam").onclick=exam;document.getElementById("ff-back").onclick=ctx.home;document.getElementById("ff-reset").onclick=ctx.reset;
    render();clean=resizeWatch(scatter,()=>{drawScatter();drawLine(curve,state.history,{color:"#ffb347"})});
    window.__NC90_FEATURE__={state,forge,train,exam,metrics,getFeature,evalExpr,setForge:(a,b)=>{state.forgeA=a;state.forgeB=b;render()},dock:(i,id)=>{state.docks[i]=id;resetModel("test dock");render()},getState:()=>deepCopy(state)};
    return()=>{clean();delete window.__NC90_FEATURE__}
  }
};
