import {seeded,randn,sigmoid,bce,mean,clamp,pct,fmt,save,load,uid,drawMatrix,drawLine,resizeWatch,logPush,logHTML,deepCopy} from "../core.js";

const META={
  id:"vision-forge",theme:"cyan",verb:"ASSEMBLE / FOCUS / INSPECT",title:"视觉锻造台",en:"Vision Forge",
  card:"不是选择“CNN”。你要把可训练滤镜装进光学轨道，决定感受野和响应压缩方式，再观察卷积核从噪声长成特征。",
  tags:["trainable filters","feature maps","translation"]
};
function image(label,rng,hidden=false){
  const n=9,img=Array.from({length:n},()=>Array(n).fill(0));
  const pos=hidden?(1+Math.floor(rng()*7)):(2+Math.floor(rng()*5));
  if(label===1){
    for(let r=0;r<n;r++){img[r][pos]=1;if(pos+1<n)img[r][pos+1]=.28}
  }else{
    for(let c=0;c<n;c++){img[pos][c]=1;if(pos+1<n)img[pos+1][c]=.28}
  }
  for(let r=0;r<n;r++)for(let c=0;c<n;c++)img[r][c]=clamp(img[r][c]+randn(rng)*(hidden?.09:.065),0,1);
  return img
}
function dataset(seed,n,hidden=false){const rng=seeded(seed),out=[];for(let i=0;i<n;i++){const y=i%2;out.push({img:image(y,rng,hidden),label:y})}return out}
const TRAIN=dataset(45001,100,false),HIDDEN=dataset(77031,200,true);

function blankModule(size=3,seed=1){
  const rng=seeded(seed);return{id:uid("filter"),size,reducer:"max",kernel:Array.from({length:size},()=>Array.from({length:size},()=>randn(rng)*.14)),bias:.02,outWeight:randn(rng)*.35,frozen:false,muted:false};
}
function fresh(){return{modules:[],outBias:0,epoch:0,history:[],sample:0,hidden:null,logs:[],solved:false}}
function restored(){return{...fresh(),...(load(META.id)||{})}}
function conv(img,k,bias){
  const s=k.length,n=img.length,m=n-s+1,z=Array.from({length:m},()=>Array(m).fill(0)),a=Array.from({length:m},()=>Array(m).fill(0));
  for(let r=0;r<m;r++)for(let c=0;c<m;c++){let v=bias;for(let kr=0;kr<s;kr++)for(let kc=0;kc<s;kc++)v+=img[r+kr][c+kc]*k[kr][kc];z[r][c]=v;a[r][c]=Math.max(0,v)}
  return{z,a}
}
function reduceMap(a,kind){
  const flat=[];for(let r=0;r<a.length;r++)for(let c=0;c<a[0].length;c++)flat.push({v:a[r][c],r,c});
  if(kind==="mean")return{value:mean(flat.map(x=>x.v)),chosen:flat.map(x=>({...x,w:1/flat.length}))};
  flat.sort((x,y)=>y.v-x.v);const take=kind==="top2"?flat.slice(0,2):flat.slice(0,1);return{value:mean(take.map(x=>x.v)),chosen:take.map(x=>({...x,w:1/take.length}))}
}

export default {
  ...META,
  mount(root,ctx){
    let state=restored(),clean=()=>{},selected=state.modules[0]?.id||null;
    root.innerHTML=`
      <section class="vision-screen">
        <header class="vision-head">
          <button id="vf-back">← LABS</button>
          <div class="vision-title"><span>EXPERIMENT 03 · OPTICAL BENCH</span><h1>VISION FORGE</h1></div>
          <div class="vision-legend"><i></i><span>horizontal = 0</span><b></b><span>vertical = 1</span></div>
          <button id="vf-reset">DISASSEMBLE</button>
        </header>
        <div class="vision-work">
          <aside class="film-stage">
            <span class="vision-label">SPECIMEN</span>
            <div class="specimen"><canvas id="vf-input"></canvas><div class="scanline"></div></div>
            <div class="sample-scrub"><button id="vf-prev">‹</button><div><span id="vf-sample-id">#0</span><b id="vf-sample-label">label 0</b></div><button id="vf-next">›</button></div>
            <div class="parts-drawer">
              <span>OPTICS DRAWER</span>
              <button draggable="true" data-size="3"><b>3×3</b><small>trainable filter</small></button>
              <button draggable="true" data-size="5"><b>5×5</b><small>wide filter</small></button>
            </div>
          </aside>
          <main class="optical-bench">
            <div class="bench-ruler"><span>IMAGE</span><i></i><span>FILTER BANK</span><i></i><span>LOGISTIC HEAD</span></div>
            <div class="optical-track" id="vf-track">
              <div class="track-input"><canvas id="vf-mini-input"></canvas><span>9×9 photons</span></div>
              <div class="filter-bank" id="vf-bank"></div>
              <div class="track-output"><div class="prob-dial"><i id="vf-needle"></i><b id="vf-prob">.500</b><span>P(vertical)</span></div></div>
            </div>
            <div class="drop-optic" id="vf-drop">DROP A TRAINABLE FILTER HERE</div>
            <div class="vision-controls">
              <button id="vf-step">EXPOSE 1 EPOCH</button><button class="hot" id="vf-train">TRAIN 300</button><button id="vf-exam">HIDDEN FILM REEL</button>
              <div><span>epoch</span><b id="vf-epoch">0</b></div><div><span>train</span><b id="vf-train-acc">—</b></div><div><span>hidden</span><b id="vf-hidden">—</b></div>
            </div>
          </main>
          <aside class="darkroom">
            <span class="vision-label">DARKROOM / SELECTED FILTER</span>
            <div id="vf-inspect"></div>
          </aside>
        </div>
        <footer class="vision-footer">
          <section><span class="vision-label">LOSS EXPOSURE</span><canvas id="vf-curve"></canvas></section>
          <section><span class="vision-label">ARCHITECTURE DIAGNOSIS</span><div id="vf-diagnosis"></div></section>
          <section><span class="vision-label">LAB LOG</span><div class="vision-log" id="vf-log"></div></section>
        </footer>
      </section>`;
    const inputCanvas=document.getElementById("vf-input"),miniInput=document.getElementById("vf-mini-input"),curve=document.getElementById("vf-curve"),bank=document.getElementById("vf-bank"),drop=document.getElementById("vf-drop");

    function persist(){save(META.id,state)}
    function addFilter(size){
      const m=blankModule(size,9100+state.modules.length*131+size);state.modules.push(m);selected=m.id;state.hidden=null;state.solved=false;logPush(state.logs,`mounted ${size}×${size} trainable filter`,"ok");persist();render();return m
    }
    function removeFilter(id){state.modules=state.modules.filter(m=>m.id!==id);if(selected===id)selected=state.modules[0]?.id||null;state.hidden=null;state.solved=false;logPush(state.logs,"filter removed","warn");persist();render()}
    function resetFilter(m,size=m.size){
      const freshM=blankModule(size,32000+state.epoch+size*17);m.size=size;m.kernel=freshM.kernel;m.bias=freshM.bias;m.outWeight=freshM.outWeight;state.epoch=0;state.history=[];state.hidden=null;state.solved=false;logPush(state.logs,`filter ${size}×${size} reinitialized`,"warn");persist()
    }
    function forward(ex){
      const mods=[],features=[];for(const m of state.modules){
        const cm=conv(ex.img,m.kernel,m.bias),red=reduceMap(cm.a,m.reducer),f=m.muted?0:red.value;mods.push({module:m,...cm,red,feature:f});features.push(f)
      }
      let z=state.outBias;for(let i=0;i<mods.length;i++)z+=mods[i].module.outWeight*features[i];return{mods,features,z,p:sigmoid(z)}
    }
    function metrics(data=TRAIN){
      if(!state.modules.length)return{loss:.693,acc:.5,errors:data};
      let loss=0,c=0,errors=[];for(const ex of data){const o=forward(ex);loss+=bce(o.p,ex.label);const ok=(o.p>=.5)==ex.label;c+=ok;if(!ok)errors.push({...ex,p:o.p})}return{loss:loss/data.length,acc:c/data.length,errors}
    }
    function trainEpoch(){
      if(!state.modules.length)return metrics();
      const g=state.modules.map(m=>({k:Array.from({length:m.size},()=>Array(m.size).fill(0)),b:0,out:0}));let gOut=0;
      for(const ex of TRAIN){
        const o=forward(ex),dz=o.p-ex.label;gOut+=dz/TRAIN.length;
        o.mods.forEach((mo,i)=>{
          const m=mo.module,gm=g[i];gm.out+=dz*mo.feature/TRAIN.length;if(m.muted)return;
          const dfeat=dz*m.outWeight,dm=Array.from({length:mo.a.length},()=>Array(mo.a.length).fill(0));
          for(const c of mo.red.chosen)dm[c.r][c.c]+=dfeat*c.w;
          for(let r=0;r<dm.length;r++)for(let c=0;c<dm.length;c++){
            const dzk=mo.z[r][c]>0?dm[r][c]:0;gm.b+=dzk/TRAIN.length;
            for(let kr=0;kr<m.size;kr++)for(let kc=0;kc<m.size;kc++)gm.k[kr][kc]+=dzk*ex.img[r+kr][c+kc]/TRAIN.length;
          }
        });
      }
      const lr=.055;state.outBias-=lr*gOut;
      state.modules.forEach((m,i)=>{if(m.frozen)return;m.outWeight-=lr*g[i].out;m.bias-=lr*g[i].b;for(let r=0;r<m.size;r++)for(let c=0;c<m.size;c++)m.kernel[r][c]-=lr*g[i].k[r][c]});
      state.epoch++;return metrics()
    }
    function train(n){
      if(!state.modules.length){ctx.toast("先把滤镜装到 optical track","bad");return}
      let m;for(let i=0;i<n;i++){m=trainEpoch();if(state.epoch===1||state.epoch%3===0)state.history.push(m.loss)}
      if(state.history.length>260)state.history.splice(0,state.history.length-260);state.hidden=null;logPush(state.logs,`exposed ${n} epochs · loss ${fmt(m.loss,4)} · acc ${pct(m.acc)}`,m.acc>.94?"ok":"");persist();render()
    }
    function exam(){
      const m=metrics(HIDDEN);state.hidden=m.acc;const pass=m.acc>=.95;logPush(state.logs,`hidden film reel ${pct(m.acc)} · ${pass?"PASS":"FAIL"}`,pass?"ok":"err");if(pass&&!state.solved){state.solved=true;ctx.complete()}persist();render()
    }
    function gradNorm(m){
      const before=JSON.stringify(m.kernel),sample=TRAIN.slice(0,24),idx=state.modules.indexOf(m);if(idx<0)return 0;
      let norm=0;
      for(const ex of sample){const o=forward(ex),dz=o.p-ex.label,mo=o.mods[idx],dfeat=dz*m.outWeight,dm=Array.from({length:mo.a.length},()=>Array(mo.a.length).fill(0));for(const c of mo.red.chosen)dm[c.r][c.c]+=dfeat*c.w;for(let r=0;r<dm.length;r++)for(let c=0;c<dm.length;c++){const dzk=mo.z[r][c]>0?dm[r][c]:0;for(let kr=0;kr<m.size;kr++)for(let kc=0;kc<m.size;kc++)norm+=Math.abs(dzk*ex.img[r+kr][c+kc])}}
      return norm/(sample.length*m.size*m.size)
    }
    function responseGap(m){
      const idx=state.modules.indexOf(m),a=[],b=[];TRAIN.slice(0,60).forEach(ex=>{const f=forward(ex).mods[idx]?.feature||0;(ex.label?b:a).push(f)});return Math.abs(mean(a)-mean(b))/(Math.abs(mean(a))+Math.abs(mean(b))+.05)
    }
    function renderBank(){
      const ex=TRAIN[state.sample],o=forward(ex);
      bank.innerHTML=state.modules.length?state.modules.map((m,i)=>`<article class="optic-card ${selected===m.id?"selected":""} ${m.muted?"muted":""}" data-filter="${m.id}"><div class="optic-head"><span>FILTER ${i+1}</span><b>${m.size}×${m.size}</b></div><div class="optic-views"><div><canvas id="kernel-${m.id}"></canvas><small>kernel</small></div><div><canvas id="map-${m.id}"></canvas><small>feature map</small></div></div><footer><span>${m.reducer.toUpperCase()}</span><b>f=${fmt(o.mods[i]?.feature,3)}</b></footer></article>`).join(""):'<div class="empty-optics">NO FILTERS INSTALLED</div>';
      state.modules.forEach((m,i)=>{const k=document.getElementById("kernel-"+m.id),fm=document.getElementById("map-"+m.id);if(k)drawMatrix(k,m.kernel,{signed:true});if(fm)drawMatrix(fm,o.mods[i].a,{signed:false})});
      document.querySelectorAll(".optic-card").forEach(c=>c.onclick=()=>{selected=c.dataset.filter;renderInspect();renderBank()});
    }
    function renderInspect(){
      const box=document.getElementById("vf-inspect"),m=state.modules.find(x=>x.id===selected);if(!m){box.innerHTML='<div class="dark-empty">安装并选择一个滤镜</div>';return}
      const gap=responseGap(m),gn=gradNorm(m);
      box.innerHTML=`<div class="dark-title"><b>FILTER ${state.modules.indexOf(m)+1}</b><span>learned pixels: ${m.size*m.size}</span></div>
        <label>KERNEL FOOTPRINT<select id="vf-size"><option value="3">3×3 local</option><option value="5">5×5 wide</option></select></label>
        <label>RESPONSE REDUCER<select id="vf-reducer"><option value="max">global max</option><option value="top2">top-2 average</option><option value="mean">global mean</option></select></label>
        <div class="dark-stats"><div><span>output weight</span><b>${fmt(m.outWeight,4)}</b></div><div><span>response gap</span><b>${fmt(gap,3)}</b></div><div><span>grad norm</span><b>${fmt(gn,5)}</b></div><div><span>bias</span><b>${fmt(m.bias,4)}</b></div></div>
        <div class="dark-buttons"><button id="vf-freeze" class="${m.frozen?"on":""}">❄ FREEZE</button><button id="vf-mute" class="${m.muted?"on":""}">⊘ MUTE</button><button id="vf-remove">REMOVE</button></div>
        <p>${m.reducer==="mean"?"平均整个 feature map 会把“出现在哪里”的局部峰值全部稀释。对于总亮度相近的横/竖线，这可能让方向证据消失。":gap<.12&&state.epoch>80?"这个滤镜对两类的 pooled response 很接近；它没有提供有用的可分特征。":"观察 kernel 中正负像素与 feature map 的亮区：模型正在把最终分类误差反传到每个卷积权重。"}</p>`;
      const sz=document.getElementById("vf-size");sz.value=m.size;sz.onchange=()=>{resetFilter(m,+sz.value);render()};
      const rd=document.getElementById("vf-reducer");rd.value=m.reducer;rd.onchange=()=>{m.reducer=rd.value;state.hidden=null;state.solved=false;persist();render()};
      document.getElementById("vf-freeze").onclick=()=>{m.frozen=!m.frozen;persist();renderInspect()};document.getElementById("vf-mute").onclick=()=>{m.muted=!m.muted;state.hidden=null;persist();render()};document.getElementById("vf-remove").onclick=()=>removeFilter(m.id);
    }
    function diagnosis(){
      const m=metrics();if(!state.modules.length)return"没有滤镜：分类器只剩 bias，任何训练都无法利用图像。";
      if(state.epoch>80&&m.acc<.75&&state.modules.every(x=>x.reducer==="mean"))return"所有滤镜都在做 global mean。横线和竖线总亮度几乎相同，局部方向证据在聚合时被抹掉了。";
      const gaps=state.modules.map(responseGap);if(state.epoch>80&&Math.max(...gaps)<.1)return"卷积核在变化，但所有 pooled feature 对两类响应仍接近。查看 feature maps，判断是 kernel 没学到方向还是 reducer 丢了信息。";
      if(state.modules.length>2&&m.acc>.95)return"已经能通关，但你用了较大的 filter bank。试着 mute 一个滤镜做消融：是否存在更小的结构？";
      return"结构改变 → kernel gradient 路径改变 → feature map 改变 → pooled feature 改变 → hidden position generalization 改变。";
    }
    function renderInput(){
      drawMatrix(inputCanvas,TRAIN[state.sample].img,{signed:false});drawMatrix(miniInput,TRAIN[state.sample].img,{signed:false});const o=forward(TRAIN[state.sample]);document.getElementById("vf-prob").textContent=fmt(o.p,3);document.getElementById("vf-needle").style.transform=`rotate(${-70+140*o.p}deg)`;document.getElementById("vf-sample-id").textContent="#"+state.sample;document.getElementById("vf-sample-label").textContent="label "+TRAIN[state.sample].label;
    }
    function render(){
      const m=metrics();renderInput();renderBank();renderInspect();drawLine(curve,state.history,{color:"#67dce4",label:"BCE"});document.getElementById("vf-epoch").textContent=state.epoch;document.getElementById("vf-train-acc").textContent=pct(m.acc);document.getElementById("vf-hidden").textContent=state.hidden===null?"—":pct(state.hidden);document.getElementById("vf-diagnosis").textContent=diagnosis();document.getElementById("vf-log").innerHTML=logHTML(state.logs);
    }
    document.querySelectorAll("[data-size]").forEach(b=>b.ondragstart=e=>e.dataTransfer.setData("text/x-filter-size",b.dataset.size));
    [drop,bank].forEach(z=>{z.ondragover=e=>e.preventDefault();z.ondrop=e=>{e.preventDefault();const s=+e.dataTransfer.getData("text/x-filter-size");if(s)addFilter(s)}});
    drop.ondblclick=()=>addFilter(3);document.getElementById("vf-prev").onclick=()=>{state.sample=(state.sample-1+TRAIN.length)%TRAIN.length;persist();render()};document.getElementById("vf-next").onclick=()=>{state.sample=(state.sample+1)%TRAIN.length;persist();render()};
    document.getElementById("vf-step").onclick=()=>train(1);document.getElementById("vf-train").onclick=()=>train(300);document.getElementById("vf-exam").onclick=exam;document.getElementById("vf-back").onclick=ctx.home;document.getElementById("vf-reset").onclick=ctx.reset;
    render();clean=resizeWatch(inputCanvas,render);
    window.__NC90_VISION__={state,addFilter,train,exam,metrics,forward,getState:()=>deepCopy(state),setReducer:(id,r)=>{const m=state.modules.find(x=>x.id===id);if(m){m.reducer=r;render()}}};
    return()=>{clean();delete window.__NC90_VISION__}
  }
};
