import {seeded,argmax,mean,clamp,pct,fmt,save,load,deepCopy,logPush,logHTML} from "../core.js";

const META={
  id:"policy-garden",theme:"blue",verb:"PLANT / SENSE / LEARN",title:"策略花园",en:"Policy Garden",
  card:"你不能移动 agent。你只能决定它能感知什么，并在环境里种有限的 reward beacons；策略必须由 TD 经验自己长出来。",
  tags:["state representation","reward shaping","Q-learning"]
};
const R=7,C=7,GOAL=[0,6],TRAPS=[[1,5],[3,3],[5,1]],WALLS=[[1,1],[1,2],[2,2],[2,4],[3,1],[3,4],[4,4],[5,4]];
const ACTIONS=[[-1,0],[0,1],[1,0],[0,-1]],ARROWS=["↑","→","↓","←"];
const SENSORS=[
  {id:"row",name:"ROW",symbol:"R",note:"absolute row coordinate"},
  {id:"col",name:"COLUMN",symbol:"C",note:"absolute column coordinate"},
  {id:"dx",name:"GOAL ΔX",symbol:"Δx",note:"horizontal distance to goal"},
  {id:"dy",name:"GOAL ΔY",symbol:"Δy",note:"vertical distance to goal"},
  {id:"goalDir",name:"GOAL DIR",symbol:"◎",note:"sign of direction to goal"},
  {id:"walls",name:"WALL RADAR",symbol:"▦",note:"4-bit blocked-neighbor mask"},
  {id:"danger",name:"DANGER",symbol:"!",note:"4-bit adjacent-trap mask"},
  {id:"region",name:"LANDMARK REGION",symbol:"◇",note:"coarse 3×3 map region"}
];
const same=(a,b)=>a[0]===b[0]&&a[1]===b[1],has=(arr,p)=>arr.some(x=>same(x,p)),key=(r,c)=>r*C+c;
function legal(p){return p[0]>=0&&p[0]<R&&p[1]>=0&&p[1]<C&&!has(WALLS,p)}
const SAFE=[];for(let r=0;r<R;r++)for(let c=0;c<C;c++){const p=[r,c];if(legal(p)&&!same(p,GOAL)&&!has(TRAPS,p))SAFE.push(p)}
function fresh(){return{sensors:[null,null,null],beacons:[],Q:{},episode:0,successHistory:[],logs:[],eval:null,last:null,solved:false,paint:false}}
function restored(){return{...fresh(),...(load(META.id)||{})}}

export default {
  ...META,
  mount(root,ctx){
    let state=restored(),replayTimer=null,agent=null;
    root.innerHTML=`
      <section class="garden-screen">
        <header class="garden-head">
          <button id="pg-back">← LABS</button>
          <div><span>EXPERIMENT 05 · REINFORCEMENT ECOSYSTEM</span><h1>POLICY GARDEN</h1></div>
          <div class="garden-contract"><small>HARVEST CONDITION</small><b>≥90% starts · ≤18 avg steps</b></div>
          <button id="pg-reset">UPROOT POLICY</button>
        </header>
        <div class="garden-main">
          <aside class="sensor-shed">
            <span class="garden-label">SENSOR SEED PACKS</span>
            <p>拖进右侧 brain trellis。最多 3 个。Q-table 只能区分你允许 agent 感知的状态。</p>
            <div id="pg-sensors"></div>
            <div class="shed-note"><b>不能直接操纵 agent。</b><span>你构造的是 state representation，不是路线。</span></div>
          </aside>
          <main class="garden-field-wrap">
            <div class="field-toolbar">
              <div><span class="garden-label">LIVE HABITAT</span><b id="pg-mode">POLICY VIEW</b></div>
              <button id="pg-beacon-brush">✦ PLANT REWARD BEACON <span id="pg-budget">3 left</span></button>
              <button id="pg-clear-beacons">clear beacons</button>
            </div>
            <div class="garden-field" id="pg-grid"></div>
            <div class="field-key"><span><i class="goal"></i>goal</span><span><i class="trap"></i>trap</span><span><i class="wall"></i>wall</span><span><i class="beacon"></i>one-shot +0.12</span></div>
          </main>
          <aside class="brain-trellis">
            <span class="garden-label">AGENT BRAIN / STATE KEY</span>
            <div id="pg-slots" class="sensor-slots"></div>
            <div class="state-preview"><span>CURRENT ABSTRACT STATE</span><b id="pg-state-key">blind</b></div>
            <div class="alias-meter"><span>state aliasing</span><b id="pg-alias">—</b><i><em id="pg-alias-bar"></em></i></div>
            <div class="reward-card"><span>FIXED REWARDS</span><div><b>goal</b><strong>+1.00</strong></div><div><b>trap</b><strong>−1.00</strong></div><div><b>step</b><strong>−0.025</strong></div><div><b>beacon</b><strong>+0.12 once</strong></div></div>
            <div id="pg-diagnosis" class="garden-diagnosis"></div>
          </aside>
        </div>
        <footer class="garden-console">
          <div class="garden-actions"><button id="pg-one">RUN 1 EPISODE</button><button class="hot" id="pg-train">GROW 1500 EPISODES</button><button id="pg-eval">HARVEST / ALL STARTS</button><button id="pg-replay">REPLAY GREEDY</button></div>
          <div class="garden-stats"><div><span>episodes</span><b id="pg-episodes">0</b></div><div><span>rolling success</span><b id="pg-rolling">—</b></div><div><span>all starts</span><b id="pg-success">—</b></div><div><span>avg steps</span><b id="pg-steps">—</b></div></div>
          <div class="td-scope"><span>LAST TD UPDATE</span><div id="pg-td"></div></div>
          <div class="garden-log" id="pg-log"></div>
        </footer>
      </section>`;
    const grid=document.getElementById("pg-grid");

    function persist(){save(META.id,state)}
    function sensorValue(id,p){
      const [r,c]=p;if(id==="row")return r;if(id==="col")return c;if(id==="dx")return GOAL[1]-c;if(id==="dy")return GOAL[0]-r;
      if(id==="goalDir")return(Math.sign(GOAL[0]-r)+1)*3+(Math.sign(GOAL[1]-c)+1);
      if(id==="walls"){let m=0;ACTIONS.forEach((d,i)=>{const n=[r+d[0],c+d[1]];if(!legal(n))m|=1<<i});return m}
      if(id==="danger"){let m=0;ACTIONS.forEach((d,i)=>{const n=[r+d[0],c+d[1]];if(has(TRAPS,n))m|=1<<i});return m}
      if(id==="region")return Math.floor(r/3)*3+Math.floor(c/3);
      return 0
    }
    function stateKey(p){
      const ids=state.sensors.filter(Boolean);return ids.length?ids.map(id=>id+":"+sensorValue(id,p)).join("|"):"blind"
    }
    function qrow(k){if(!state.Q[k])state.Q[k]=[0,0,0,0];return state.Q[k]}
    function transition(p,a,visitedBeacons){
      const d=ACTIONS[a],tryP=[p[0]+d[0],p[1]+d[1]],next=legal(tryP)?tryP:p;
      let reward=-.025,done=false,kind="step";
      if(same(next,GOAL)){reward=1;done=true;kind="goal"}
      else if(has(TRAPS,next)){reward=-1;done=true;kind="trap"}
      else{const bi=state.beacons.findIndex(b=>same(b,next));if(bi>=0&&!visitedBeacons.has(bi)){visitedBeacons.add(bi);reward+=.12;kind="beacon"}}
      return{next,reward,done,kind}
    }
    function greedy(p){return argmax(qrow(stateKey(p)))}
    function runEpisode(train=true,start=null){
      const rng=seeded(990001+state.episode*197+(train?0:555)),begin=start||SAFE[Math.floor(rng()*SAFE.length)];let p=[...begin],trace=[[...p]],total=0,last=null,success=false,visited=new Set();
      for(let step=0;step<70;step++){
        const sk=stateKey(p),row=qrow(sk),a=train&&rng()<.22?Math.floor(rng()*4):argmax(row),tr=transition(p,a,visited),nk=stateKey(tr.next),old=row[a],target=tr.reward+(tr.done?0:.94*Math.max(...qrow(nk)));
        if(train)row[a]+=.35*(target-old);
        last={state:sk,pos:[...p],a,reward:tr.reward,old,target,newQ:row[a],next:nk};total+=tr.reward;p=[...tr.next];trace.push([...p]);if(tr.done){success=tr.kind==="goal";break}
      }
      if(train){state.episode++;state.successHistory.push(success?1:0);if(state.successHistory.length>120)state.successHistory.shift();state.eval=null;state.last={trace,last,success,total};if(state.episode<=3||state.episode%250===0)logPush(state.logs,`episode ${state.episode} · ${success?"goal":"failed"} · ${trace.length-1} steps`,success?"ok":"warn")}
      return{success,steps:trace.length-1,total,trace,last}
    }
    function train(n){if(!state.sensors.some(Boolean)){ctx.toast("blind agent 也能训练，但它无法区分位置","bad")}let r;for(let i=0;i<n;i++)r=runEpisode(true);logPush(state.logs,`grew ${n} episodes · rolling ${pct(mean(state.successHistory))}`,mean(state.successHistory)>.8?"ok":"");persist();render(r?.last)}
    function evaluate(){
      let ok=0,steps=[];for(const s of SAFE){const r=runEpisode(false,s);if(r.success){ok++;steps.push(r.steps)}}state.eval={success:ok/SAFE.length,avgSteps:steps.length?mean(steps):70};const pass=state.eval.success>=.9&&state.eval.avgSteps<=18;
      logPush(state.logs,`harvest ${pct(state.eval.success)} · ${fmt(state.eval.avgSteps,1)} avg steps · ${pass?"PASS":"FAIL"}`,pass?"ok":"err");if(pass&&!state.solved){state.solved=true;ctx.complete()}persist();render();return state.eval
    }
    function aliasing(){
      const groups={};for(const p of SAFE){const k=stateKey(p);(groups[k]??=[]).push(p)}const aliased=Object.values(groups).filter(v=>v.length>1).reduce((s,v)=>s+v.length,0);return{groups:Object.keys(groups).length,aliased:aliased/SAFE.length,max:Math.max(...Object.values(groups).map(v=>v.length))}
    }
    function conflictEstimate(){
      const a=aliasing();if(!state.sensors.some(Boolean))return 1;
      let bad=0,total=0;const groups={};for(const p of SAFE){const k=stateKey(p);(groups[k]??=[]).push(p)}
      for(const poses of Object.values(groups)){if(poses.length<2)continue;const dirs=new Set(poses.map(p=>{const dr=GOAL[0]-p[0],dc=GOAL[1]-p[1];return Math.abs(dc)>=Math.abs(dr)?(dc>0?1:3):(dr>0?2:0)}));if(dirs.size>1)bad+=poses.length;total+=poses.length}
      return total?bad/total:0
    }
    function loopInTrace(){const t=state.last?.trace||[],seen=new Set();for(const p of t){const k=p.join(",");if(seen.has(k))return true;seen.add(k)}return false}
    function diagnosis(){
      const a=aliasing(),conf=conflictEstimate();if(!state.sensors.some(Boolean))return"BLIND STATE：所有格子共用同一行 Q-values。来自不同位置的 TD target 会互相覆盖。";
      if(conf>.4)return"STATE ALIASING：多个需要不同动作的位置被压成同一个 abstract state。观察右侧 aliasing，再添加能区分这些位置的 sensor。";
      if(loopInTrace())return"LOOP DETECTED：最近轨迹重复访问同一格。当前表示 / reward 让局部循环与向目标前进难以区分。";
      if(state.beacons.length&&mean(state.successHistory)<.5&&state.episode>300)return"Reward beacons 改变了局部 TD target，但并没有自动补足缺失的状态信息。奖励不能替代观测。";
      if(state.sensors.filter(Boolean).length===3&&a.aliased<.05)return"状态几乎完全可辨。试着移除一个 sensor 做消融：是否能用更紧凑的表示获得同一策略？";
      return"每次替换 sensor，Q-table 的“状态地址空间”都会改变。结构先决定 agent 能区分什么，经验才在这些地址上写入动作价值。";
    }
    function installSensor(slot,id){
      if(state.sensors.includes(id)&&state.sensors[slot]!==id){ctx.toast("同一个 sensor 不需要装两次","bad");return}
      state.sensors[slot]=id;state.Q={};state.episode=0;state.successHistory=[];state.eval=null;state.solved=false;state.last=null;logPush(state.logs,"brain representation rebuilt","warn");persist();render()
    }
    function removeSensor(slot){state.sensors[slot]=null;state.Q={};state.episode=0;state.successHistory=[];state.eval=null;state.solved=false;persist();render()}
    function plant(p){
      if(!state.paint)return;if(!legal(p)||same(p,GOAL)||has(TRAPS,p))return;
      const i=state.beacons.findIndex(b=>same(b,p));if(i>=0)state.beacons.splice(i,1);else if(state.beacons.length<3)state.beacons.push([...p]);else{ctx.toast("reward beacon budget 已用完","bad");return}
      state.Q={};state.episode=0;state.successHistory=[];state.eval=null;state.solved=false;persist();render()
    }
    function renderSensors(){
      document.getElementById("pg-sensors").innerHTML=SENSORS.map(s=>`<button class="sensor-pack" draggable="true" data-sensor="${s.id}"><b>${s.symbol}</b><span>${s.name}</span><small>${s.note}</small></button>`).join("");
      document.querySelectorAll(".sensor-pack").forEach(s=>s.ondragstart=e=>e.dataTransfer.setData("text/x-sensor",s.dataset.sensor));
      document.getElementById("pg-slots").innerHTML=state.sensors.map((id,i)=>{const s=SENSORS.find(x=>x.id===id);return`<div class="brain-slot ${id?"filled":""}" data-slot="${i}"><span>S${i+1}</span>${s?`<b>${s.symbol} ${s.name}</b><button data-rm="${i}">×</button>`:"<b>DROP SENSOR</b>"}</div>`}).join("");
      document.querySelectorAll(".brain-slot").forEach(s=>{s.ondragover=e=>e.preventDefault();s.ondrop=e=>{e.preventDefault();const id=e.dataTransfer.getData("text/x-sensor");if(id)installSensor(+s.dataset.slot,id)}});
      document.querySelectorAll("[data-rm]").forEach(b=>b.onclick=()=>removeSensor(+b.dataset.rm))
    }
    function renderGrid(pos=agent){
      grid.innerHTML="";grid.style.gridTemplateColumns=`repeat(${C},1fr)`;for(let r=0;r<R;r++)for(let c=0;c<C;c++){const p=[r,c],cell=document.createElement("button");cell.className="garden-cell";if(has(WALLS,p))cell.classList.add("wall");if(has(TRAPS,p))cell.classList.add("trap");if(same(p,GOAL))cell.classList.add("goal");if(has(state.beacons,p))cell.classList.add("beacon");const q=state.Q[stateKey(p)]||[0,0,0,0],a=argmax(q);
        if(legal(p)&&!same(p,GOAL)&&!has(TRAPS,p))cell.innerHTML=`<span class="policy-arrow">${ARROWS[a]}</span><small>${fmt(q[a],2)}</small>`;if(same(p,GOAL))cell.innerHTML='<b class="goal-mark">◎</b>';if(has(TRAPS,p))cell.innerHTML='<b class="trap-mark">×</b>';if(has(state.beacons,p))cell.innerHTML+='<i class="beacon-mark">✦</i>';if(pos&&same(pos,p))cell.innerHTML+='<em class="garden-agent"></em>';
        cell.onclick=()=>plant(p);grid.appendChild(cell)}
    }
    function replay(){
      if(replayTimer)clearInterval(replayTimer);const r=runEpisode(false,[6,0]),trace=r.trace;let i=0;agent=trace[0];renderGrid(agent);replayTimer=setInterval(()=>{i++;if(i>=trace.length){clearInterval(replayTimer);replayTimer=null;agent=null;render();return}agent=trace[i];renderGrid(agent)},140)
    }
    function tdHTML(last){
      if(!last)return'<span>run an episode to inspect a temporal-difference update</span>';
      return`<b>${last.state}</b><i>${ARROWS[last.a]}</i><span>r ${fmt(last.reward,3)}</span><span>old ${fmt(last.old,3)}</span><span>target ${fmt(last.target,3)}</span><strong>new ${fmt(last.newQ,3)}</strong>`
    }
    function render(last=state.last?.last){
      renderSensors();renderGrid();const al=aliasing(),rolling=mean(state.successHistory);document.getElementById("pg-state-key").textContent=stateKey([6,0]);document.getElementById("pg-alias").textContent=pct(al.aliased);document.getElementById("pg-alias-bar").style.width=clamp(al.aliased,0,1)*100+"%";document.getElementById("pg-budget").textContent=(3-state.beacons.length)+" left";document.getElementById("pg-mode").textContent=state.paint?"BEACON PAINT MODE":"POLICY VIEW";document.getElementById("pg-beacon-brush").classList.toggle("on",state.paint);
      document.getElementById("pg-diagnosis").textContent=diagnosis();document.getElementById("pg-episodes").textContent=state.episode;document.getElementById("pg-rolling").textContent=pct(rolling);document.getElementById("pg-success").textContent=pct(state.eval?.success);document.getElementById("pg-steps").textContent=fmt(state.eval?.avgSteps,1);document.getElementById("pg-td").innerHTML=tdHTML(last);document.getElementById("pg-log").innerHTML=logHTML(state.logs)
    }
    document.getElementById("pg-beacon-brush").onclick=()=>{state.paint=!state.paint;persist();render()};document.getElementById("pg-clear-beacons").onclick=()=>{state.beacons=[];state.Q={};state.episode=0;state.successHistory=[];state.eval=null;persist();render()};
    document.getElementById("pg-one").onclick=()=>{const r=runEpisode(true);persist();render(r.last)};document.getElementById("pg-train").onclick=()=>train(1500);document.getElementById("pg-eval").onclick=evaluate;document.getElementById("pg-replay").onclick=replay;document.getElementById("pg-back").onclick=ctx.home;document.getElementById("pg-reset").onclick=ctx.reset;
    render();
    window.__NC90_POLICY__={state,train,evaluate,runEpisode,installSensor,plantAt:p=>{state.paint=true;plant(p);state.paint=false},aliasing,getState:()=>deepCopy(state)};
    return()=>{if(replayTimer)clearInterval(replayTimer);delete window.__NC90_POLICY__}
  }
};
