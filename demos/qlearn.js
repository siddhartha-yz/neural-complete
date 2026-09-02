import {seeded,fmt,pct,saveState,loadState,deepCopy,challengeHtml,logPush,logHtml,mean,clamp,argmax} from "../core.js";
import {demoShell,rangeControl,actions,bindRange} from "./util.js";

const META={
  id:"qlearn",code:"DEMO 05",kind:"REINFORCEMENT LEARNING",title:"策略训练场",en:"Q-Lab",
  color:"#78aee8",
  card:"不告诉智能体哪一步该往哪走，只给奖励。让 Q-learning 在探索中把未来回报写进状态-动作表，最后读出一张策略。",
  tags:["Q-Learning","Exploration","Reward Shaping"],
  status:"reinforcement learning · temporal-difference updates",
  subtitle:"你不能直接操纵 agent。你能设计的是奖励、探索率和学习参数；路线必须从经验里长出来。",
  missionTitle:"全图策略成功率 ≥ 90%",
  mission:"训练一个策略，让贪心 agent 从绝大多数安全起点都能到达目标，并把平均路径压到 18 步以内。",
  stageTitle:"POLICY FIELD",
  notices:[
    "Q 值不是地图上预先写好的箭头，而是每次 transition 后做 TD update。",
    "epsilon 决定探索与利用；太低可能早早锁死，太高则难以稳定利用。",
    "step penalty 是 reward shaping：它不告诉路线，但改变了“绕远路”的价值。"
  ]
};

const ROWS=6,COLS=6,ACTIONS=[[-1,0],[0,1],[1,0],[0,-1]],ARROWS=["↑","→","↓","←"];
const GOAL=[0,5],TRAPS=[[1,4],[3,3],[4,1]],WALLS=[[1,1],[1,2],[2,2],[2,4],[3,1],[4,4]];
const key=(r,c)=>r*COLS+c;
const same=(a,b)=>a[0]===b[0]&&a[1]===b[1];
const isIn=(arr,p)=>arr.some(x=>same(x,p));
function legal(r,c){return r>=0&&r<ROWS&&c>=0&&c<COLS&&!isIn(WALLS,[r,c])}
const SAFE=[];for(let r=0;r<ROWS;r++)for(let c=0;c<COLS;c++)if(legal(r,c)&&!same([r,c],GOAL)&&!isIn(TRAPS,[r,c]))SAFE.push([r,c]);

function fresh(){return {alpha:.35,gamma:.94,epsilon:.22,goalReward:1,stepReward:-.025,trapReward:-1.2,episode:0,Q:Array.from({length:ROWS*COLS},()=>[0,0,0,0]),successHistory:[],logs:[],eval:null,lastTrace:[],solved:false}}
function normalize(s){const z={...fresh(),...s};if(!Array.isArray(z.Q)||z.Q.length!==ROWS*COLS)z.Q=fresh().Q;return z}

export default {
  ...META,
  mount(root,ctx){
    let state=normalize(loadState(META.id)||fresh());
    const ui=demoShell(root,META);
    ui.controls.innerHTML=
      rangeControl({id:"q-eps",label:"EPSILON / 探索率",min:0,max:.6,step:.02,value:state.epsilon})+
      rangeControl({id:"q-alpha",label:"ALPHA / 学习率",min:.05,max:.8,step:.05,value:state.alpha})+
      rangeControl({id:"q-gamma",label:"GAMMA / 未来折扣",min:.5,max:.99,step:.01,value:state.gamma})+
      rangeControl({id:"q-step-r",label:"STEP REWARD / 每步奖励",min:-.1,max:.02,step:.005,value:state.stepReward})+
      rangeControl({id:"q-trap-r",label:"TRAP REWARD / 陷阱奖励",min:-2,max:-.2,step:.1,value:state.trapReward})+
      actions([
        {id:"q-episode",label:"运行 1 episode",icon:"›"},
        {id:"q-train",label:"训练 1000 episodes",icon:"▶",primary:true},
        {id:"q-eval",label:"全图策略评测",icon:"◆"},
        {id:"q-replay",label:"重放贪心策略",icon:"▷"},
        {id:"q-reset",label:"清空 Q-table",icon:"↺"}
      ]);
    ui.workspace.innerHTML=`
      <div class="viz-card">
        <div class="viz-head"><b>LEARNED POLICY</b><span>箭头 = argmaxₐ Q(s,a)</span></div>
        <div id="q-grid" class="gridworld" style="grid-template-columns:repeat(6,1fr)"></div>
      </div>
      <div class="stack">
        <div class="subpanel"><div class="subpanel-head"><b>TD UPDATE</b><span>Q ← Q + α[r + γ max Q′ − Q]</span></div><div id="q-td"></div></div>
        <div class="subpanel"><div class="subpanel-head"><b>RECENT LEARNING</b><span>rolling success</span></div><div id="q-history"></div></div>
        <div class="subpanel"><div class="subpanel-head"><b>RUN LOG</b><span>episodes</span></div><div class="log" id="q-log"></div></div>
      </div>`;
    ui.inspector.innerHTML='<span class="section-kicker">POLICY DIAGNOSTICS</span><div id="q-metrics"></div><div id="q-challenge"></div><div id="q-result"></div><div class="inline-note">训练 episode 会从多个安全起点随机开始。最终评测遍历整张地图，因此只背一条固定路线无法过关。</div>';

    let replayTimer=null;
    function transition(pos,a){
      const d=ACTIONS[a],np=[pos[0]+d[0],pos[1]+d[1]],next=legal(np[0],np[1])?np:pos;
      if(same(next,GOAL))return {next,reward:state.goalReward,done:true,kind:"goal"};
      if(isIn(TRAPS,next))return {next,reward:state.trapReward,done:true,kind:"trap"};
      return {next,reward:state.stepReward,done:false,kind:"step"};
    }
    function greedy(pos){return argmax(state.Q[key(...pos)])}
    function runEpisode(train=true,start=null){
      const rng=seeded(900001+state.episode*131+(train?0:777)),begin=start||SAFE[Math.floor(rng()*SAFE.length)];
      let pos=[...begin],trace=[[...pos]],total=0,last=null,success=false;
      for(let step=0;step<60;step++){
        let a;if(train&&rng()<state.epsilon)a=Math.floor(rng()*4);else a=greedy(pos);
        const tr=transition(pos,a);total+=tr.reward;
        if(train){
          const si=key(...pos),ni=key(...tr.next),old=state.Q[si][a],target=tr.reward+(tr.done?0:state.gamma*Math.max(...state.Q[ni]));
          state.Q[si][a]+=state.alpha*(target-old);last={s:[...pos],a,reward:tr.reward,target,old,newQ:state.Q[si][a]};
        }
        pos=[...tr.next];trace.push([...pos]);if(tr.done){success=tr.kind==="goal";break}
      }
      if(train){
        state.episode++;state.successHistory.push(success?1:0);if(state.successHistory.length>100)state.successHistory.shift();state.eval=null;
        if(state.episode<=3||state.episode%200===0)logPush(state.logs,"episode "+state.episode+" · "+(success?"goal":"failed")+" · "+(trace.length-1)+" steps",success?"ok":"warn");
      }
      state.lastTrace=trace;return {success,steps:trace.length-1,total,last,trace};
    }
    function train(n){
      let out;for(let i=0;i<n;i++)out=runEpisode(true);
      logPush(state.logs,"trained "+n+" episodes · rolling success "+pct(mean(state.successHistory)),mean(state.successHistory)>.8?"ok":"");
      saveState(META.id,state);render(out?.last)
    }
    function evaluate(){
      let good=0,steps=[];for(const s of SAFE){const r=runEpisode(false,s);if(r.success){good++;steps.push(r.steps)}}
      const ev={success:good/SAFE.length,avgSteps:steps.length?mean(steps):60};state.eval=ev;
      const pass=ev.success>=.9&&ev.avgSteps<=18;
      logPush(state.logs,"full-map eval "+pct(ev.success)+" · "+fmt(ev.avgSteps,1)+" steps"+(pass?" · PASS":" · FAIL"),pass?"ok":"err");
      if(pass&&!state.solved){state.solved=true;ctx.complete()}saveState(META.id,state);render();return ev
    }
    function reset(){state={...fresh(),alpha:state.alpha,gamma:state.gamma,epsilon:state.epsilon,stepReward:state.stepReward,trapReward:state.trapReward};logPush(state.logs,"Q-table cleared","warn");saveState(META.id,state);render()}
    function replay(){
      if(replayTimer)clearInterval(replayTimer);
      const r=runEpisode(false,[5,0]),trace=r.trace;let i=0;renderGrid(trace[0]);
      replayTimer=setInterval(()=>{i++;if(i>=trace.length){clearInterval(replayTimer);replayTimer=null;render();return}renderGrid(trace[i])},140)
    }
    function renderGrid(agent=null){
      const grid=document.getElementById("q-grid");if(!grid)return;grid.innerHTML="";
      let maxAbs=1e-6;for(const row of state.Q)for(const v of row)maxAbs=Math.max(maxAbs,Math.abs(v));
      for(let r=0;r<ROWS;r++)for(let c=0;c<COLS;c++){
        const p=[r,c],cell=document.createElement("div");cell.className="grid-cell";
        if(isIn(WALLS,p))cell.classList.add("wall");if(same(p,GOAL))cell.classList.add("goal");if(isIn(TRAPS,p))cell.classList.add("trap");if(same(p,[5,0]))cell.classList.add("start");
        const q=state.Q[key(r,c)],a=argmax(q),qv=q[a];
        if(legal(r,c)&&!same(p,GOAL)&&!isIn(TRAPS,p))cell.innerHTML='<span class="policy">'+ARROWS[a]+'</span><span class="qmini">'+fmt(qv,2)+'</span>';
        if(same(p,GOAL))cell.innerHTML='<span style="font-size:16px;color:#75d6a1">◎</span>';
        if(isIn(TRAPS,p))cell.innerHTML='<span style="font-size:15px;color:#ef7674">×</span>';
        if(agent&&same(agent,p))cell.innerHTML+='<span class="agent"></span>';
        grid.appendChild(cell);
      }
    }
    function render(last=null){
      ui.epoch.textContent="EPISODE "+state.episode;ui.status.textContent=state.episode?"POLICY LEARNING":"Q = 0";
      document.getElementById("q-log").innerHTML=logHtml(state.logs);renderGrid();
      const rolling=mean(state.successHistory);
      document.getElementById("q-history").innerHTML='<div class="metric-strip"><div class="metric '+(rolling>.8?"good":"")+'"><span>last '+state.successHistory.length+'</span><b>'+pct(rolling)+'</b></div><div class="metric"><span>epsilon</span><b>'+fmt(state.epsilon,2)+'</b></div><div class="metric"><span>states</span><b>'+SAFE.length+'</b></div></div><div class="progress" style="margin-top:8px"><i style="width:'+rolling*100+'%;background:#78aee8"></i></div>';
      const td=last||null;document.getElementById("q-td").innerHTML=td?
        '<div class="feature-table"><table class="feature-table"><tr><td>state</td><td>['+td.s.join(",")+']</td></tr><tr><td>action</td><td>'+ARROWS[td.a]+'</td></tr><tr><td>reward</td><td>'+fmt(td.reward,3)+'</td></tr><tr><td>old Q</td><td>'+fmt(td.old,3)+'</td></tr><tr><td>TD target</td><td>'+fmt(td.target,3)+'</td></tr><tr><td>new Q</td><td>'+fmt(td.newQ,3)+'</td></tr></table></div>':
        '<div class="empty-state" style="height:110px"><b>Run an episode</b><span>最近一次 TD update 会显示在这里</span></div>';
      const ev=state.eval;
      document.getElementById("q-metrics").innerHTML=
        '<div class="metric-card"><span class="section-kicker">ROLLING TRAIN SUCCESS</span><div class="big">'+pct(rolling)+'</div><small>last '+state.successHistory.length+' exploratory episodes</small></div>'+
        '<div class="metric-strip"><div class="metric '+(ev?.success>=.9?"good":"")+'"><span>all starts</span><b>'+pct(ev?.success)+'</b></div><div class="metric '+(ev?.avgSteps<=18?"good":"")+'"><span>avg steps</span><b>'+fmt(ev?.avgSteps,1)+'</b></div><div class="metric"><span>episodes</span><b>'+state.episode+'</b></div></div>';
      document.getElementById("q-challenge").innerHTML=challengeHtml({
        status:state.solved?"PASSED":!ev?"NOT TESTED":ev.success>=.9&&ev.avgSteps<=18?"PASSED":"FAILED",
        body:"目标：全图安全起点成功率 ≥ 90%，成功路径平均 ≤ 18 步。评测时 epsilon=0，只读取学到的 Q-table。",
        progress:ev?Math.min(1,ev.success)*(ev.avgSteps<=18?1:.82):rolling*.65,pass:state.solved,fail:!!ev&&!state.solved,
        badges:[{text:"TD LEARNING",kind:"good"},{text:"ε="+fmt(state.epsilon,2)},{text:"STEP "+fmt(state.stepReward,3),kind:state.stepReward<0?"good":""}]
      });
      document.getElementById("q-result").innerHTML=!ev?'<div class="result-banner">训练成功率包含探索动作；真正策略要用 epsilon=0 的全图评测。</div>':
        '<div class="result-banner '+(state.solved?"pass":"fail")+'"><strong>'+pct(ev.success)+'</strong> success · <strong>'+fmt(ev.avgSteps,1)+'</strong> avg steps · '+(state.solved?"奖励与探索共同塑造出稳定的短路径策略。":"继续调整探索/奖励并训练；不要直接给 agent 指路。")+'</div>';
    }

    bindRange("q-eps",v=>{state.epsilon=v;saveState(META.id,state)},v=>Number(v).toFixed(2));
    bindRange("q-alpha",v=>{state.alpha=v;saveState(META.id,state)},v=>Number(v).toFixed(2));
    bindRange("q-gamma",v=>{state.gamma=v;saveState(META.id,state)},v=>Number(v).toFixed(2));
    bindRange("q-step-r",v=>{state.stepReward=v;state.eval=null;saveState(META.id,state);render()},v=>Number(v).toFixed(3));
    bindRange("q-trap-r",v=>{state.trapReward=v;state.eval=null;saveState(META.id,state);render()},v=>Number(v).toFixed(1));
    document.getElementById("q-episode").onclick=()=>{const r=runEpisode(true);saveState(META.id,state);render(r.last)};
    document.getElementById("q-train").onclick=()=>train(1000);document.getElementById("q-eval").onclick=evaluate;document.getElementById("q-replay").onclick=replay;document.getElementById("q-reset").onclick=reset;
    render();
    window.__NC5_QLEARN__={state,getState:()=>deepCopy(state),train,evaluate,runEpisode,reset};
    return ()=>{if(replayTimer)clearInterval(replayTimer);delete window.__NC5_QLEARN__}
  }
};
