import {seeded,randn,sigmoid,tanh,relu,bce,pct,fmt,saveState,loadState,deepCopy,drawLineChart,drawAxes,challengeHtml,logPush,logHtml,resizeObserver,clamp} from "../core.js";
import {demoShell,selectControl,rangeControl,actions,bindRange} from "./util.js";

const META={
  id:"xor",code:"DEMO 02",kind:"REPRESENTATION LEARNING",title:"异或工坊",en:"XOR Workshop",
  color:"#b8a0e3",
  card:"让一个真正的小型 MLP 用反向传播学会 XOR。线性模型会系统性失败；隐藏层必须自己形成新的内部表示。",
  tags:["MLP","Backpropagation","Nonlinearity"],
  status:"representation learning · backpropagation",
  subtitle:"你不负责写出 XOR 公式。你只决定网络容量与激活函数，然后看隐藏单元能不能学出把空间“折起来”的表示。",
  missionTitle:"隐藏评测 ≥ 94%",
  mission:"训练网络把同号象限判为 0、异号象限判为 1。样本带坐标抖动，隐藏评测使用另一批未见点。",
  stageTitle:"NONLINEAR DECISION FIELD",
  notices:[
    "输出层的误差会通过链式法则回传到每个隐藏单元。",
    "没有足够隐藏容量时，优化不是“再跑久一点”就能解决。",
    "点击决策场任意位置，可以查看该点的隐藏层激活。"
  ]
};

function data(seed,n,noise=.18){
  const rng=seeded(seed),pts=[];
  const centers=[[-.72,-.72,0],[-.72,.72,1],[.72,-.72,1],[.72,.72,0]];
  for(let i=0;i<n;i++){
    const c=centers[i%4];
    pts.push({x:clamp(c[0]+randn(rng)*noise,-.98,.98),y:clamp(c[1]+randn(rng)*noise,-.98,.98),label:c[2]});
  }
  return pts;
}
const TRAIN=data(4321,160,.16),HIDDEN=data(91827,260,.20);

function fresh(){return {hidden:1,activation:"tanh",lr:.22,epoch:0,W1:[],b1:[],W2:[],b2:0,history:[],hiddenAcc:null,logs:[],probe:{x:.25,y:.25},solved:false}}
function initWeights(state){
  const rng=seeded(700+state.hidden*17+(state.activation==="relu"?91:0));
  state.W1=Array.from({length:state.hidden},()=>[randn(rng)*.7,randn(rng)*.7]);
  state.b1=Array.from({length:state.hidden},()=>randn(rng)*.08);
  state.W2=Array.from({length:state.hidden},()=>randn(rng)*.7);
  state.b2=0;state.epoch=0;state.history=[];state.hiddenAcc=null;state.solved=false;
}
function normalize(s){
  const z={...fresh(),...s};
  if(!Array.isArray(z.W1)||z.W1.length!==z.hidden||z.W1.some(r=>!Array.isArray(r)||r.length!==2))initWeights(z);
  return z;
}

export default {
  ...META,
  mount(root,ctx){
    let state=normalize(loadState(META.id)||fresh());
    const ui=demoShell(root,META);
    ui.controls.innerHTML=
      rangeControl({id:"xor-hidden",label:"HIDDEN UNITS / 隐藏单元",min:1,max:4,step:1,value:state.hidden})+
      selectControl({id:"xor-act",label:"ACTIVATION / 激活函数",value:state.activation,options:[
        {value:"linear",label:"Linear · 无非线性"},
        {value:"tanh",label:"tanh · 双侧饱和"},
        {value:"relu",label:"ReLU · 分段线性"}
      ]})+
      rangeControl({id:"xor-lr",label:"LEARNING RATE / 学习率",min:.03,max:.5,step:.01,value:state.lr})+
      actions([
        {id:"xor-step",label:"1 epoch",icon:"›"},
        {id:"xor-train",label:"训练 800 epochs",icon:"▶",primary:true},
        {id:"xor-exam",label:"隐藏评测",icon:"◆"},
        {id:"xor-reset",label:"重新初始化",icon:"↺"}
      ]);
    ui.workspace.innerHTML=`
      <div class="viz-card">
        <div class="viz-head"><b>DECISION FIELD</b><span>点击任意位置检查 hidden activations</span></div>
        <div class="canvas-wrap"><canvas id="xor-field"></canvas><div class="canvas-overlay"><span class="legend-chip">● y=0</span><span class="legend-chip">● y=1</span></div></div>
      </div>
      <div class="stack">
        <div class="subpanel"><div class="subpanel-head"><b>NETWORK STATE</b><span id="xor-arch"></span></div><canvas class="chart" id="xor-loss"></canvas></div>
        <div class="subpanel"><div class="subpanel-head"><b>PROBE / HIDDEN REPRESENTATION</b><span id="xor-probe-coord"></span></div><div id="xor-hidden-bars"></div></div>
        <div class="subpanel"><div class="subpanel-head"><b>RUN LOG</b><span>backprop events</span></div><div class="log" id="xor-log"></div></div>
      </div>`;
    ui.inspector.innerHTML='<span class="section-kicker">MODEL DIAGNOSTICS</span><div id="xor-metrics"></div><div id="xor-challenge"></div><div id="xor-result"></div>';

    const field=document.getElementById("xor-field"),loss=document.getElementById("xor-loss");
    let cleanupResize=()=>{};

    function act(z){
      if(state.activation==="linear")return z;
      if(state.activation==="relu")return relu(z);
      return tanh(z);
    }
    function dact(z){
      if(state.activation==="linear")return 1;
      if(state.activation==="relu")return z>0?1:0;
      const t=tanh(z);return 1-t*t;
    }
    function forward(p){
      const z1=[],h=[];
      for(let j=0;j<state.hidden;j++){const z=state.W1[j][0]*p.x+state.W1[j][1]*p.y+state.b1[j];z1.push(z);h.push(act(z))}
      let z2=state.b2;for(let j=0;j<state.hidden;j++)z2+=state.W2[j]*h[j];
      return {z1,h,z2,p:sigmoid(z2)};
    }
    function metrics(ds=TRAIN){
      let l=0,c=0;for(const p of ds){const o=forward(p);l+=bce(o.p,p.label);c+=(o.p>=.5)==p.label}
      return {loss:l/ds.length,acc:c/ds.length};
    }
    function oneEpoch(){
      const gW1=Array.from({length:state.hidden},()=>[0,0]),gb1=Array(state.hidden).fill(0),gW2=Array(state.hidden).fill(0);let gb2=0;
      for(const p of TRAIN){
        const o=forward(p),dz2=o.p-p.label;
        gb2+=dz2/TRAIN.length;
        for(let j=0;j<state.hidden;j++){
          gW2[j]+=dz2*o.h[j]/TRAIN.length;
          const dz1=dz2*state.W2[j]*dact(o.z1[j]);
          gW1[j][0]+=dz1*p.x/TRAIN.length;gW1[j][1]+=dz1*p.y/TRAIN.length;gb1[j]+=dz1/TRAIN.length;
        }
      }
      for(let j=0;j<state.hidden;j++){
        state.W1[j][0]-=state.lr*gW1[j][0];state.W1[j][1]-=state.lr*gW1[j][1];state.b1[j]-=state.lr*gb1[j];state.W2[j]-=state.lr*gW2[j];
      }
      state.b2-=state.lr*gb2;state.epoch++;
      const m=metrics();if(state.epoch===1||state.epoch%5===0)state.history.push(m.loss);if(state.history.length>260)state.history.shift();
      state.hiddenAcc=null;return m;
    }
    function train(n){
      let m;for(let i=0;i<n;i++)m=oneEpoch();
      logPush(state.logs,"backprop "+n+" epochs · loss "+fmt(m.loss,4)+" · train "+pct(m.acc),m.acc>.93?"ok":"");
      saveState(META.id,state);render();
    }
    function exam(){
      const m=metrics(HIDDEN);state.hiddenAcc=m.acc;const pass=m.acc>=.94;
      logPush(state.logs,"hidden XOR evaluation "+pct(m.acc)+(pass?" · PASS":" · FAIL"),pass?"ok":"err");
      if(pass&&!state.solved){state.solved=true;ctx.complete()}
      saveState(META.id,state);render();
    }
    function reset(){
      initWeights(state);logPush(state.logs,"weights randomized for "+state.hidden+" hidden units","warn");saveState(META.id,state);render()
    }
    function drawField(){
      const r=field.getBoundingClientRect(),dpr=Math.min(2,devicePixelRatio||1);field.width=Math.round(r.width*dpr);field.height=Math.round(r.height*dpr);
      const g=field.getContext("2d");g.setTransform(dpr,0,0,dpr,0,0);const w=r.width,h=r.height,pad=30;g.fillStyle="#0b1013";g.fillRect(0,0,w,h);
      const res=36,cw=(w-2*pad)/res,ch=(h-2*pad)/res;
      for(let iy=0;iy<res;iy++)for(let ix=0;ix<res;ix++){
        const x=-1+2*(ix+.5)/res,y=1-2*(iy+.5)/res,p=forward({x,y}).p;
        const a=[22,25,34],b=[116,75,155];
        g.fillStyle=`rgb(${a[0]+(b[0]-a[0])*p},${a[1]+(b[1]-a[1])*p},${a[2]+(b[2]-a[2])*p})`;g.fillRect(pad+ix*cw,pad+iy*ch,cw+.7,ch+.7);
      }
      drawAxes(g,w,h,pad);
      for(const p of TRAIN){
        const x=pad+(p.x+1)/2*(w-2*pad),y=pad+(1-p.y)/2*(h-2*pad);g.beginPath();g.arc(x,y,2.9,0,Math.PI*2);g.fillStyle=p.label?"#d0b3ff":"#62d4dc";g.fill()
      }
      const px=pad+(state.probe.x+1)/2*(w-2*pad),py=pad+(1-state.probe.y)/2*(h-2*pad);
      g.strokeStyle="#fff";g.lineWidth=1.4;g.beginPath();g.arc(px,py,6,0,Math.PI*2);g.stroke();g.beginPath();g.moveTo(px-9,py);g.lineTo(px+9,py);g.moveTo(px,py-9);g.lineTo(px,py+9);g.stroke();
    }
    function hiddenBars(){
      const o=forward(state.probe);
      document.getElementById("xor-probe-coord").textContent="("+fmt(state.probe.x,2)+", "+fmt(state.probe.y,2)+") → "+fmt(o.p,3);
      document.getElementById("xor-hidden-bars").innerHTML=o.h.map((v,i)=>{
        const norm=state.activation==="relu"?clamp(v/2,0,1):clamp((v+1)/2,0,1);
        return '<div class="control-row"><span>h'+(i+1)+' · z='+fmt(o.z1[i],2)+'</span><div style="width:95px;height:7px;background:#252d33"><i style="display:block;width:'+norm*100+'%;height:100%;background:#b8a0e3"></i></div><b>'+fmt(v,3)+'</b></div>'
      }).join("");
    }
    function render(){
      const m=metrics();ui.epoch.textContent="EPOCH "+state.epoch;ui.status.textContent=state.epoch?"BACKPROP ACTIVE":"UNTRAINED";
      document.getElementById("xor-arch").textContent="2 → "+state.hidden+" "+state.activation+" → 1";
      document.getElementById("xor-log").innerHTML=logHtml(state.logs);hiddenBars();
      document.getElementById("xor-metrics").innerHTML=
        '<div class="metric-card"><span class="section-kicker">BCE LOSS</span><div class="big">'+fmt(m.loss,4)+'</div><small>gradient computed through every layer</small></div>'+
        '<div class="metric-strip"><div class="metric '+(m.acc>=.94?"good":"")+'"><span>train</span><b>'+pct(m.acc)+'</b></div><div class="metric '+(state.hiddenAcc>=.94?"good":"")+'"><span>hidden</span><b>'+pct(state.hiddenAcc)+'</b></div><div class="metric"><span>units</span><b>'+state.hidden+'</b></div></div>';
      document.getElementById("xor-challenge").innerHTML=challengeHtml({
        status:state.solved?"PASSED":state.hiddenAcc===null?"NOT TESTED":state.hiddenAcc>=.94?"PASSED":"FAILED",
        body:"目标：隐藏集 ≥ 94%。线性激活会把多层网络重新坍缩成一个线性变换；真正的关键不是“层数”三个字，而是中间表示能否非线性地改变几何。",
        progress:state.hiddenAcc??m.acc*.75,pass:state.solved,fail:state.hiddenAcc!==null&&state.hiddenAcc<.94,
        badges:[{text:"CHAIN RULE",kind:"good"},{text:state.activation.toUpperCase(),kind:state.activation==="linear"?"bad":"good"},{text:state.hidden+" HIDDEN"}]
      });
      document.getElementById("xor-result").innerHTML=state.hiddenAcc===null?'<div class="result-banner">训练集只是第一关。最终分数来自另一批带抖动的 XOR 点。</div>':
        '<div class="result-banner '+(state.hiddenAcc>=.94?"pass":"fail")+'"><strong>'+pct(state.hiddenAcc)+'</strong> hidden accuracy · '+(state.hiddenAcc>=.94?"隐藏层学到的分区规则泛化到了未见点。":"检查容量与非线性；继续堆 epochs 可能不会改变表示能力。")+'</div>';
      drawField();drawLineChart(loss,[{values:state.history,color:"#b8a0e3"}],{label:"BCE"});
    }

    bindRange("xor-hidden",v=>{state.hidden=v;initWeights(state);saveState(META.id,state);render()},v=>v);
    document.getElementById("xor-act").addEventListener("change",e=>{state.activation=e.target.value;initWeights(state);saveState(META.id,state);render()});
    bindRange("xor-lr",v=>{state.lr=v;saveState(META.id,state)},v=>Number(v).toFixed(2));
    document.getElementById("xor-step").onclick=()=>train(1);
    document.getElementById("xor-train").onclick=()=>train(800);
    document.getElementById("xor-exam").onclick=exam;
    document.getElementById("xor-reset").onclick=reset;
    field.addEventListener("click",e=>{
      const r=field.getBoundingClientRect(),pad=30;state.probe.x=clamp((e.clientX-r.left-pad)/(r.width-2*pad)*2-1,-1,1);state.probe.y=clamp(1-(e.clientY-r.top-pad)/(r.height-2*pad)*2,-1,1);render()
    });
    render();cleanupResize=resizeObserver(field,drawField);
    window.__NC5_XOR__={state,getState:()=>deepCopy(state),train,exam,metrics,forward,reset};
    return ()=>{cleanupResize();delete window.__NC5_XOR__}
  }
};
