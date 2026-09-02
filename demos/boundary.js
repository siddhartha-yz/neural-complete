import {seeded,randn,sigmoid,bce,mean,pct,fmt,saveState,loadState,deepCopy,drawLineChart,clearCanvas,drawAxes,clamp,challengeHtml,logPush,logHtml,resizeObserver} from "../core.js";
import {demoShell,selectControl,rangeControl,actions,bindRange} from "./util.js";

const META={
  id:"boundary",code:"DEMO 01",kind:"SUPERVISED LEARNING",title:"边界铸造厂",en:"Boundary Foundry",
  color:"#ffb347",
  card:"给模型看带标签的二维样本，选择表示方式，再用真实梯度下降学习概率边界。错误的特征表示会让优化再久也无能为力。",
  tags:["Logistic Regression","Gradient Descent","Generalization"],
  status:"supervised learning · gradient descent",
  subtitle:"不是手动画分类线。你只决定模型能看见什么，然后让损失函数去推动参数。",
  missionTitle:"隐藏评测 ≥ 95%",
  mission:"训练一个二分类器识别“核心区 / 外环”。公开样本会显示，但隐藏样本来自另一批坐标。只记住训练点不算完成。",
  stageTitle:"DECISION FIELD",
  notices:[
    "训练按钮不会移动预制边界；每一步都重新计算 BCE 梯度。",
    "如果特征空间表达不了环形边界，loss 会停在一个无法突破的平台。",
    "隐藏评测与训练集坐标不同，检验的是泛化，不是背答案。"
  ]
};

function makeData(seed,n){
  const rng=seeded(seed),out=[];
  for(let i=0;i<n;i++){
    const a=rng()*Math.PI*2;
    const r=Math.sqrt(rng())*.95;
    const x=r*Math.cos(a),y=r*Math.sin(a);
    const rr=x*x+y*y;
    out.push({x,y,label:rr>0.34?1:0});
  }
  return out;
}
const TRAIN=makeData(12031,140);
const HIDDEN=makeData(93017,220);

function features(mode,p){
  if(mode==="raw")return [1,p.x,p.y];
  if(mode==="cross")return [1,p.x,p.y,p.x*p.y];
  return [1,p.x,p.y,p.x*p.x+p.y*p.y];
}
function names(mode){
  if(mode==="raw")return ["bias","x₁","x₂"];
  if(mode==="cross")return ["bias","x₁","x₂","x₁x₂"];
  return ["bias","x₁","x₂","r²"];
}
function fresh(){
  return {mode:"raw",lr:.18,samples:60,weights:[0,0,0],epoch:0,history:[],hiddenAcc:null,logs:[],solved:false};
}
function normalize(s){
  const z={...fresh(),...s};const d=features(z.mode,TRAIN[0]).length;
  if(!Array.isArray(z.weights)||z.weights.length!==d)z.weights=Array(d).fill(0);
  return z;
}

export default {
  ...META,
  mount(root,ctx){
    let state=normalize(loadState(META.id)||fresh());
    let stopResize=()=>{};
    const ui=demoShell(root,META);
    ui.controls.innerHTML=
      selectControl({id:"bf-mode",label:"MODEL INPUT / 特征表示",value:state.mode,options:[
        {value:"raw",label:"Raw coordinates · [x₁, x₂]"},
        {value:"cross",label:"Interaction · [x₁, x₂, x₁x₂]"},
        {value:"radial",label:"Radial · [x₁, x₂, r²]"}
      ]})+
      rangeControl({id:"bf-samples",label:"TRAINING SAMPLES / 可见样本",min:20,max:120,step:10,value:state.samples})+
      rangeControl({id:"bf-lr",label:"LEARNING RATE / 学习率",min:.02,max:.5,step:.02,value:state.lr})+
      actions([
        {id:"bf-step",label:"1 epoch",icon:"›"},
        {id:"bf-train",label:"训练 200 epochs",icon:"▶",primary:true},
        {id:"bf-exam",label:"隐藏评测",icon:"◆"},
        {id:"bf-reset-model",label:"重置参数",icon:"↺"}
      ]);
    ui.workspace.innerHTML=`
      <div class="viz-card">
        <div class="viz-head"><b>PROBABILITY FIELD</b><span>颜色 = P(class = outer ring)</span></div>
        <div class="canvas-wrap"><canvas id="bf-field"></canvas><div class="canvas-overlay"><span class="legend-chip">● inner = 0</span><span class="legend-chip">● outer = 1</span></div></div>
      </div>
      <div class="stack">
        <div class="subpanel"><div class="subpanel-head"><b>TRAINING CURVE</b><span>BCE loss</span></div><canvas class="chart" id="bf-loss"></canvas></div>
        <div class="subpanel"><div class="subpanel-head"><b>LEARNED PARAMETERS</b><span id="bf-dim"></span></div><div id="bf-weights"></div></div>
        <div class="subpanel"><div class="subpanel-head"><b>RUN LOG</b><span>optimizer events</span></div><div class="log" id="bf-log"></div></div>
      </div>`;
    ui.inspector.innerHTML='<span class="section-kicker">MODEL DIAGNOSTICS</span><div id="bf-metrics"></div><div id="bf-challenge"></div><div id="bf-result"></div>';

    const field=document.getElementById("bf-field"),lossCanvas=document.getElementById("bf-loss");
    function predict(p){const f=features(state.mode,p);let z=0;for(let i=0;i<f.length;i++)z+=state.weights[i]*f[i];return sigmoid(z)}
    function metrics(data=TRAIN.slice(0,state.samples)){
      let loss=0,correct=0;for(const p of data){const pr=predict(p);loss+=bce(pr,p.label);correct+=(pr>=.5)==p.label}
      return {loss:loss/data.length,acc:correct/data.length}
    }
    function resetWeights(){
      const rng=seeded(991+state.mode.length);state.weights=features(state.mode,TRAIN[0]).map(()=>randn(rng)*.04);
      state.epoch=0;state.history=[];state.hiddenAcc=null;state.solved=false;
      logPush(state.logs,"model parameters reinitialized","warn");persist();render();
    }
    function epoch(){
      const data=TRAIN.slice(0,state.samples),grad=Array(state.weights.length).fill(0);
      for(const p of data){
        const f=features(state.mode,p),pr=predict(p),e=pr-p.label;
        for(let j=0;j<grad.length;j++)grad[j]+=e*f[j]/data.length;
      }
      for(let j=0;j<state.weights.length;j++)state.weights[j]-=state.lr*grad[j];
      state.epoch++;
      const m=metrics();if(state.epoch===1||state.epoch%5===0)state.history.push(m.loss);
      if(state.history.length>240)state.history.shift();
      state.hiddenAcc=null;
      return m;
    }
    function train(n){
      let m;for(let i=0;i<n;i++)m=epoch();
      logPush(state.logs,"trained "+n+" epochs · loss "+fmt(m.loss,4)+" · train "+pct(m.acc),m.acc>.95?"ok":"");
      persist();render();
    }
    function exam(){
      const m=metrics(HIDDEN);state.hiddenAcc=m.acc;
      const pass=m.acc>=.95;
      logPush(state.logs,"hidden evaluation "+pct(m.acc)+(pass?" · PASS":" · FAIL"),pass?"ok":"err");
      if(pass&&!state.solved){state.solved=true;ctx.complete()}
      persist();render();
    }
    function persist(){saveState(META.id,state)}
    function drawField(){
      const rect=field.getBoundingClientRect();const dpr=Math.min(2,devicePixelRatio||1);
      field.width=Math.round(rect.width*dpr);field.height=Math.round(rect.height*dpr);
      const g=field.getContext("2d");g.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height,pad=30;
      g.fillStyle="#0b1013";g.fillRect(0,0,w,h);
      const res=34,cw=(w-2*pad)/res,ch=(h-2*pad)/res;
      for(let iy=0;iy<res;iy++)for(let ix=0;ix<res;ix++){
        const x=-1+2*(ix+.5)/res,y=1-2*(iy+.5)/res,pr=predict({x,y});
        const t=pr;const a=[17,30,37],b=[137,82,32];
        g.fillStyle=`rgb(${a[0]+(b[0]-a[0])*t},${a[1]+(b[1]-a[1])*t},${a[2]+(b[2]-a[2])*t})`;
        g.fillRect(pad+ix*cw,pad+iy*ch,cw+.8,ch+.8);
      }
      drawAxes(g,w,h,pad);
      const data=TRAIN.slice(0,state.samples);
      for(const p of data){
        const x=pad+(p.x+1)/2*(w-2*pad),y=pad+(1-p.y)/2*(h-2*pad);
        g.beginPath();g.arc(x,y,3.8,0,Math.PI*2);g.fillStyle=p.label?"#ffbd62":"#65d9e1";g.fill();
        g.strokeStyle="#0c1114";g.lineWidth=1;g.stroke();
      }
    }
    function render(){
      const m=metrics();
      ui.epoch.textContent="EPOCH "+state.epoch;
      ui.status.textContent=state.epoch?"LEARNING":"UNTRAINED";
      document.getElementById("bf-dim").textContent=state.weights.length+" parameters";
      document.getElementById("bf-weights").innerHTML='<table class="feature-table"><thead><tr><th>feature</th><th>weight</th></tr></thead><tbody>'+names(state.mode).map((n,i)=>'<tr><td>'+n+'</td><td>'+fmt(state.weights[i],4)+'</td></tr>').join("")+'</tbody></table>';
      document.getElementById("bf-log").innerHTML=logHtml(state.logs);
      document.getElementById("bf-metrics").innerHTML=
        '<div class="metric-card"><span class="section-kicker">CURRENT LOSS</span><div class="big">'+fmt(m.loss,4)+'</div><small>binary cross entropy · lower is better</small></div>'+
        '<div class="metric-strip">'+
          '<div class="metric '+(m.acc>=.95?"good":"")+'"><span>train acc</span><b>'+pct(m.acc)+'</b></div>'+
          '<div class="metric '+(state.hiddenAcc>=.95?"good":"")+'"><span>hidden acc</span><b>'+pct(state.hiddenAcc)+'</b></div>'+
          '<div class="metric"><span>samples</span><b>'+state.samples+'</b></div></div>';
      const prog=state.hiddenAcc===null?Math.min(.75,m.acc*.75):state.hiddenAcc;
      document.getElementById("bf-challenge").innerHTML=challengeHtml({
        status:state.solved?"PASSED":state.hiddenAcc===null?"NOT TESTED":state.hiddenAcc>=.95?"PASSED":"FAILED",
        body:"目标：隐藏集准确率 ≥ 95%。如果 loss 长期停在 0.6 左右，不一定是学习率问题，也可能是特征空间根本画不出一个圆环。",
        progress:prog,pass:state.solved,fail:state.hiddenAcc!==null&&state.hiddenAcc<.95,
        badges:[
          {text:"REAL SGD",kind:"good"},
          {text:"HELD-OUT DATA",kind:"good"},
          {text:state.mode.toUpperCase(),kind:state.mode==="radial"?"good":""}
        ]
      });
      document.getElementById("bf-result").innerHTML=state.hiddenAcc===null?'<div class="result-banner">隐藏评测尚未运行。训练集表现不等于泛化能力。</div>':
        '<div class="result-banner '+(state.hiddenAcc>=.95?"pass":"fail")+'"><strong>'+pct(state.hiddenAcc)+'</strong> hidden accuracy · '+(state.hiddenAcc>=.95?"模型在未见坐标上也找到了同一结构。":"继续改变表示或优化设置，而不是手工移动边界。")+'</div>';
      drawField();drawLineChart(lossCanvas,[{values:state.history,color:"#ffb347"}],{label:"loss"});
    }

    document.getElementById("bf-mode").addEventListener("change",e=>{
      state.mode=e.target.value;state.weights=Array(features(state.mode,TRAIN[0]).length).fill(0);resetWeights()
    });
    bindRange("bf-samples",v=>{state.samples=v;state.hiddenAcc=null;persist();render()},v=>v);
    bindRange("bf-lr",v=>{state.lr=v;persist()},v=>Number(v).toFixed(2));
    document.getElementById("bf-step").onclick=()=>train(1);
    document.getElementById("bf-train").onclick=()=>train(200);
    document.getElementById("bf-exam").onclick=exam;
    document.getElementById("bf-reset-model").onclick=resetWeights;

    if(!state.weights.some(x=>x!==0))resetWeights();else render();
    stopResize=resizeObserver(field,()=>render());
    window.__NC5_BOUNDARY__={state,getState:()=>deepCopy(state),train,exam,predict,metrics,resetWeights};
    return ()=>{stopResize();delete window.__NC5_BOUNDARY__}
  }
};
