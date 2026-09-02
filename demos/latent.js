import {seeded,randn,mean,mse,fmt,pct,saveState,loadState,deepCopy,drawLineChart,drawMatrix,challengeHtml,logPush,logHtml,resizeObserver,clamp} from "../core.js";
import {demoShell,rangeControl,actions,bindRange} from "./util.js";

const META={
  id:"latent",code:"DEMO 04",kind:"UNSUPERVISED LEARNING",title:"潜空间金库",en:"Latent Vault",
  color:"#75d6a1",
  card:"不给任何类别标签，只要求网络把图像压进极小瓶颈再重建。观察编码器如何自动把两个连续生成因素折叠进潜变量。",
  tags:["Autoencoder","Bottleneck","Latent Space"],
  status:"unsupervised learning · reconstruction",
  subtitle:"没有 class label。监督信号就是输入本身：压缩后还能不能把它恢复出来？",
  missionTitle:"2 维瓶颈完成压缩",
  mission:"用不超过 2 个 latent dimensions 重构一批由两个独立连续因素生成的 6×6 信号图。隐藏集使用未见组合。",
  stageTitle:"ENCODE → BOTTLENECK → DECODE",
  notices:[
    "encoder / decoder 都从随机权重开始，目标只是真实的 reconstruction MSE。",
    "瓶颈太窄会丢信息；太宽虽然容易重构，却没有完成压缩任务。",
    "潜空间散点不是人工坐标：点的位置来自训练出的 encoder。"
  ]
};

function makeBases(){
  const a=Array.from({length:6},()=>Array(6).fill(0)),b=Array.from({length:6},()=>Array(6).fill(0));
  for(let r=0;r<6;r++)for(let c=0;c<6;c++){
    a[r][c]=(c%2===0?.34:.04);
    b[r][c]=(r%2===0?.34:.04);
  }
  return [a.flat(),b.flat()];
}
const [B1,B2]=makeBases();
function dataset(seed,n){
  const rng=seeded(seed),out=[];
  for(let i=0;i<n;i++){
    const a=.08+.88*rng(),b=.08+.88*rng(),x=[];
    for(let j=0;j<36;j++)x.push(clamp(.04+a*B1[j]+b*B2[j]+randn(rng)*.008,0,1));
    out.push({x,a,b});
  }
  return out;
}
const TRAIN=dataset(15191,72),HIDDEN=dataset(77731,120);

function fresh(){return {latent:1,lr:.08,epoch:0,We:[],be:[],Wd:[],bd:[],history:[],hiddenMse:null,sampleA:7,sampleB:51,interp:.5,logs:[],solved:false}}
function init(s){
  const rng=seeded(881+s.latent*41);
  s.We=Array.from({length:s.latent},()=>Array.from({length:36},()=>randn(rng)*.055));
  s.be=Array(s.latent).fill(0);
  s.Wd=Array.from({length:36},()=>Array.from({length:s.latent},()=>randn(rng)*.055));
  s.bd=Array(36).fill(.12);
  s.epoch=0;s.history=[];s.hiddenMse=null;s.solved=false;
}
function normalize(s){const z={...fresh(),...s};if(!Array.isArray(z.We)||z.We.length!==z.latent||!Array.isArray(z.Wd)||z.Wd.length!==36)init(z);return z}

export default {
  ...META,
  mount(root,ctx){
    let state=normalize(loadState(META.id)||fresh());
    const ui=demoShell(root,META);
    ui.controls.innerHTML=
      rangeControl({id:"ae-latent",label:"BOTTLENECK / 潜变量维度",min:1,max:4,step:1,value:state.latent})+
      rangeControl({id:"ae-lr",label:"LEARNING RATE / 学习率",min:.01,max:.12,step:.005,value:state.lr})+
      rangeControl({id:"ae-a",label:"SOURCE A / 样本 A",min:0,max:35,step:1,value:state.sampleA})+
      rangeControl({id:"ae-b",label:"SOURCE B / 样本 B",min:36,max:71,step:1,value:state.sampleB})+
      rangeControl({id:"ae-t",label:"LATENT INTERPOLATION / 插值",min:0,max:1,step:.05,value:state.interp})+
      actions([
        {id:"ae-step",label:"1 epoch",icon:"›"},
        {id:"ae-train",label:"训练 1000 epochs",icon:"▶",primary:true},
        {id:"ae-exam",label:"隐藏重构评测",icon:"◆"},
        {id:"ae-reset",label:"重置网络",icon:"↺"}
      ]);
    ui.workspace.innerHTML=`
      <div class="viz-card">
        <div class="viz-head"><b>RECONSTRUCTION</b><span id="ae-sample-meta"></span></div>
        <div class="image-pair">
          <div class="pixel-board"><b>ORIGINAL A</b><canvas id="ae-original"></canvas></div>
          <div class="pixel-board"><b>RECONSTRUCTED A</b><canvas id="ae-recon"></canvas></div>
        </div>
        <div class="divider"></div>
        <div class="image-pair">
          <div class="pixel-board"><b>ORIGINAL B</b><canvas id="ae-original-b"></canvas></div>
          <div class="pixel-board"><b>DECODED LATENT INTERPOLATION</b><canvas id="ae-interp"></canvas></div>
        </div>
      </div>
      <div class="stack">
        <div class="subpanel"><div class="subpanel-head"><b>LATENT SPACE</b><span>encoder outputs · first two dimensions</span></div><canvas id="ae-latent-map" class="chart" style="height:180px"></canvas></div>
        <div class="subpanel"><div class="subpanel-head"><b>TRAINING CURVE</b><span>reconstruction MSE</span></div><canvas class="chart" id="ae-loss"></canvas></div>
        <div class="subpanel"><div class="subpanel-head"><b>RUN LOG</b><span>unsupervised optimizer</span></div><div class="log" id="ae-log"></div></div>
      </div>`;
    ui.inspector.innerHTML='<span class="section-kicker">AUTOENCODER DIAGNOSTICS</span><div id="ae-metrics"></div><div id="ae-challenge"></div><div id="ae-result"></div><div class="inline-note">训练数据没有 0/1 类别。图像由两个连续因素混合产生；模型必须自己找到足够描述它们的低维坐标。</div>';

    const lossCanvas=document.getElementById("ae-loss"),latentCanvas=document.getElementById("ae-latent-map");let cleanup=()=>{};
    function encode(x){
      const z=Array(state.latent).fill(0);
      for(let k=0;k<state.latent;k++){let v=state.be[k];for(let j=0;j<36;j++)v+=state.We[k][j]*x[j];z[k]=v}
      return z;
    }
    function decode(z){
      const y=Array(36).fill(0);for(let j=0;j<36;j++){let v=state.bd[j];for(let k=0;k<state.latent;k++)v+=state.Wd[j][k]*z[k];y[j]=v}return y;
    }
    function forward(x){const z=encode(x);return {z,y:decode(z)}}
    function metrics(ds=TRAIN){return mean(ds.map(ex=>mse(forward(ex.x).y,ex.x)))}
    function oneEpoch(){
      const gWe=Array.from({length:state.latent},()=>Array(36).fill(0)),gbe=Array(state.latent).fill(0);
      const gWd=Array.from({length:36},()=>Array(state.latent).fill(0)),gbd=Array(36).fill(0);
      const n=TRAIN.length;
      for(const ex of TRAIN){
        const o=forward(ex.x),dy=Array(36);
        for(let j=0;j<36;j++)dy[j]=2*(o.y[j]-ex.x[j])/36;
        const dz=Array(state.latent).fill(0);
        for(let j=0;j<36;j++){
          gbd[j]+=dy[j]/n;
          for(let k=0;k<state.latent;k++){gWd[j][k]+=dy[j]*o.z[k]/n;dz[k]+=dy[j]*state.Wd[j][k]}
        }
        for(let k=0;k<state.latent;k++){
          gbe[k]+=dz[k]/n;for(let j=0;j<36;j++)gWe[k][j]+=dz[k]*ex.x[j]/n;
        }
      }
      const lr=state.lr;
      for(let k=0;k<state.latent;k++){state.be[k]-=lr*gbe[k];for(let j=0;j<36;j++)state.We[k][j]-=lr*gWe[k][j]}
      for(let j=0;j<36;j++){state.bd[j]-=lr*gbd[j];for(let k=0;k<state.latent;k++)state.Wd[j][k]-=lr*gWd[j][k]}
      state.epoch++;const m=metrics();if(state.epoch===1||state.epoch%3===0)state.history.push(m);if(state.history.length>260)state.history.shift();state.hiddenMse=null;return m;
    }
    function train(n){let m;for(let i=0;i<n;i++)m=oneEpoch();logPush(state.logs,"autoencoder "+n+" epochs · MSE "+fmt(m,5),m<.004?"ok":"");saveState(META.id,state);render()}
    function exam(){
      const m=metrics(HIDDEN);state.hiddenMse=m;const pass=m<=.0025&&state.latent===2;
      logPush(state.logs,"hidden reconstruction MSE "+fmt(m,5)+(pass?" · PASS":" · FAIL"),pass?"ok":"err");
      if(pass&&!state.solved){state.solved=true;ctx.complete()}saveState(META.id,state);render()
    }
    function reset(){init(state);logPush(state.logs,"encoder/decoder randomized","warn");saveState(META.id,state);render()}
    function mat(v){const out=[];for(let r=0;r<6;r++)out.push(v.slice(r*6,r*6+6).map(x=>clamp(x,0,1)));return out}
    function drawImages(){
      const A=TRAIN[state.sampleA],B=TRAIN[state.sampleB],oa=forward(A.x),za=oa.z,zb=encode(B.x),t=state.interp,z=za.map((v,k)=>v*(1-t)+zb[k]*t),yi=decode(z);
      drawMatrix(document.getElementById("ae-original"),mat(A.x),{signed:false});drawMatrix(document.getElementById("ae-recon"),mat(oa.y),{signed:false});
      drawMatrix(document.getElementById("ae-original-b"),mat(B.x),{signed:false});drawMatrix(document.getElementById("ae-interp"),mat(yi),{signed:false});
      document.getElementById("ae-sample-meta").textContent="A mse "+fmt(mse(oa.y,A.x),5)+" · latent ["+za.map(v=>fmt(v,2)).join(", ")+"]";
    }
    function drawLatent(){
      const r=latentCanvas.getBoundingClientRect(),dpr=Math.min(2,devicePixelRatio||1);latentCanvas.width=r.width*dpr;latentCanvas.height=r.height*dpr;
      const g=latentCanvas.getContext("2d");g.setTransform(dpr,0,0,dpr,0,0);g.fillStyle="#0b1013";g.fillRect(0,0,r.width,r.height);
      const zs=TRAIN.map(ex=>({z:encode(ex.x),a:ex.a,b:ex.b})),xv=zs.map(o=>o.z[0]),yv=state.latent>1?zs.map(o=>o.z[1]):zs.map(o=>o.a);
      let xmin=Math.min(...xv),xmax=Math.max(...xv),ymin=Math.min(...yv),ymax=Math.max(...yv);if(xmax-xmin<1e-6)xmax=xmin+1;if(ymax-ymin<1e-6)ymax=ymin+1;
      const pad=16;for(const o of zs){
        const x=pad+(o.z[0]-xmin)/(xmax-xmin)*(r.width-2*pad),yv0=state.latent>1?o.z[1]:o.a,y=r.height-pad-(yv0-ymin)/(ymax-ymin)*(r.height-2*pad);
        const cr=Math.round(80+150*o.a),cg=Math.round(95+120*o.b);g.fillStyle=`rgb(${cr},${cg},145)`;g.beginPath();g.arc(x,y,3,0,Math.PI*2);g.fill()
      }
      g.fillStyle="#6e7a82";g.font="7px ui-monospace";g.fillText(state.latent>1?"z₁ vs z₂":"z₁ vs true factor (diagnostic)",8,10);
    }
    function render(){
      const m=metrics();ui.epoch.textContent="EPOCH "+state.epoch;ui.status.textContent=state.epoch?"RECONSTRUCTION LEARNING":"UNTRAINED";
      document.getElementById("ae-log").innerHTML=logHtml(state.logs);
      const ratio=state.latent/36;
      document.getElementById("ae-metrics").innerHTML=
        '<div class="metric-card"><span class="section-kicker">TRAIN RECONSTRUCTION MSE</span><div class="big">'+fmt(m,5)+'</div><small>no labels · target equals input</small></div>'+
        '<div class="metric-strip"><div class="metric '+(m<=.0025?"good":"")+'"><span>train mse</span><b>'+fmt(m,4)+'</b></div><div class="metric '+(state.hiddenMse!==null&&state.hiddenMse<=.0025?"good":"")+'"><span>hidden mse</span><b>'+fmt(state.hiddenMse,4)+'</b></div><div class="metric '+(state.latent===2?"good":"bad")+'"><span>code/input</span><b>'+pct(ratio)+'</b></div></div>';
      document.getElementById("ae-challenge").innerHTML=challengeHtml({
        status:state.solved?"PASSED":state.hiddenMse===null?"NOT TESTED":state.hiddenMse<=.0025&&state.latent===2?"PASSED":"FAILED",
        body:"目标：恰好使用 2D latent，并把隐藏重构 MSE 压到 ≤ 0.0025。1D 会把两个独立因素挤在同一轴上；3D/4D 虽更宽，却没有完成目标压缩率。",
        progress:state.hiddenMse===null?Math.min(.78,.0025/Math.max(m,.0025)*.78):Math.min(1,.0025/Math.max(state.hiddenMse,.00001)),pass:state.solved,fail:state.hiddenMse!==null&&!state.solved,
        badges:[{text:"NO LABELS",kind:"good"},{text:state.latent+"D LATENT",kind:state.latent===2?"good":"bad"},{text:"MSE OBJECTIVE"}]
      });
      document.getElementById("ae-result").innerHTML=state.hiddenMse===null?'<div class="result-banner">隐藏组合尚未评测。训练误差低不保证瓶颈真的抓住生成因素。</div>':
        '<div class="result-banner '+(state.solved?"pass":"fail")+'"><strong>'+fmt(state.hiddenMse,5)+'</strong> hidden MSE · '+(state.solved?"两个潜变量足以保留这批数据的主要自由度。":"检查瓶颈宽度与训练收敛；同时满足“低误差”和“低维”。")+'</div>';
      drawImages();drawLatent();drawLineChart(lossCanvas,[{values:state.history,color:"#75d6a1"}],{label:"MSE"});
    }

    bindRange("ae-latent",v=>{state.latent=v;init(state);saveState(META.id,state);render()},v=>v+"D");
    bindRange("ae-lr",v=>{state.lr=v;saveState(META.id,state)},v=>Number(v).toFixed(3));
    bindRange("ae-a",v=>{state.sampleA=v;saveState(META.id,state);render()},v=>"#"+v);
    bindRange("ae-b",v=>{state.sampleB=v;saveState(META.id,state);render()},v=>"#"+v);
    bindRange("ae-t",v=>{state.interp=v;saveState(META.id,state);render()},v=>Number(v).toFixed(2));
    document.getElementById("ae-step").onclick=()=>train(1);document.getElementById("ae-train").onclick=()=>train(1000);document.getElementById("ae-exam").onclick=exam;document.getElementById("ae-reset").onclick=reset;
    render();cleanup=resizeObserver(document.getElementById("ae-original"),()=>{drawImages();drawLatent()});
    window.__NC5_LATENT__={state,getState:()=>deepCopy(state),train,exam,metrics,forward,reset};
    return ()=>{cleanup();delete window.__NC5_LATENT__}
  }
};
