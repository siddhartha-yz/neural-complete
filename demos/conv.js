import {seeded,randn,sigmoid,bce,pct,fmt,saveState,loadState,deepCopy,drawLineChart,drawMatrix,challengeHtml,logPush,logHtml,resizeObserver,clamp} from "../core.js";
import {demoShell,selectControl,rangeControl,actions,bindRange} from "./util.js";

const META={
  id:"conv",code:"DEMO 03",kind:"CONVOLUTIONAL LEARNING",title:"卷积锻造炉",en:"Conv Forge",
  color:"#67dce4",
  card:"训练真正会变化的 3×3 卷积核，让网络从带噪小图中自己学出“横线 / 竖线”局部特征，并观察特征图和池化响应。",
  tags:["CNN","Learned Kernels","Feature Maps"],
  status:"convolutional learning · trainable kernels",
  subtitle:"卷积核不是你手填的边缘检测模板。它们从随机数开始，被分类误差一路反传到每个像素权重。",
  missionTitle:"隐藏图像 ≥ 95%",
  mission:"区分带位置抖动和像素噪声的横线与竖线。隐藏集使用新噪声、新位置；卷积的价值是共享局部参数，而不是记住绝对坐标。",
  stageTitle:"LEARNED FEATURE MAPS",
  notices:[
    "每个 3×3 kernel 都是可训练参数，梯度来自最终分类损失。",
    "池化决定局部证据怎样被压缩为位置不敏感的特征。",
    "训练后应该能直接从 kernel 和 feature map 看到网络偏好的方向结构。"
  ]
};

function makeImage(label,rng){
  const img=Array.from({length:7},()=>Array(7).fill(0));
  const pos=1+Math.floor(rng()*5);
  if(label===1){for(let r=0;r<7;r++){img[r][pos]=1;if(pos+1<7)img[r][pos+1]=.24}}
  else{for(let c=0;c<7;c++){img[pos][c]=1;if(pos+1<7)img[pos+1][c]=.24}}
  for(let r=0;r<7;r++)for(let c=0;c<7;c++)img[r][c]=clamp(img[r][c]+(rng()-.5)*.22,0,1);
  return img;
}
function dataset(seed,n){
  const rng=seeded(seed),out=[];for(let i=0;i<n;i++){const y=i%2;out.push({img:makeImage(y,rng),label:y})}return out
}
const TRAIN=dataset(7109,96),HIDDEN=dataset(28711,180);

function fresh(){return {filters:2,pool:"max",lr:.06,epoch:0,kernels:[],fbias:[],outW:[],outB:0,history:[],hiddenAcc:null,sample:0,logs:[],solved:false}}
function init(s){
  const rng=seeded(440+s.filters*37+(s.pool==="mean"?71:0));
  s.kernels=Array.from({length:s.filters},()=>Array.from({length:3},()=>Array.from({length:3},()=>randn(rng)*.18)));
  s.fbias=Array(s.filters).fill(.06);
  s.outW=Array.from({length:s.filters},()=>randn(rng)*.35);
  s.outB=0;s.epoch=0;s.history=[];s.hiddenAcc=null;s.solved=false;
}
function normalize(s){const z={...fresh(),...s};if(!Array.isArray(z.kernels)||z.kernels.length!==z.filters)init(z);return z}

export default {
  ...META,
  mount(root,ctx){
    let state=normalize(loadState(META.id)||fresh());
    const ui=demoShell(root,META);
    ui.controls.innerHTML=
      rangeControl({id:"conv-filters",label:"FILTER BANK / 卷积核数量",min:1,max:3,step:1,value:state.filters})+
      selectControl({id:"conv-pool",label:"POOLING / 池化",value:state.pool,options:[
        {value:"max",label:"Global max · 最强局部响应"},
        {value:"mean",label:"Global mean · 平均局部响应"}
      ]})+
      rangeControl({id:"conv-lr",label:"LEARNING RATE / 学习率",min:.01,max:.16,step:.01,value:state.lr})+
      rangeControl({id:"conv-sample",label:"INSPECT SAMPLE / 查看样本",min:0,max:15,step:1,value:state.sample})+
      actions([
        {id:"conv-step",label:"1 epoch",icon:"›"},
        {id:"conv-train",label:"训练 300 epochs",icon:"▶",primary:true},
        {id:"conv-exam",label:"隐藏评测",icon:"◆"},
        {id:"conv-reset",label:"重置卷积核",icon:"↺"}
      ]);
    ui.workspace.innerHTML=`
      <div class="viz-card">
        <div class="viz-head"><b>INPUT → CONV → POOL</b><span id="conv-sample-meta"></span></div>
        <div class="image-pair">
          <div class="pixel-board"><b>INPUT 7×7</b><canvas id="conv-input"></canvas></div>
          <div class="pixel-board"><b>CLASS PROBABILITY</b><canvas id="conv-prob"></canvas></div>
        </div>
        <div class="divider"></div>
        <div class="feature-map-grid" id="conv-maps"></div>
      </div>
      <div class="stack">
        <div class="subpanel"><div class="subpanel-head"><b>LEARNED KERNELS</b><span>3×3 trainable weights</span></div><div class="feature-map-grid" id="conv-kernels"></div></div>
        <div class="subpanel"><div class="subpanel-head"><b>TRAINING CURVE</b><span>BCE</span></div><canvas class="chart" id="conv-loss"></canvas></div>
        <div class="subpanel"><div class="subpanel-head"><b>RUN LOG</b><span>gradient events</span></div><div class="log" id="conv-log"></div></div>
      </div>`;
    ui.inspector.innerHTML='<span class="section-kicker">CNN DIAGNOSTICS</span><div id="conv-metrics"></div><div id="conv-challenge"></div><div id="conv-result"></div><div class="inline-note">类别定义：horizontal = 0，vertical = 1。卷积层没有接收“方向”这个人工标签；它只收到最终分类误差。</div>';

    const loss=document.getElementById("conv-loss");let cleanup=()=>{};
    function convOne(img,kernel,bias){
      const z=Array.from({length:5},()=>Array(5).fill(0)),a=Array.from({length:5},()=>Array(5).fill(0));
      for(let r=0;r<5;r++)for(let c=0;c<5;c++){
        let v=bias;for(let kr=0;kr<3;kr++)for(let kc=0;kc<3;kc++)v+=img[r+kr][c+kc]*kernel[kr][kc];
        z[r][c]=v;a[r][c]=Math.max(0,v);
      }
      return {z,a};
    }
    function forward(ex){
      const maps=[],pooled=[],arg=[];
      for(let f=0;f<state.filters;f++){
        const m=convOne(ex.img,state.kernels[f],state.fbias[f]);maps.push(m);
        if(state.pool==="max"){
          let best=-Infinity,bi=[0,0];for(let r=0;r<5;r++)for(let c=0;c<5;c++)if(m.a[r][c]>best){best=m.a[r][c];bi=[r,c]}
          pooled.push(best);arg.push(bi);
        }else{
          let s=0;for(const row of m.a)for(const v of row)s+=v;pooled.push(s/25);arg.push(null);
        }
      }
      let z=state.outB;for(let f=0;f<state.filters;f++)z+=state.outW[f]*pooled[f];
      return {maps,pooled,arg,z,p:sigmoid(z)};
    }
    function metrics(ds=TRAIN){
      let l=0,c=0;for(const ex of ds){const o=forward(ex);l+=bce(o.p,ex.label);c+=(o.p>=.5)==ex.label}return {loss:l/ds.length,acc:c/ds.length}
    }
    function oneEpoch(){
      const gK=Array.from({length:state.filters},()=>Array.from({length:3},()=>Array(3).fill(0)));
      const gFb=Array(state.filters).fill(0),gOut=Array(state.filters).fill(0);let gOb=0;
      for(const ex of TRAIN){
        const o=forward(ex),dz=o.p-ex.label;gOb+=dz/TRAIN.length;
        for(let f=0;f<state.filters;f++){
          gOut[f]+=dz*o.pooled[f]/TRAIN.length;
          const dpool=dz*state.outW[f];
          const dA=Array.from({length:5},()=>Array(5).fill(0));
          if(state.pool==="max"){const [r,c]=o.arg[f];dA[r][c]=dpool}
          else for(let r=0;r<5;r++)for(let c=0;c<5;c++)dA[r][c]=dpool/25;
          for(let r=0;r<5;r++)for(let c=0;c<5;c++){
            const dzf=o.maps[f].z[r][c]>0?dA[r][c]:0;
            gFb[f]+=dzf/TRAIN.length;
            for(let kr=0;kr<3;kr++)for(let kc=0;kc<3;kc++)gK[f][kr][kc]+=dzf*ex.img[r+kr][c+kc]/TRAIN.length;
          }
        }
      }
      for(let f=0;f<state.filters;f++){
        state.outW[f]-=state.lr*gOut[f];state.fbias[f]-=state.lr*gFb[f];
        for(let r=0;r<3;r++)for(let c=0;c<3;c++)state.kernels[f][r][c]-=state.lr*gK[f][r][c];
      }
      state.outB-=state.lr*gOb;state.epoch++;const m=metrics();
      if(state.epoch===1||state.epoch%3===0)state.history.push(m.loss);if(state.history.length>250)state.history.shift();state.hiddenAcc=null;return m;
    }
    function train(n){
      let m;for(let i=0;i<n;i++)m=oneEpoch();
      logPush(state.logs,"CNN trained "+n+" epochs · loss "+fmt(m.loss,4)+" · acc "+pct(m.acc),m.acc>.94?"ok":"");saveState(META.id,state);render()
    }
    function exam(){
      const m=metrics(HIDDEN);state.hiddenAcc=m.acc;const pass=m.acc>=.95;logPush(state.logs,"hidden image set "+pct(m.acc)+(pass?" · PASS":" · FAIL"),pass?"ok":"err");
      if(pass&&!state.solved){state.solved=true;ctx.complete()}saveState(META.id,state);render()
    }
    function reset(){init(state);logPush(state.logs,"kernel bank randomized","warn");saveState(META.id,state);render()}
    function drawInput(){
      drawMatrix(document.getElementById("conv-input"),TRAIN[state.sample].img,{signed:false});
      const o=forward(TRAIN[state.sample]),canvas=document.getElementById("conv-prob"),r=canvas.getBoundingClientRect(),dpr=Math.min(2,devicePixelRatio||1);canvas.width=r.width*dpr;canvas.height=r.height*dpr;
      const g=canvas.getContext("2d");g.setTransform(dpr,0,0,dpr,0,0);g.fillStyle="#0b1013";g.fillRect(0,0,r.width,r.height);
      const p=o.p;g.fillStyle="#1d2a30";g.fillRect(16,r.height-34,r.width-32,14);g.fillStyle=p>=.5?"#67dce4":"#ffb347";g.fillRect(16,r.height-34,(r.width-32)*p,14);
      g.fillStyle="#dfe6e2";g.font="700 27px ui-monospace,monospace";g.textAlign="center";g.fillText(fmt(p,3),r.width/2,r.height/2-3);
      g.fillStyle="#708089";g.font="8px ui-monospace,monospace";g.fillText("P(vertical)",r.width/2,r.height/2+16);
    }
    function renderMaps(){
      const ex=TRAIN[state.sample],o=forward(ex);document.getElementById("conv-sample-meta").textContent="label "+ex.label+" · predicted "+(o.p>=.5?1:0);
      document.getElementById("conv-maps").innerHTML=o.maps.map((m,f)=>'<div class="tiny-map"><b>FEATURE '+(f+1)+' · pooled '+fmt(o.pooled[f],3)+'</b><canvas id="conv-map-'+f+'"></canvas></div>').join("");
      document.getElementById("conv-kernels").innerHTML=state.kernels.map((k,f)=>'<div class="tiny-map"><b>KERNEL '+(f+1)+' · out '+fmt(state.outW[f],3)+'</b><canvas id="conv-kernel-'+f+'"></canvas></div>').join("");
      requestAnimationFrame(()=>{
        o.maps.forEach((m,f)=>drawMatrix(document.getElementById("conv-map-"+f),m.a,{signed:false}));
        state.kernels.forEach((k,f)=>drawMatrix(document.getElementById("conv-kernel-"+f),k,{signed:true,labels:true}));
      });
    }
    function render(){
      const m=metrics();ui.epoch.textContent="EPOCH "+state.epoch;ui.status.textContent=state.epoch?"KERNELS LEARNING":"RANDOM FILTERS";
      document.getElementById("conv-log").innerHTML=logHtml(state.logs);
      document.getElementById("conv-metrics").innerHTML=
        '<div class="metric-card"><span class="section-kicker">CLASSIFICATION LOSS</span><div class="big">'+fmt(m.loss,4)+'</div><small>gradient reaches every kernel weight</small></div>'+
        '<div class="metric-strip"><div class="metric '+(m.acc>=.95?"good":"")+'"><span>train</span><b>'+pct(m.acc)+'</b></div><div class="metric '+(state.hiddenAcc>=.95?"good":"")+'"><span>hidden</span><b>'+pct(state.hiddenAcc)+'</b></div><div class="metric"><span>filters</span><b>'+state.filters+'</b></div></div>';
      document.getElementById("conv-challenge").innerHTML=challengeHtml({
        status:state.solved?"PASSED":state.hiddenAcc===null?"NOT TESTED":state.hiddenAcc>=.95?"PASSED":"FAILED",
        body:"目标：隐藏图像 ≥ 95%。观察训练前后 kernel 是否从随机噪声变成有方向偏好的局部模板，并检查池化是否帮助抵抗位置变化。",
        progress:state.hiddenAcc??m.acc*.75,pass:state.solved,fail:state.hiddenAcc!==null&&state.hiddenAcc<.95,
        badges:[{text:"TRAINABLE KERNELS",kind:"good"},{text:state.pool.toUpperCase()+" POOL"},{text:state.filters+" FILTERS"}]
      });
      document.getElementById("conv-result").innerHTML=state.hiddenAcc===null?'<div class="result-banner">最终评测使用新的线条位置与像素噪声。</div>':
        '<div class="result-banner '+(state.hiddenAcc>=.95?"pass":"fail")+'"><strong>'+pct(state.hiddenAcc)+'</strong> hidden accuracy · '+(state.hiddenAcc>=.95?"卷积核提取了可迁移的局部方向特征。":"继续训练或改变 filter bank / pooling。")+'</div>';
      drawInput();renderMaps();drawLineChart(loss,[{values:state.history,color:"#67dce4"}],{label:"BCE"});
    }

    bindRange("conv-filters",v=>{state.filters=v;init(state);saveState(META.id,state);render()},v=>v);
    document.getElementById("conv-pool").addEventListener("change",e=>{state.pool=e.target.value;init(state);saveState(META.id,state);render()});
    bindRange("conv-lr",v=>{state.lr=v;saveState(META.id,state)},v=>Number(v).toFixed(2));
    bindRange("conv-sample",v=>{state.sample=v;saveState(META.id,state);render()},v=>"#"+v);
    document.getElementById("conv-step").onclick=()=>train(1);
    document.getElementById("conv-train").onclick=()=>train(300);
    document.getElementById("conv-exam").onclick=exam;
    document.getElementById("conv-reset").onclick=reset;
    render();cleanup=resizeObserver(document.getElementById("conv-input"),()=>{drawInput();renderMaps()});
    window.__NC5_CONV__={state,getState:()=>deepCopy(state),train,exam,metrics,forward,reset};
    return ()=>{cleanup();delete window.__NC5_CONV__}
  }
};
