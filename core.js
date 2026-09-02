export function clamp(x,a,b){return Math.max(a,Math.min(b,x))}
export function sigmoid(x){return 1/(1+Math.exp(-clamp(x,-40,40)))}
export function tanh(x){return Math.tanh(x)}
export function relu(x){return x>0?x:0}
export function mean(xs){return xs.length?xs.reduce((a,b)=>a+b,0)/xs.length:0}
export function argmax(xs){let k=0;for(let i=1;i<xs.length;i++)if(xs[i]>xs[k])k=i;return k}
export function fmt(x,d=3){
  if(x===null||x===undefined||Number.isNaN(Number(x)))return "—";
  const n=Number(x);if(!Number.isFinite(n))return "—";
  if(Math.abs(n)>=1000)return n.toExponential(1);
  if(Math.abs(n)<1e-4&&n!==0)return n.toExponential(1);
  return n.toFixed(d).replace(/\.0+$|(?<=\.[0-9]*?)0+$/,"").replace(/\.$/,"");
}
export function pct(x){return Number.isFinite(x)?Math.round(x*100)+"%":"—"}
export function seeded(seed=1){
  let s=seed>>>0;
  return ()=>{
    s+=0x6D2B79F5;
    let t=s;
    t=Math.imul(t^t>>>15,t|1);
    t^=t+Math.imul(t^t>>>7,t|61);
    return ((t^t>>>14)>>>0)/4294967296;
  };
}
export function randn(rng=Math.random){
  let u=0,v=0;while(!u)u=rng();while(!v)v=rng();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
}
export function shuffle(arr,rng=Math.random){
  const a=[...arr];for(let i=a.length-1;i>0;i--){const j=Math.floor(rng()*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a
}
export function zeros(r,c=null){return c===null?Array(r).fill(0):Array.from({length:r},()=>Array(c).fill(0))}
export function deepCopy(x){return JSON.parse(JSON.stringify(x))}
export function dot(a,b){let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s}
export function mse(a,b){let s=0;for(let i=0;i<a.length;i++){const d=a[i]-b[i];s+=d*d}return s/a.length}
export function bce(p,y){const q=clamp(p,1e-7,1-1e-7);return -(y*Math.log(q)+(1-y)*Math.log(1-q))}
export function el(tag,cls="",html=""){
  const n=document.createElement(tag);if(cls)n.className=cls;if(html!==undefined)n.innerHTML=html;return n
}
export function canvas2d(canvas){
  const r=canvas.getBoundingClientRect(),dpr=Math.min(2,window.devicePixelRatio||1);
  const w=Math.max(10,Math.round(r.width*dpr)),h=Math.max(10,Math.round(r.height*dpr));
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}
  const ctx=canvas.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx,w:r.width,h:r.height,dpr}
}
export function clearCanvas(canvas,bg="#0b1013"){
  const {ctx,w,h}=canvas2d(canvas);ctx.clearRect(0,0,w,h);ctx.fillStyle=bg;ctx.fillRect(0,0,w,h);return {ctx,w,h}
}
export function drawAxes(ctx,w,h,pad=28){
  ctx.strokeStyle="#2b343c";ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad,h-pad);ctx.lineTo(w-pad,h-pad);ctx.moveTo(pad,pad);ctx.lineTo(pad,h-pad);ctx.stroke();
  ctx.fillStyle="#68747c";ctx.font="8px ui-monospace, monospace";ctx.textAlign="center";
  for(let i=0;i<=4;i++){const x=pad+(w-2*pad)*i/4;ctx.fillText((i/2-1).toFixed(1),x,h-10)}
  ctx.textAlign="right";
  for(let i=0;i<=4;i++){const y=h-pad-(h-2*pad)*i/4;ctx.fillText((i/2-1).toFixed(1),pad-6,y+3)}
}
export function drawLineChart(canvas,series,{minY=null,maxY=null,label=""}={}){
  const {ctx,w,h}=clearCanvas(canvas);const pad=22;
  const all=series.flatMap(s=>s.values).filter(Number.isFinite);
  if(!all.length){ctx.fillStyle="#64717a";ctx.font="9px sans-serif";ctx.fillText("waiting for data",10,18);return}
  let lo=minY??Math.min(...all),hi=maxY??Math.max(...all);if(Math.abs(hi-lo)<1e-9){hi=lo+1}
  ctx.strokeStyle="#283139";ctx.lineWidth=1;
  for(let i=0;i<4;i++){const y=pad+(h-2*pad)*i/3;ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(w-8,y);ctx.stroke()}
  for(const s of series){
    const vals=s.values;if(vals.length<1)continue;ctx.strokeStyle=s.color||"#ffb347";ctx.lineWidth=1.5;ctx.beginPath();
    vals.forEach((v,i)=>{const x=pad+(w-pad-8)*(vals.length===1?0:i/(vals.length-1));const y=h-pad-(h-2*pad)*(v-lo)/(hi-lo);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)});ctx.stroke()
  }
  ctx.fillStyle="#65727b";ctx.font="7px ui-monospace, monospace";ctx.fillText(fmt(hi,2),3,pad);ctx.fillText(fmt(lo,2),3,h-pad);
  if(label){ctx.fillStyle="#89949b";ctx.fillText(label,w-50,12)}
}
export function heatColor(v,mode="signed"){
  if(mode==="prob"){const t=clamp(v,0,1);const a=[30,46,53],b=[255,179,71];return `rgb(${a[0]+(b[0]-a[0])*t},${a[1]+(b[1]-a[1])*t},${a[2]+(b[2]-a[2])*t})`}
  const t=clamp(Math.abs(v),0,1);
  const b=v>=0?[103,220,228]:[239,118,116],a=[20,27,31];
  return `rgb(${a[0]+(b[0]-a[0])*t},${a[1]+(b[1]-a[1])*t},${a[2]+(b[2]-a[2])*t})`
}
export function drawMatrix(canvas,mat,{signed=true,labels=false}={}){
  const {ctx,w,h}=clearCanvas(canvas);const rows=mat.length,cols=mat[0]?.length||0;if(!rows||!cols)return;
  const cw=w/cols,ch=h/rows;let max=1;
  if(signed)max=Math.max(1e-8,...mat.flat().map(x=>Math.abs(x)));
  for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){
    const v=mat[r][c];ctx.fillStyle=signed?heatColor(v/max,"signed"):heatColor(v,"prob");ctx.fillRect(c*cw,r*ch,cw+.5,ch+.5);
    if(labels&&cw>28){ctx.fillStyle="#e6ebe6";ctx.font="7px ui-monospace";ctx.textAlign="center";ctx.fillText(fmt(v,2),c*cw+cw/2,r*ch+ch/2+2)}
  }
}
export function toast(msg,type=""){
  const t=document.getElementById("toast");if(!t)return;t.textContent=msg;t.className="toast show "+type;clearTimeout(toast.timer);toast.timer=setTimeout(()=>t.className="toast",1800)
}
export function saveState(key,obj){try{localStorage.setItem("nc5:"+key,JSON.stringify(obj))}catch{}}
export function loadState(key){try{const v=localStorage.getItem("nc5:"+key);return v?JSON.parse(v):null}catch{return null}}
export function clearState(key){try{localStorage.removeItem("nc5:"+key)}catch{}}
export function sleep(ms){return new Promise(r=>setTimeout(r,ms))}
export function metric(label,value,cls=""){return `<div class="metric ${cls}"><span>${label}</span><b>${value}</b></div>`}
export function challengeHtml({title="MISSION CHECK",status="READY",body="",progress=0,pass=false,fail=false,badges=[]}){
  const cls=pass?"pass":fail?"fail":"";
  return `<div class="challenge-card ${cls}">
    <div class="challenge-head"><span>${title}</span><b>${status}</b></div>
    <p>${body}</p><div class="progress"><i style="width:${clamp(progress,0,1)*100}%"></i></div>
    <div class="badge-row">${badges.map(b=>`<span class="badge ${b.kind||""}">${b.text}</span>`).join("")}</div>
  </div>`
}
export function logPush(arr,text,kind="",cap=60){
  arr.push({text,kind,time:new Date().toLocaleTimeString([], {hour:"2-digit",minute:"2-digit",second:"2-digit"})});if(arr.length>cap)arr.splice(0,arr.length-cap)
}
export function logHtml(arr){return arr.slice().reverse().map(x=>`<div class="${x.kind||""}"><span>${x.time}</span> · ${x.text}</div>`).join("")}
export function resizeObserver(elm,fn){
  const ro=new ResizeObserver(()=>requestAnimationFrame(fn));ro.observe(elm);return ()=>ro.disconnect()
}
export function downloadJSON(name,data){
  const blob=new Blob([JSON.stringify(data,null,2)],{type:"application/json"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)
}
