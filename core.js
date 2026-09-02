export const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
export const sigmoid=x=>1/(1+Math.exp(-clamp(x,-40,40)));
export const tanh=Math.tanh;
export const relu=x=>x>0?x:0;
export const mean=xs=>xs.length?xs.reduce((a,b)=>a+b,0)/xs.length:0;
export const sum=xs=>xs.reduce((a,b)=>a+b,0);
export const argmax=xs=>{let k=0;for(let i=1;i<xs.length;i++)if(xs[i]>xs[k])k=i;return k};
export const bce=(p,y)=>{const q=clamp(p,1e-7,1-1e-7);return -(y*Math.log(q)+(1-y)*Math.log(1-q))};
export const mse=(a,b)=>mean(a.map((v,i)=>(v-b[i])**2));
export const pct=x=>Number.isFinite(x)?(x*100).toFixed(x>=.995?0:1)+"%":"—";
export const fmt=(x,d=3)=>{
  if(x===null||x===undefined||!Number.isFinite(Number(x)))return "—";
  const n=Number(x);
  if(Math.abs(n)>=1000)return n.toExponential(1);
  if(Math.abs(n)>0&&Math.abs(n)<10**(-d))return n.toExponential(1);
  return n.toFixed(d).replace(/\.0+$|(?<=\.[0-9]*?)0+$/,"").replace(/\.$/,"");
};
export function seeded(seed=1){
  let s=seed>>>0;
  return ()=>{s+=0x6D2B79F5;let t=s;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296};
}
export function randn(rng=Math.random){let u=0,v=0;while(!u)u=rng();while(!v)v=rng();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)}
export function shuffle(arr,rng=Math.random){const a=[...arr];for(let i=a.length-1;i>0;i--){const j=Math.floor(rng()*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a}
export const deepCopy=x=>JSON.parse(JSON.stringify(x));
export function save(key,value){try{localStorage.setItem("nc90:"+key,JSON.stringify(value))}catch{}}
export function load(key){try{const v=localStorage.getItem("nc90:"+key);return v?JSON.parse(v):null}catch{return null}}
export function clearSaved(key){try{localStorage.removeItem("nc90:"+key)}catch{}}
export function uid(prefix="id"){return prefix+"-"+Math.random().toString(36).slice(2,9)}
export function toast(message,kind=""){
  const el=document.getElementById("toast");if(!el)return;
  el.textContent=message;el.className="toast show "+kind;clearTimeout(toast.t);toast.t=setTimeout(()=>el.className="toast",1900);
}
export function canvasBox(canvas){
  const r=canvas.getBoundingClientRect(),dpr=Math.min(2,devicePixelRatio||1);
  const w=Math.max(10,Math.round(r.width*dpr)),h=Math.max(10,Math.round(r.height*dpr));
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}
  const ctx=canvas.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);return{ctx,w:r.width,h:r.height,dpr};
}
export function clearCanvas(canvas,color="#0b0f12"){const {ctx,w,h}=canvasBox(canvas);ctx.clearRect(0,0,w,h);ctx.fillStyle=color;ctx.fillRect(0,0,w,h);return{ctx,w,h}}
export function drawLine(canvas,values,{color="#ffb347",min=null,max=null,label=""}={}){
  const {ctx,w,h}=clearCanvas(canvas);const p=18,vals=values.filter(Number.isFinite);
  if(!vals.length){ctx.fillStyle="#64717a";ctx.font="9px sans-serif";ctx.fillText("NO DATA",9,16);return}
  let lo=min??Math.min(...vals),hi=max??Math.max(...vals);if(hi-lo<1e-9)hi=lo+1;
  ctx.strokeStyle="#273139";ctx.lineWidth=1;for(let i=0;i<4;i++){const y=p+(h-2*p)*i/3;ctx.beginPath();ctx.moveTo(p,y);ctx.lineTo(w-7,y);ctx.stroke()}
  ctx.strokeStyle=color;ctx.lineWidth=1.6;ctx.beginPath();
  values.forEach((v,i)=>{const x=p+(w-p-8)*(values.length===1?0:i/(values.length-1)),y=h-p-(h-2*p)*(v-lo)/(hi-lo);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();
  ctx.fillStyle="#6f7a82";ctx.font="7px ui-monospace,monospace";ctx.fillText(fmt(hi,2),2,p);ctx.fillText(fmt(lo,2),2,h-p);if(label)ctx.fillText(label,w-48,11);
}
export function matrixColor(v,maxAbs=1){
  const t=clamp(Math.abs(v)/(maxAbs||1),0,1),a=[18,24,28],b=v>=0?[92,205,216]:[226,105,101];
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*t)},${Math.round(a[1]+(b[1]-a[1])*t)},${Math.round(a[2]+(b[2]-a[2])*t)})`;
}
export function drawMatrix(canvas,mat,{signed=false,text=false}={}){
  const {ctx,w,h}=clearCanvas(canvas),rows=mat.length,cols=mat[0]?.length||0;if(!rows||!cols)return;
  let maxAbs=signed?Math.max(1e-9,...mat.flat().map(Math.abs)):1;const cw=w/cols,ch=h/rows;
  for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){const v=mat[r][c];ctx.fillStyle=signed?matrixColor(v,maxAbs):matrixColor(v-.5,.5);ctx.fillRect(c*cw,r*ch,cw+.7,ch+.7);if(text&&cw>24){ctx.fillStyle="#edf0eb";ctx.font="7px ui-monospace";ctx.textAlign="center";ctx.fillText(fmt(v,2),c*cw+cw/2,r*ch+ch/2+2)}}
}
export function resizeWatch(el,fn){let active=true,raf=0;const ro=new ResizeObserver(()=>{if(!active)return;cancelAnimationFrame(raf);raf=requestAnimationFrame(()=>{if(active&&el.isConnected)fn()})});ro.observe(el);return()=>{active=false;cancelAnimationFrame(raf);ro.disconnect()}}
export function logPush(arr,text,kind="",cap=80){arr.push({text,kind,t:new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit"})});if(arr.length>cap)arr.splice(0,arr.length-cap)}
export function logHTML(arr){return arr.slice().reverse().map(x=>`<div class="${x.kind||""}"><span>${x.t}</span> · ${x.text}</div>`).join("")}
export function htmlEscape(s){return String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
export function setCompleted(id){
  let s;try{s=new Set(JSON.parse(localStorage.getItem("nc90:completed")||"[]"))}catch{s=new Set()}
  s.add(id);localStorage.setItem("nc90:completed",JSON.stringify([...s]));
}
export function completed(){try{return new Set(JSON.parse(localStorage.getItem("nc90:completed")||"[]"))}catch{return new Set()}}
export function isFiniteDeep(value){
  if(typeof value==="number")return Number.isFinite(value);
  if(Array.isArray(value))return value.every(isFiniteDeep);
  if(value&&typeof value==="object")return Object.values(value).every(isFiniteDeep);
  return true;
}
