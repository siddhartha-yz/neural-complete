import {toast,clearState} from "./core.js";
import boundary from "./demos/boundary.js";
import xor from "./demos/xor.js";
import conv from "./demos/conv.js";
import latent from "./demos/latent.js";
import qlearn from "./demos/qlearn.js";

const demos=[boundary,xor,conv,latent,qlearn];
const byId=Object.fromEntries(demos.map(d=>[d.id,d]));
const app=document.getElementById("app");
const topCenter=document.getElementById("topCenter");
const statusText=document.getElementById("statusText");
let current=null;
let currentCleanup=null;

function completedSet(){
  try{return new Set(JSON.parse(localStorage.getItem("nc5:completed")||"[]"))}
  catch{return new Set()}
}
function markCompleted(id){
  const s=completedSet();
  s.add(id);
  localStorage.setItem("nc5:completed",JSON.stringify([...s]));
  renderTopMeta();
}
function renderTopMeta(){
  const done=completedSet().size;
  document.getElementById("buildMeta").textContent=done+"/5 solved · no external ML library";
}
function home(){
  if(currentCleanup){currentCleanup();currentCleanup=null}
  current=null;
  if(location.hash) history.replaceState(null,"",location.pathname+location.search);
  topCenter.textContent="选择一个实验";
  statusText.textContent="所有训练都在浏览器内真实计算";
  const done=completedSet();
  app.innerHTML=
    '<section class="home">'+
      '<div class="hero">'+
        '<span class="eyebrow">NEURAL COMPLETE / FIVE EXPERIMENTS</span>'+
        '<h1>不要再“模拟机器学习”。<br>让模型真的学。</h1>'+
        '<p>五个互不相同的成品级实验。参数会从数据或奖励中更新；训练失败也是真失败。你需要改变数据、结构、表示或学习策略，而不是照着公式接线。</p>'+
      '</div>'+
      '<div class="demo-grid">'+
        demos.map((d,i)=>
          '<article class="demo-card" data-open="'+d.id+'" style="--card-color:'+d.color+'">'+
            '<span class="index">'+String(i+1).padStart(2,"0")+(done.has(d.id)?" · SOLVED":"")+'</span>'+
            '<h2>'+d.title+'</h2><span class="en">'+d.en+'</span>'+
            '<p>'+d.card+'</p>'+
            '<div class="tags">'+d.tags.map(t=>'<span>'+t+'</span>').join("")+'</div>'+
            '<div class="go"><span>'+(done.has(d.id)?"再次实验":"进入实验")+'</span><b>→</b></div>'+
          '</article>'
        ).join("")+
      '</div>'+
      '<div class="home-note"><b>验收规则</b><span>每个 Demo 都必须运行实际学习算法；必须有训练前后可量化差异、失败路径、独立隐藏评测以及可复现实验状态。浏览器刷新不会把训练结果伪造成预制动画。</span></div>'+
    '</section>';
  app.querySelectorAll("[data-open]").forEach(n=>n.addEventListener("click",()=>openDemo(n.dataset.open)));
  renderTopMeta();
}
function openDemo(id){
  const d=byId[id];
  if(!d){home();return}
  if(currentCleanup){currentCleanup();currentCleanup=null}
  current=d;
  if(location.hash!=="#"+id) history.replaceState(null,"","#"+id);
  topCenter.textContent=d.code+" · "+d.en.toUpperCase();
  statusText.textContent=d.status;
  app.innerHTML='<div class="empty-state"><b>Loading '+d.title+'</b><span>initializing numerical runtime…</span></div>';
  const ctx={
    toast,
    complete:()=>{markCompleted(id);toast(d.title+"：实验完成","good")},
    home,
    setStatus:s=>statusText.textContent=s
  };
  currentCleanup=d.mount(app,ctx)||null;
  renderTopMeta();
}
function resetCurrent(){
  if(!current){toast("当前没有打开实验");return}
  if(!confirm("重置「"+current.title+"」的训练状态？"))return;
  clearState(current.id);
  openDemo(current.id);
  toast("已重置当前实验");
}

document.getElementById("homeBtn").addEventListener("click",home);
document.getElementById("globalResetBtn").addEventListener("click",resetCurrent);
const about=document.getElementById("aboutDialog");
document.getElementById("aboutBtn").addEventListener("click",()=>about.showModal());
document.getElementById("closeAbout").addEventListener("click",()=>about.close());
window.addEventListener("hashchange",()=>{
  const id=location.hash.slice(1);
  if(id&&id!==current?.id) openDemo(id);
  else if(!id&&current) home();
});
window.__NC5__={
  demos,
  openDemo,
  home,
  get current(){return current},
  complete:markCompleted
};
const initial=location.hash.slice(1);
if(initial&&byId[initial]) openDemo(initial);
else home();
