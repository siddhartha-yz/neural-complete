import {completed,setCompleted,toast,clearSaved} from "./core.js";
import xorLab from "./demos/xor-lab.js";
import featureFoundry from "./demos/feature-foundry.js";
import visionForge from "./demos/vision-forge.js";
import latentCartographer from "./demos/latent-cartographer.js";
import policyGarden from "./demos/policy-garden.js";

const demos=[xorLab,featureFoundry,visionForge,latentCartographer,policyGarden];
const byId=Object.fromEntries(demos.map(d=>[d.id,d]));
const app=document.getElementById("app");
let active=null,cleanup=null;

function mini(d){
  if(d.id==="xor-lab")return '<div class="mini-graph"><i></i><i></i><i></i><b></b><b></b><span></span></div>';
  if(d.id==="feature-foundry")return '<div class="mini-factory"><i>x₁</i><b>²</b><em>+</em><strong>f</strong></div>';
  if(d.id==="vision-forge")return '<div class="mini-vision"><i></i><i></i><i></i><span></span><span></span></div>';
  if(d.id==="latent-cartographer")return '<div class="mini-latent"><i></i><b></b><em></em><span></span></div>';
  return '<div class="mini-garden"><i></i><i></i><i></i><i></i><b>→</b><span>◎</span></div>';
}
export function home(){
  if(cleanup){cleanup();cleanup=null}
  active=null;
  if(location.hash)history.replaceState(null,"",location.pathname+location.search);
  const done=completed();
  app.innerHTML=`
    <section class="launcher">
      <header class="launch-head">
        <div class="brand90"><div class="brandbars"><i></i><i></i><i></i></div><div><b>NEURAL COMPLETE</b><span>BUILD THE LEARNING MACHINE</span></div></div>
        <div class="launch-score"><b>${done.size}/5</b><span>EXPERIMENTS SOLVED</span></div>
      </header>
      <div class="launch-hero">
        <span class="overline">SECOND PROTOTYPE SERIES / STRUCTURE FIRST</span>
        <h1>这次你不是“调模型”。<br><em>你要把模型造出来。</em></h1>
        <p>五个实验，五种完全不同的操作空间。共同规则只有一个：开发者给你基础元件，正确结构必须由你自己发现。</p>
      </div>
      <div class="launch-grid">
        ${demos.map((d,i)=>`
          <button class="launch-card theme-${d.theme}" data-demo="${d.id}">
            <div class="card-num">${String(i+1).padStart(2,"0")} ${done.has(d.id)?'<span>✓ SOLVED</span>':""}</div>
            ${mini(d)}
            <div class="card-copy"><small>${d.verb}</small><h2>${d.title}</h2><span>${d.en}</span><p>${d.card}</p></div>
            <div class="card-foot"><div>${d.tags.map(t=>`<i>${t}</i>`).join("")}</div><b>ENTER →</b></div>
          </button>`).join("")}
      </div>
      <footer class="launch-foot">
        <span>真实训练 · hidden evaluation · structural failure · internal probes</span>
        <a href="./DEMO_90_PLUS_REPORT.md" target="_blank">90+ design contract ↗</a>
      </footer>
    </section>`;
  app.querySelectorAll("[data-demo]").forEach(b=>b.onclick=()=>openDemo(b.dataset.demo));
}
export function openDemo(id){
  const d=byId[id];if(!d)return home();
  if(cleanup){cleanup();cleanup=null}
  active=d;if(location.hash!=="#"+id)history.replaceState(null,"","#"+id);
  app.innerHTML="";
  const ctx={
    home,
    toast,
    complete:()=>{setCompleted(id);toast(d.title+"：隐藏评测通过","good")},
    reset:()=>{if(confirm("清空这个实验的结构与训练状态？")){clearSaved(id);openDemo(id)}}
  };
  cleanup=d.mount(app,ctx)||null;
}
window.addEventListener("hashchange",()=>{const id=location.hash.slice(1);if(id&&id!==active?.id)openDemo(id);else if(!id&&active)home()});
window.__NC90__={home,openDemo,demos,get active(){return active}};
const initial=location.hash.slice(1);initial&&byId[initial]?openDemo(initial):home();
