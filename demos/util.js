import {el} from "../core.js";

export function demoShell(root,d){
  root.innerHTML=`
    <section class="demo" data-demo="${d.id}">
      <aside class="rail">
        <span class="section-kicker">${d.code} / ${d.kind}</span>
        <h1>${d.title}</h1>
        <p class="subtitle">${d.subtitle}</p>
        <div class="mission-card"><b>${d.missionTitle}</b><p>${d.mission}</p></div>
        <div id="${d.id}-controls"></div>
        <div class="divider"></div>
        <span class="section-kicker">WHAT TO NOTICE</span>
        <ul class="rule-list">${d.notices.map(x=>`<li>${x}</li>`).join("")}</ul>
      </aside>
      <section class="stage">
        <div class="stage-head"><div class="mode"><b>${d.stageTitle}</b><span id="${d.id}-stage-status">READY</span></div><span class="epoch-badge" id="${d.id}-epoch">STEP 0</span></div>
        <div class="stage-body"><div id="${d.id}-workspace" class="workspace-grid"></div></div>
      </section>
      <aside class="inspector" id="${d.id}-inspector"></aside>
    </section>`;
  return {
    controls:document.getElementById(d.id+"-controls"),
    workspace:document.getElementById(d.id+"-workspace"),
    inspector:document.getElementById(d.id+"-inspector"),
    status:document.getElementById(d.id+"-stage-status"),
    epoch:document.getElementById(d.id+"-epoch")
  };
}
export function rangeControl({id,label,min,max,step,value,unit=""}){
  return `<div class="control-card"><label for="${id}">${label}</label><div class="control-row"><input id="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${value}"><b class="control-value" id="${id}-v">${value}${unit}</b></div></div>`
}
export function selectControl({id,label,options,value}){
  return `<div class="control-card"><label for="${id}">${label}</label><select id="${id}">${options.map(o=>`<option value="${o.value}" ${String(o.value)===String(value)?"selected":""}>${o.label}</option>`).join("")}</select></div>`
}
export function actions(items){
  return `<div class="action-stack">${items.map(x=>`<button class="action-btn ${x.primary?"primary":""}" id="${x.id}">${x.icon||""}<span>${x.label}</span></button>`).join("")}</div>`
}
export function bindRange(id,cb,format=v=>v){
  const n=document.getElementById(id),out=document.getElementById(id+"-v");if(!n)return()=>{};
  const f=()=>{out.textContent=format(n.value);cb(Number(n.value))};n.addEventListener("input",f);return()=>n.removeEventListener("input",f)
}
export function setHtml(elm,html){if(elm)elm.innerHTML=html}
export function button(id){return document.getElementById(id)}
