(() => {
'use strict';

const $ = (id) => document.getElementById(id);
const q = (sel, root=document) => root.querySelector(sel);
const qa = (sel, root=document) => [...root.querySelectorAll(sel)];
const GRID = 16;

const PARTS = {
  const: {name:'CONST', symbol:'C', desc:'固定标量', inputs:[], outputs:['out'], limit:3, color:'accent'},
  mul:   {name:'MULTIPLY', symbol:'×', desc:'a × b', inputs:['a','b'], outputs:['out'], limit:2, color:'cyan'},
  add:   {name:'ADD', symbol:'+', desc:'a + b', inputs:['a','b'], outputs:['out'], limit:2, color:'green'},
  step:  {name:'STEP', symbol:'Θ', desc:'x ≥ 0 ? 1 : 0', inputs:['x'], outputs:['out'], limit:1, color:'purple'}
};
const MACRO = {name:'NEURON_01', symbol:'N', desc:'已封装神经元', inputs:['x1','x2'], outputs:['out'], limit:4, color:'green'};

const VISIBLE_TESTS = [
  {x1:0.00,x2:0.00,y:0},
  {x1:1.00,x2:0.00,y:1},
  {x1:0.00,x2:1.00,y:0},
  {x1:1.00,x2:1.00,y:1},
  {x1:0.35,x2:0.55,y:0},
  {x1:0.80,x2:0.55,y:1}
];
const VERIFY_TESTS = [
  {x1:0.40,x2:0.70,y:0},
  {x1:0.70,x2:0.70,y:1},
  {x1:0.30,x2:0.10,y:1},
  {x1:0.30,x2:0.40,y:0}
];

const state = {
  nodes: [],
  wires: [],
  selected: null,
  pendingSource: null,
  activeTest: -1,
  visibleResults: Array(VISIBLE_TESTS.length).fill(null),
  verifyResults: Array(VERIFY_TESTS.length).fill(null),
  failRuns: 0,
  history: [],
  simValues: new Map(),
  simError: '',
  compiled: false,
  compiledDefinition: null,
  runToken: 0
};

const FIXED = [
  {id:'input-x1', type:'input', label:'INPUT x₁', x:32, y:92, fixed:true, inputKey:'x1'},
  {id:'input-x2', type:'input', label:'INPUT x₂', x:32, y:238, fixed:true, inputKey:'x2'},
  {id:'output-y', type:'output', label:'OUTPUT ŷ', x:710, y:166, fixed:true}
];

function snap(n){ return Math.round(n/GRID)*GRID; }
function uid(prefix){ return prefix+'-'+Math.random().toString(36).slice(2,8); }
function cloneState(){
  return JSON.stringify({
    nodes:state.nodes,
    wires:state.wires,
    compiled:state.compiled,
    compiledDefinition:state.compiledDefinition
  });
}
function pushHistory(){
  const s=cloneState();
  if(state.history.at(-1)!==s) state.history.push(s);
  if(state.history.length>60) state.history.shift();
  $('undoBtn').disabled = state.history.length<2;
}
function restoreSnapshot(s){
  const v=JSON.parse(s);
  state.nodes=v.nodes||[]; state.wires=v.wires||[]; state.compiled=!!v.compiled;
  state.compiledDefinition=v.compiledDefinition||null;
  state.selected=null; state.pendingSource=null; state.simValues=new Map(); state.simError='';
  clearResults(false); renderAll();
}
function undo(){
  if(state.history.length<2) return;
  state.history.pop();
  restoreSnapshot(state.history.at(-1));
  saveLocal();
  $('undoBtn').disabled=state.history.length<2;
  toast('已撤销');
}

function initialNodes(){
  return FIXED.map(n=>({...n}));
}
function resetBoard(confirmFirst=true){
  if(confirmFirst && state.nodes.length>FIXED.length && !confirm('清空当前画布？已封装的组件会保留。')) return;
  const keepCompiled=state.compiled, keepDefinition=state.compiledDefinition;
  state.nodes=initialNodes(); state.wires=[]; state.selected=null; state.pendingSource=null;
  state.simValues=new Map(); state.simError=''; state.compiled=keepCompiled; state.compiledDefinition=keepDefinition; state.history=[];
  clearResults(true); renderAll(); pushHistory();
  if(state.compiled) saveLocal(); else localStorage.removeItem('neural-complete-p01');
}
function saveLocal(){
  try { localStorage.setItem('neural-complete-p01', cloneState()); } catch {}
}
function loadLocal(){
  try {
    const raw=localStorage.getItem('neural-complete-p01');
    if(!raw) return false;
    const v=JSON.parse(raw);
    if(!Array.isArray(v.nodes)||!Array.isArray(v.wires)) return false;
    state.nodes=v.nodes; state.wires=v.wires; state.compiled=!!v.compiled;
    state.compiledDefinition=v.compiledDefinition||null;
    return true;
  } catch { return false; }
}

function partCounts(){
  const counts={};
  Object.keys(PARTS).forEach(t=>counts[t]=state.nodes.filter(n=>n.type===t).length);
  return counts;
}
function addPart(type, x=280, y=150){
  const counts=partCounts();
  const p=type==='neuron'?MACRO:PARTS[type];
  const used=type==='neuron'?state.nodes.filter(n=>n.type==='neuron').length:counts[type];
  if(type==='neuron' && !state.compiled){toast('先完成关卡并封装神经元','error');return null}
  if(!p || used>=p.limit) { toast('这个元件已经用完了','error'); return null; }
  const board=$('board').getBoundingClientRect();
  const nx=Math.max(0,Math.min(board.width-120,snap(x)));
  const ny=Math.max(0,Math.min(board.height-82,snap(y)));
  const node={id:uid(type),type,label:p.name,x:nx,y:ny,fixed:false,value:type==='const'?1:undefined};
  state.nodes.push(node); state.selected={kind:'node',id:node.id}; clearResults();
  pushHistory(); saveLocal(); renderAll(); return node;
}

function portsFor(node){
  if(node.type==='input') return {inputs:[],outputs:['out']};
  if(node.type==='output') return {inputs:['in'],outputs:[]};
  const p=node.type==='neuron'?MACRO:PARTS[node.type];
  return {inputs:p?.inputs||[],outputs:p?.outputs||[]};
}
function getNode(id){return state.nodes.find(n=>n.id===id)}
function incoming(nodeId,port){
  return state.wires.find(w=>w.to.node===nodeId && w.to.port===port);
}
function outgoing(nodeId,port='out'){
  return state.wires.filter(w=>w.from.node===nodeId && w.from.port===port);
}
function wireKey(w){return w.id}
function nodeValueKey(id,port='out'){return id+':'+port}

function connect(from,to){
  if(from.node===to.node){toast('不能把元件接到自己','error');return}
  const targetNode=getNode(to.node);
  if(!targetNode || !portsFor(targetNode).inputs.includes(to.port)) return;
  const sourceNode=getNode(from.node);
  if(!sourceNode || !portsFor(sourceNode).outputs.includes(from.port)) return;
  const existing=incoming(to.node,to.port);
  if(existing) state.wires=state.wires.filter(w=>w.id!==existing.id);
  const same=state.wires.find(w=>w.from.node===from.node&&w.from.port===from.port&&w.to.node===to.node&&w.to.port===to.port);
  if(same) return;
  const wire={id:uid('wire'),from:{...from},to:{...to}};
  state.wires.push(wire); state.pendingSource=null; state.selected={kind:'wire',id:wire.id};
  clearResults(); pushHistory(); saveLocal(); renderAll();
}
function deleteSelection(){
  if(!state.selected) return;
  if(state.selected.kind==='node'){
    const node=getNode(state.selected.id);
    if(!node || node.fixed) {toast('输入/输出端口属于关卡，不能删除','error');return}
    state.nodes=state.nodes.filter(n=>n.id!==node.id);
    state.wires=state.wires.filter(w=>w.from.node!==node.id&&w.to.node!==node.id);
  } else {
    state.wires=state.wires.filter(w=>w.id!==state.selected.id);
  }
  state.selected=null; clearResults(); pushHistory(); saveLocal(); renderAll();
}

function evaluateGraph(nodes,wires,test,compiledDefinition=null){
  const memo=new Map(), visiting=new Set(), error={msg:''};
  const nget=id=>nodes.find(n=>n.id===id);
  const inc=(nodeId,port)=>wires.find(w=>w.to.node===nodeId&&w.to.port===port);

  function evalNode(nodeId,port='out'){
    const key=nodeValueKey(nodeId,port);
    if(memo.has(key)) return memo.get(key);
    if(visiting.has(key)){error.msg='检测到回路：本关只支持前馈计算图。';return null}
    visiting.add(key);
    const node=nget(nodeId);
    if(!node){error.msg='导线指向了不存在的元件。';visiting.delete(key);return null}
    let value=null;

    if(node.type==='input'){
      value=Number(test[node.inputKey]);
    } else if(node.type==='const'){
      value=Number(node.value);
    } else if(node.type==='output'){
      const w=inc(node.id,'in');
      if(!w) error.msg='输出 ŷ 还没有接线。';
      else value=evalNode(w.from.node,w.from.port);
    } else {
      const vals={};
      for(const input of portsFor(node).inputs){
        const w=inc(node.id,input);
        if(!w){error.msg=node.label+' 的 '+input+' 输入未连接。';value=null;break}
        vals[input]=evalNode(w.from.node,w.from.port);
        if(vals[input]===null||Number.isNaN(vals[input])){value=null;break}
      }
      if(!error.msg){
        if(node.type==='mul') value=vals.a*vals.b;
        else if(node.type==='add') value=vals.a+vals.b;
        else if(node.type==='step') value=vals.x>=0?1:0;
        else if(node.type==='neuron'){
          if(!compiledDefinition){error.msg='NEURON_01 的内部定义不存在。';}
          else {
            const inner=evaluateGraph(compiledDefinition.nodes,compiledDefinition.wires,{x1:vals.x1,x2:vals.x2},null);
            if(inner.error) error.msg='NEURON_01 内部错误：'+inner.error;
            else value=inner.y;
          }
        }
      }
    }
    visiting.delete(key);
    if(value!==null&&Number.isFinite(value)) memo.set(key,value);
    return value;
  }

  const y=evalNode('output-y','out');
  const values=new Map(memo);
  for(const w of wires){
    const v=evalNode(w.from.node,w.from.port);
    if(v!==null&&Number.isFinite(v)) values.set('wire:'+w.id,v);
  }
  return {y,error:error.msg,values};
}

function evalCircuit(test){
  return evaluateGraph(state.nodes,state.wires,test,state.compiledDefinition);
}

function formatValue(v){
  if(v===null||v===undefined||Number.isNaN(v)) return '—';
  if(Math.abs(v)<1e-8) return '0';
  if(Number.isInteger(v)) return String(v);
  return Number(v).toFixed(Math.abs(v)<10?2:1).replace(/\.00$/,'');
}
function signalClass(v){
  if(v===null||v===undefined) return '';
  if(Math.abs(v)<1e-8) return 'signal-zero';
  return v>0?'signal-pos':'signal-neg';
}

function runOne(index, source='visible'){
  const tests=source==='verify'?VERIFY_TESTS:VISIBLE_TESTS;
  const test=tests[index];
  if(!test) return false;
  const result=evalCircuit(test);
  state.simValues=result.values; state.simError=result.error;
  if(source==='visible'){
    state.activeTest=index;
    state.visibleResults[index]=result.error?false:(result.y===test.y);
  }
  updateSignalUI(test,result,source,index);
  renderBoard();
  renderTests();
  return !result.error && result.y===test.y;
}

async function runVisible(){
  state.runToken++; const token=state.runToken;
  state.visibleResults=Array(VISIBLE_TESTS.length).fill(null);
  $('footerStatus').textContent='SIMULATING'; $('runAllBtn').disabled=true;
  let pass=0;
  for(let i=0;i<VISIBLE_TESTS.length;i++){
    if(token!==state.runToken) return;
    state.activeTest=i; renderTests();
    const ok=runOne(i,'visible'); if(ok) pass++;
    await wait(180);
  }
  $('runAllBtn').disabled=false; $('footerStatus').textContent='EDIT MODE';
  const all=pass===VISIBLE_TESTS.length;
  if(all){
    $('verifyBtn').disabled=false;
    $('verifyMeta').textContent='公开测试通过，可以进行额外校验';
    toast('公开测试全部通过。进行最终验证。','success');
  } else {
    state.failRuns++;
    toast('还有测试失败。沿着信号值检查你的设计。','error');
    maybeUnlockHint();
  }
  renderTests(); saveLocal();
}
async function verifyDesign(){
  state.verifyResults=Array(VERIFY_TESTS.length).fill(null);
  $('verifyBtn').disabled=true; $('verifyTitle').textContent='正在校验…';
  let pass=0, firstFail=null;
  for(let i=0;i<VERIFY_TESTS.length;i++){
    const result=evalCircuit(VERIFY_TESTS[i]);
    const ok=!result.error&&result.y===VERIFY_TESTS[i].y;
    state.verifyResults[i]=ok;
    if(ok) pass++; else if(firstFail===null) firstFail={i,result};
    await wait(130);
  }
  if(pass===VERIFY_TESTS.length){
    $('verifyTitle').textContent='4 / 4 通过';
    $('verifyMeta').textContent='设计满足完整关卡契约';
    const used=state.nodes.filter(n=>!n.fixed).length;
    const constants=state.nodes.filter(n=>n.type==='const').map(n=>Number(n.value));
    $('victoryStats').innerHTML='<div><span>元件数</span><b>'+used+'</b></div><div><span>导线数</span><b>'+state.wires.length+'</b></div><div><span>常数</span><b>'+constants.map(formatValue).join(', ')+'</b></div>';
    setTimeout(()=>$('victory').classList.remove('hidden'),250);
  } else {
    state.failRuns++;
    $('verifyTitle').textContent=pass+' / 4 通过';
    $('verifyMeta').textContent='发现一个反例，已送到信号监视器';
    $('verifyBtn').disabled=false;
    if(firstFail){
      const t=VERIFY_TESTS[firstFail.i];
      state.simValues=firstFail.result.values; state.simError=firstFail.result.error;
      updateSignalUI(t,firstFail.result,'verify',firstFail.i);
      renderBoard();
      $('simMessage').textContent='额外校验反例：x₁='+t.x1+'，x₂='+t.x2+'，期望 '+t.y+'。调整后再验证。';
      $('simMessage').className='sim-message error';
    }
    toast('最终验证失败：发现了公开测试没有覆盖的反例。','error');
    maybeUnlockHint();
  }
}

function wait(ms){return new Promise(r=>setTimeout(r,ms))}
function clearResults(resetVerify=true){
  state.visibleResults=Array(VISIBLE_TESTS.length).fill(null);
  if(resetVerify) state.verifyResults=Array(VERIFY_TESTS.length).fill(null);
  state.simValues=new Map();state.simError='';
  $('verifyBtn').disabled=true;$('verifyTitle').textContent='额外校验 × 4';$('verifyMeta').textContent='先通过上方 6 个公开测试';
}
function maybeUnlockHint(){
  if(state.failRuns>=2){
    $('hintBtn').disabled=false;$('hintMeta').textContent='可用 · 不会直接给出参数';
  }
}

function updateSignalUI(test,result,source,index){
  $('ioX1').textContent=formatValue(test.x1);$('ioX2').textContent=formatValue(test.x2);$('ioY').textContent=formatValue(result.y);
  $('activeCaseLabel').textContent=(source==='verify'?'VERIFY ':'CASE ')+String(index+1).padStart(2,'0');
  if(result.error){
    $('simMessage').textContent=result.error;$('simMessage').className='sim-message error';
  } else {
    const ok=result.y===test.y;
    $('simMessage').textContent='输出 '+formatValue(result.y)+' · 期望 '+test.y+(ok?' · PASS':' · FAIL');
    $('simMessage').className='sim-message '+(ok?'ok':'error');
  }
}
function stepTest(){
  const next=(state.activeTest+1)%VISIBLE_TESTS.length;
  runOne(next,'visible');
}

function renderPalette(){
  const counts=partCounts();
  $('palette').innerHTML='';
  for(const [type,p] of Object.entries(PARTS)){
    const left=p.limit-counts[type], el=document.createElement('div');
    el.className='part'+(left<=0?' empty':'');
    el.draggable=left>0; el.dataset.type=type;
    el.innerHTML='<div class="part-icon">'+p.symbol+'</div><div class="part-copy"><b>'+p.name+'</b><small>'+p.desc+'</small></div><div class="part-count">'+left+'/'+p.limit+'</div>';
    el.addEventListener('dragstart',e=>{e.dataTransfer.setData('application/x-neural-part',type);e.dataTransfer.effectAllowed='copy'});
    el.addEventListener('dblclick',()=>{ if(left>0) addPart(type,270+Math.random()*100,100+Math.random()*120) });
    $('palette').appendChild(el);
  }
  const shelf=$('compiledShelf');
  shelf.classList.toggle('hidden',!state.compiled);
  const chip=q('.compiled-chip',shelf);
  if(chip){
    chip.draggable=!!state.compiled;
    chip.dataset.type='neuron';
    chip.ondragstart=e=>{e.dataTransfer.setData('application/x-neural-part','neuron');e.dataTransfer.effectAllowed='copy'};
    chip.ondblclick=()=>{if(state.compiled)addPart('neuron',Math.max(180,$('board').clientWidth*.45),Math.max(90,$('board').clientHeight*.42))};
  }
}
function nodeMarkup(node){
  const def=node.fixed?null:(node.type==='neuron'?MACRO:PARTS[node.type]);
  const symbol=node.type==='input'?'→':node.type==='output'?'◎':def.symbol;
  const p=portsFor(node);
  const inputs=p.inputs.map(name=>'<div class="port-wrap input"><button class="port input-port" data-node="'+node.id+'" data-port="'+name+'" data-kind="input" aria-label="'+name+' input"></button><span class="port-label">'+name+'</span></div>').join('');
  const outputs=p.outputs.map(name=>'<div class="port-wrap output"><button class="port output-port" data-node="'+node.id+'" data-port="'+name+'" data-kind="output" aria-label="'+name+' output"></button><span class="port-label">'+name+'</span></div>').join('');
  const value=state.simValues.get(nodeValueKey(node.id,'out'));
  let live='';
  if(node.type==='const') live='<span class="node-live '+(value!==undefined?'hot':'')+'">'+formatValue(Number(node.value))+'</span>';
  else if(value!==undefined) live='<span class="node-live hot">'+formatValue(value)+'</span>';
  let desc=node.type==='input'?'测试输入':node.type==='output'?'关卡输出':def.desc;
  return '<div class="node-header" data-drag-handle><span class="node-title">'+node.label+'</span><span class="node-symbol">'+symbol+'</span></div><div class="node-body"><span class="node-desc">'+desc+'</span>'+live+'<div class="port-row"><div class="ports inputs">'+inputs+'</div><div class="ports outputs">'+outputs+'</div></div></div>';
}
function layoutFixedTerminals(){
  const board=$('board');
  if(!board)return;
  const w=board.clientWidth,h=board.clientHeight;
  const x1=getNode('input-x1'),x2=getNode('input-x2'),out=getNode('output-y');
  if(x1){x1.x=24;x1.y=snap(Math.max(42,h*.26-35))}
  if(x2){x2.x=24;x2.y=snap(Math.max(130,h*.64-35))}
  if(out){out.x=snap(Math.max(150,w-140));out.y=snap(Math.max(80,h*.45-35))}
}
function renderBoard(){
  layoutFixedTerminals();
  const layer=$('nodeLayer'); layer.innerHTML='';
  for(const node of state.nodes){
    const el=document.createElement('div');el.className='node'+(node.fixed?' fixed':'')+(state.selected?.kind==='node'&&state.selected.id===node.id?' selected':'');
    el.dataset.node=node.id; el.style.left=node.x+'px';el.style.top=node.y+'px';el.innerHTML=nodeMarkup(node);
    el.addEventListener('pointerdown',ev=>{if(ev.target.closest('.port'))return;selectNode(node.id);if(ev.target.closest('[data-drag-handle]'))startNodeDrag(ev,node.id)});
    layer.appendChild(el);
  }
  bindPorts(); renderWires();
  $('boardNote').classList.toggle('hidden',state.nodes.length>FIXED.length);
  $('partCount').textContent=(state.nodes.length-FIXED.length)+' components · '+state.wires.length+' wires';
  const outputConnected=!!incoming('output-y','in');
  $('boardStatus').textContent=outputConnected?'计算图已接到输出':'未接通';
  renderInspector();
}
function bindPorts(){
  qa('.port').forEach(port=>{
    const ref={node:port.dataset.node,port:port.dataset.port};
    if(port.dataset.kind==='output'){
      if(outgoing(ref.node,ref.port).length) port.classList.add('connected');
      const v=state.simValues.get(nodeValueKey(ref.node,ref.port));if(v!==undefined)port.classList.add(signalClass(v));
      if(state.pendingSource&&state.pendingSource.node===ref.node&&state.pendingSource.port===ref.port)port.classList.add('source-pending');
      port.addEventListener('click',ev=>{ev.stopPropagation();state.pendingSource=ref;state.selected={kind:'node',id:ref.node};updateWiringMode();renderBoard()});
    } else {
      const w=incoming(ref.node,ref.port);if(w)port.classList.add('connected');
      if(w){const v=state.simValues.get('wire:'+w.id);if(v!==undefined)port.classList.add(signalClass(v))}
      port.addEventListener('click',ev=>{ev.stopPropagation();if(state.pendingSource){connect(state.pendingSource,ref)}else{state.selected={kind:'node',id:ref.node};renderBoard()}});
    }
  });
}
function portCenter(ref){
  const el=q('.port[data-node="'+ref.node+'"][data-port="'+ref.port+'"]');
  if(!el)return null;const r=el.getBoundingClientRect(),b=$('board').getBoundingClientRect();
  return{x:r.left-b.left+r.width/2,y:r.top-b.top+r.height/2};
}
function bezierPath(a,b){
  const dx=Math.max(45,Math.abs(b.x-a.x)*.45);
  return 'M '+a.x+' '+a.y+' C '+(a.x+dx)+' '+a.y+', '+(b.x-dx)+' '+b.y+', '+b.x+' '+b.y;
}
function renderWires(){
  const svg=$('wireLayer');svg.innerHTML='';
  for(const wire of state.wires){
    const a=portCenter(wire.from),b=portCenter(wire.to);if(!a||!b)continue;
    const path=document.createElementNS('http://www.w3.org/2000/svg','path');
    const val=state.simValues.get('wire:'+wire.id);
    path.setAttribute('d',bezierPath(a,b));path.setAttribute('class','wire '+signalClass(val)+(state.selected?.kind==='wire'&&state.selected.id===wire.id?' selected':''));
    path.dataset.wire=wire.id;path.addEventListener('click',ev=>{ev.stopPropagation();state.selected={kind:'wire',id:wire.id};renderAll()});
    svg.appendChild(path);
  }
  if(state.pendingSource){
    const a=portCenter(state.pendingSource);if(a){
      const path=document.createElementNS('http://www.w3.org/2000/svg','path');path.id='ghostWire';path.setAttribute('class','wire ghost');path.setAttribute('d',bezierPath(a,a));svg.appendChild(path);
    }
  }
}
function updateGhost(ev){
  if(!state.pendingSource)return;const a=portCenter(state.pendingSource),path=$('ghostWire'),b=$('board').getBoundingClientRect();if(!a||!path)return;
  path.setAttribute('d',bezierPath(a,{x:ev.clientX-b.left,y:ev.clientY-b.top}));
}
function updateWiringMode(){
  const active=!!state.pendingSource;$('wireCancelBtn').classList.toggle('hidden',!active);
  q('.wiring-help')?.classList.toggle('active',active);
  $('wireHelp').textContent=active?'选择一个输入端口完成连接':'点击任意输出端口开始接线';
  $('footerStatus').textContent=active?'WIRING':'EDIT MODE';
}
function cancelWire(){state.pendingSource=null;updateWiringMode();renderBoard()}

function selectNode(id){state.selected={kind:'node',id};renderAll()}
function startNodeDrag(ev,id){
  const node=getNode(id);if(!node||node.fixed)return;
  ev.preventDefault();const board=$('board').getBoundingClientRect();const start={x:ev.clientX,y:ev.clientY,nx:node.x,ny:node.y};let moved=false;
  function move(e){moved=true;node.x=Math.max(0,Math.min(board.width-120,snap(start.nx+e.clientX-start.x)));node.y=Math.max(0,Math.min(board.height-80,snap(start.ny+e.clientY-start.y)));const el=q('.node[data-node="'+id+'"]');if(el){el.style.left=node.x+'px';el.style.top=node.y+'px'}renderWires()}
  function up(){window.removeEventListener('pointermove',move);window.removeEventListener('pointerup',up);if(moved){clearResults();pushHistory();saveLocal();renderAll()}}
  window.addEventListener('pointermove',move);window.addEventListener('pointerup',up);
}

function renderTests(){
  const root=$('testRows');root.innerHTML='';
  VISIBLE_TESTS.forEach((t,i)=>{
    const res=state.visibleResults[i],row=document.createElement('div');
    row.className='test-row'+(i===state.activeTest?' active':'')+(res===true?' pass':res===false?' fail':'');
    row.dataset.case=i;row.innerHTML='<span class="case">'+String(i+1).padStart(2,'0')+'</span><span>'+formatValue(t.x1)+'</span><span>'+formatValue(t.x2)+'</span><span class="result-mark">'+(res===true?'✓':res===false?'×':'·')+'</span><span class="expect">→ '+t.y+'</span>';
    row.addEventListener('click',()=>runOne(i,'visible'));root.appendChild(row);
  });
  const pass=state.visibleResults.filter(x=>x===true).length;$('testSummary').textContent=pass+' / '+VISIBLE_TESTS.length;
  if(pass===VISIBLE_TESTS.length){$('verifyBtn').disabled=false;$('verifyMeta').textContent='公开测试通过，可以进行额外校验'}
}

function renderInspector(){
  const empty=$('inspectorEmpty'),content=$('inspectorContent');
  if(!state.selected){empty.classList.remove('hidden');content.classList.add('hidden');return}
  empty.classList.add('hidden');content.classList.remove('hidden');
  if(state.selected.kind==='wire'){
    const w=state.wires.find(x=>x.id===state.selected.id);if(!w){state.selected=null;return renderInspector()}
    const v=state.simValues.get('wire:'+w.id);
    content.innerHTML='<span class="type-code">WIRE</span><h3>信号导线</h3><div class="inspect-card"><div class="inspect-row"><span>FROM</span><b>'+labelOf(w.from.node)+'.'+w.from.port+'</b></div><div class="inspect-row"><span>TO</span><b>'+labelOf(w.to.node)+'.'+w.to.port+'</b></div><div class="inspect-row"><span>VALUE</span><b>'+formatValue(v)+'</b></div></div><button class="delete-btn" id="inspectDelete">删除导线</button>';
    $('inspectDelete').onclick=deleteSelection;return;
  }
  const n=getNode(state.selected.id);if(!n){state.selected=null;return renderInspector()}
  const p=portsFor(n),v=state.simValues.get(nodeValueKey(n.id,'out'));
  let html='<span class="type-code">'+n.type.toUpperCase()+'</span><h3>'+n.label+'</h3>';
  if(n.type==='const'){
    html+='<div class="inspect-card"><label>CONSTANT VALUE</label><input id="constInput" type="number" step="0.05" min="-5" max="5" value="'+Number(n.value)+'"><div class="inspect-row"><span>当前输出</span><b>'+formatValue(Number(n.value))+'</b></div></div>';
  } else {
    html+='<div class="inspect-card"><div class="inspect-row"><span>输入端口</span><b>'+p.inputs.length+'</b></div><div class="inspect-row"><span>输出端口</span><b>'+p.outputs.length+'</b></div><div class="inspect-row"><span>当前输出</span><b>'+formatValue(v)+'</b></div></div>';
  }
  if(!n.fixed) html+='<button class="delete-btn" id="inspectDelete">删除元件</button>';
  content.innerHTML=html;
  if(n.type==='const'){
    const input=$('constInput');input.addEventListener('change',()=>{let val=Math.max(-5,Math.min(5,Number(input.value)));if(!Number.isFinite(val))val=0;n.value=val;clearResults();pushHistory();saveLocal();renderAll()});
  }
  if($('inspectDelete'))$('inspectDelete').onclick=deleteSelection;
}
function labelOf(id){return getNode(id)?.label||id}

function centerView(){
  const movable=state.nodes.filter(n=>!n.fixed);
  if(!movable.length)return;
  const board=$('board').getBoundingClientRect();
  const left=160,right=Math.max(left+250,board.width-285);
  const xs={
    const:left,
    mul:left+(right-left)*.36,
    add:left+(right-left)*.68,
    step:right,
    neuron:left+(right-left)*.54
  };
  const groups={};
  movable.forEach(n=>(groups[n.type]??=[]).push(n));
  for(const [type,nodes] of Object.entries(groups)){
    const span=Math.min(board.height-140,Math.max(90,(nodes.length-1)*125));
    const top=Math.max(45,(board.height-span)/2-32);
    nodes.forEach((n,i)=>{
      n.x=snap(Math.min(board.width-130,xs[type]??left+(right-left)*.5));
      n.y=snap(nodes.length===1?board.height*.46-35:top+i*(span/Math.max(1,nodes.length-1)));
    });
  }
  clearResults();pushHistory();saveLocal();renderAll();
}
function renderAll(){renderPalette();renderBoard();renderTests();updateWiringMode();$('undoBtn').disabled=state.history.length<2}

function setupTabs(){
  qa('.tab').forEach(tab=>tab.addEventListener('click',()=>{qa('.tab').forEach(x=>x.classList.remove('active'));qa('.tab-pane').forEach(x=>x.classList.remove('active'));tab.classList.add('active');$(tab.dataset.tab+'Tab').classList.add('active')}));
}
function setupBoardDnD(){
  const board=$('board');
  board.addEventListener('dragover',e=>{if(e.dataTransfer.types.includes('application/x-neural-part')){e.preventDefault();e.dataTransfer.dropEffect='copy'}});
  board.addEventListener('drop',e=>{const type=e.dataTransfer.getData('application/x-neural-part');if(!type)return;e.preventDefault();const r=board.getBoundingClientRect();addPart(type,e.clientX-r.left-58,e.clientY-r.top-35)});
  board.addEventListener('pointermove',updateGhost);
  board.addEventListener('click',e=>{if(e.target===board||e.target===q('.node-layer')){state.selected=null;renderAll()}});
}
function toast(msg,type=''){
  const el=$('toast');el.textContent=msg;el.className='toast show '+type;clearTimeout(toast.t);toast.t=setTimeout(()=>el.className='toast',1800)
}
function compileNeuron(){
  if(state.compiled && state.nodes.some(n=>n.type==='neuron')){
    $('victory').classList.add('hidden');
    toast('NEURON_01 已经封装在元件库中','success');
    return;
  }
  const allowed=new Set(state.nodes.filter(n=>n.type!=='neuron').map(n=>n.id));
  state.compiledDefinition={
    nodes:JSON.parse(JSON.stringify(state.nodes.filter(n=>allowed.has(n.id)))),
    wires:JSON.parse(JSON.stringify(state.wires.filter(w=>allowed.has(w.from.node)&&allowed.has(w.to.node))))
  };
  state.compiled=true;
  saveLocal();pushHistory();
  $('victory').classList.add('hidden');renderAll();
  toast('NEURON_01 已封装：现在可以像普通元件一样拖入画布','success');
  $('footerStatus').textContent='COMPONENT UNLOCKED';
}
function showHint(){
  $('hintText').classList.remove('hidden');
  $('hintText').innerHTML='<b>提示：</b>试着让一个输入“推动”总分，让另一个输入“拉低”总分。最后再问：总分跨过 0 时，输出应该发生什么变化？';
}
function keyboard(e){
  if(e.target.matches('input'))return;
  if(e.key==='Delete'||e.key==='Backspace'){e.preventDefault();deleteSelection()}
  if(e.key==='Escape')cancelWire();
  if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();undo()}
}

function init(){
  loadLocal(); if(!state.nodes.length)state.nodes=initialNodes();
  setupTabs();setupBoardDnD();
  if(localStorage.getItem('neural-complete-p01-intro')==='seen') $('intro').classList.add('hidden');
  $('enterBtn').onclick=()=>{localStorage.setItem('neural-complete-p01-intro','seen');$('intro').classList.add('hidden');$('board').focus()};
  $('undoBtn').onclick=undo;$('resetBtn').onclick=()=>resetBoard(true);$('wireCancelBtn').onclick=cancelWire;$('centerBtn').onclick=centerView;
  $('stepTestBtn').onclick=stepTest;$('runAllBtn').onclick=runVisible;$('verifyBtn').onclick=verifyDesign;
  $('hintBtn').onclick=showHint;$('compileBtn').onclick=compileNeuron;$('stayBtn').onclick=()=>$('victory').classList.add('hidden');
  window.addEventListener('keydown',keyboard);window.addEventListener('resize',()=>renderBoard());
  renderAll();pushHistory();
  window.__NC__={state,addPart,connect,evalCircuit,runVisible,verifyDesign,resetBoard,PARTS,VISIBLE_TESTS,VERIFY_TESTS};
}
init();
})();