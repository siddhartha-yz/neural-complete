import {sigmoid,tanh,relu,bce,mean,seeded,randn,deepCopy,isFiniteDeep} from "./core.js";

function act(kind,z){
  if(kind==="relu")return relu(z);
  if(kind==="tanh")return tanh(z);
  if(kind==="sigmoid")return sigmoid(z);
  return z;
}
function dact(kind,z,h){
  if(kind==="relu")return z>0?1:0;
  if(kind==="tanh")return 1-h*h;
  if(kind==="sigmoid")return h*(1-h);
  return 1;
}
function hashCode(s){let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)}return h>>>0}

export class LearningGraph{
  constructor(state=null){
    this.nodes=state?.nodes?deepCopy(state.nodes):[
      {id:"x1",type:"input",key:"x1",label:"x₁",x:50,y:160,fixed:true},
      {id:"x2",type:"input",key:"x2",label:"x₂",x:50,y:330,fixed:true},
      {id:"out",type:"output",label:"ŷ",x:780,y:245,fixed:true,bias:0,weights:{},frozen:false}
    ];
    this.edges=state?.edges?deepCopy(state.edges):[];
    this.version=state?.version||1;
    this.ensureParameters();
  }
  toJSON(){return{nodes:deepCopy(this.nodes),edges:deepCopy(this.edges),version:this.version}}
  node(id){return this.nodes.find(n=>n.id===id)}
  incoming(id){return this.edges.filter(e=>e.to===id)}
  outgoing(id){return this.edges.filter(e=>e.from===id)}
  addNode(node){this.nodes.push(deepCopy(node));this.ensureParameters();return node}
  removeNode(id){
    const n=this.node(id);if(!n||n.fixed)return false;
    this.edges=this.edges.filter(e=>e.from!==id&&e.to!==id);this.nodes=this.nodes.filter(n=>n.id!==id);this.ensureParameters();return true
  }
  addEdge(from,to,id){
    if(from===to||!this.node(from)||!this.node(to))return null;
    if(this.edges.some(e=>e.from===from&&e.to===to))return null;
    const edge={id:id||"e"+Math.random().toString(36).slice(2,9),from,to};this.edges.push(edge);this.ensureParameters();return edge
  }
  removeEdge(id){this.edges=this.edges.filter(e=>e.id!==id);this.ensureParameters()}
  reinitialize(seed=1337){
    const rng=seeded(seed);
    for(const n of this.nodes){
      if(n.type==="neuron"||n.type==="output"){
        const fan=Math.max(1,this.incoming(n.id).length);
        const scale=n.type==="neuron"&&n.activation==="relu"?Math.sqrt(2/fan):Math.sqrt(1/fan);
        n.bias=randn(rng)*0.04;
        n.weights={};
        for(const e of this.incoming(n.id))n.weights[e.id]=randn(rng)*scale;
      }
    }
    this.ensureParameters();
  }
  ensureParameters(){
    for(const n of this.nodes){
      if(n.type==="neuron"||n.type==="output"){
        if(!Number.isFinite(n.bias))n.bias=0;
        n.weights=n.weights||{};
        const valid=new Set(this.incoming(n.id).map(e=>e.id));
        for(const k of Object.keys(n.weights))if(!valid.has(k))delete n.weights[k];
        for(const e of this.incoming(n.id)){
          if(!Number.isFinite(n.weights[e.id])){
            const rng=seeded(hashCode(n.id+"|"+e.id));n.weights[e.id]=randn(rng)*0.55;
          }
        }
      }
    }
  }
  topological(){
    const indeg=new Map(this.nodes.map(n=>[n.id,0]));
    for(const e of this.edges)if(indeg.has(e.to)&&indeg.has(e.from))indeg.set(e.to,indeg.get(e.to)+1);
    const q=this.nodes.filter(n=>indeg.get(n.id)===0).map(n=>n.id),order=[];
    while(q.length){
      const id=q.shift();order.push(id);
      for(const e of this.outgoing(id)){indeg.set(e.to,indeg.get(e.to)-1);if(indeg.get(e.to)===0)q.push(e.to)}
    }
    return order.length===this.nodes.length?order:null;
  }
  validate(){
    const topo=this.topological();if(!topo)return{ok:false,reason:"cycle",message:"检测到回路：当前网络不是前馈 DAG。"};
    const reachable=new Set(["x1","x2"]);
    for(const id of topo){
      const n=this.node(id),ins=this.incoming(id);
      if(n.type==="input")continue;
      if(n.type==="op"){
        const need=(n.op==="square"||n.op==="abs")?1:2;
        if(ins.length<need)return{ok:false,reason:"missing-input",node:id,message:`${n.label||n.op} 缺少输入。`};
      }
      if((n.type==="neuron"||n.type==="output")&&ins.length<1)return{ok:false,reason:"missing-input",node:id,message:`${n.label||n.type} 尚未连接任何特征。`};
      if(ins.some(e=>reachable.has(e.from)))reachable.add(id);
    }
    if(!reachable.has("out"))return{ok:false,reason:"disconnected",message:"输出节点没有连到输入数据。"};
    return{ok:true,order:topo};
  }
  forward(sample){
    this.ensureParameters();const valid=this.validate();if(!valid.ok)return{ok:false,error:valid.message,reason:valid.reason};
    const values={},zs={},edgeValues={},order=valid.order;
    for(const id of order){
      const n=this.node(id),ins=this.incoming(id),xs=ins.map(e=>values[e.from]);
      if(n.type==="input"){values[id]=Number(sample[n.key]);continue}
      if(n.type==="op"){
        if(n.op==="add")values[id]=xs.reduce((a,b)=>a+b,0);
        else if(n.op==="mul")values[id]=xs.reduce((a,b)=>a*b,1);
        else if(n.op==="square")values[id]=xs[0]*xs[0];
        else if(n.op==="abs")values[id]=Math.abs(xs[0]);
        else values[id]=xs[0]??0;
      }else if(n.type==="neuron"||n.type==="output"){
        let z=n.bias||0;
        for(const e of ins){edgeValues[e.id]=values[e.from];z+=(n.weights[e.id]||0)*values[e.from]}
        zs[id]=z;
        if(n.type==="output")values[id]=sigmoid(z);
        else values[id]=n.muted?0:act(n.activation||"tanh",z);
      }
      if(!Number.isFinite(values[id]))return{ok:false,error:"数值异常：节点 "+id+" 产生 NaN / Inf。",reason:"numeric"};
    }
    return{ok:true,p:values.out,values,zs,edgeValues,order};
  }
  gradients(sample){
    const f=this.forward(sample);if(!f.ok)return{...f,loss:Infinity};
    const y=Number(sample.label),grads={},biasGrads={},weightGrads={};for(const n of this.nodes)grads[n.id]=0;
    grads.out=0;
    const rev=[...f.order].reverse();
    for(const id of rev){
      const n=this.node(id),ins=this.incoming(id);
      if(n.type==="output"){
        const dz=f.p-y;grads[id]=dz;
        biasGrads[id]=(biasGrads[id]||0)+dz;
        for(const e of ins){weightGrads[e.id]=(weightGrads[e.id]||0)+dz*f.values[e.from];grads[e.from]+=(n.weights[e.id]||0)*dz}
      }else if(n.type==="neuron"){
        const upstream=grads[id];const h=f.values[id],z=f.zs[id];const dz=n.muted?0:upstream*dact(n.activation||"tanh",z,h);
        grads[id]=dz;biasGrads[id]=(biasGrads[id]||0)+dz;
        for(const e of ins){weightGrads[e.id]=(weightGrads[e.id]||0)+dz*f.values[e.from];grads[e.from]+=(n.weights[e.id]||0)*dz}
      }else if(n.type==="op"){
        const up=grads[id],xs=ins.map(e=>f.values[e.from]);
        if(n.op==="add"){for(const e of ins)grads[e.from]+=up}
        else if(n.op==="mul"){
          for(let i=0;i<ins.length;i++){let prod=1;for(let j=0;j<xs.length;j++)if(i!==j)prod*=xs[j];grads[ins[i].from]+=up*prod}
        }else if(n.op==="square"){grads[ins[0].from]+=up*2*xs[0]}
        else if(n.op==="abs"){grads[ins[0].from]+=up*(xs[0]>0?1:xs[0]<0?-1:0)}
        else if(ins[0])grads[ins[0].from]+=up;
      }
    }
    const loss=bce(f.p,y);
    return{...f,loss,grads,biasGrads,weightGrads};
  }
  trainEpoch(data,lr=0.08){
    this.ensureParameters();const aggB={},aggW={},nodeAbs={},nodeGrad={},reluDead={},reluCount={};let loss=0,correct=0,count=0;
    for(const s of data){
      const g=this.gradients(s);if(!g.ok)return{ok:false,error:g.error};
      loss+=g.loss;correct+=(g.p>=.5)==s.label;count++;
      for(const [k,v] of Object.entries(g.biasGrads))aggB[k]=(aggB[k]||0)+v;
      for(const [k,v] of Object.entries(g.weightGrads))aggW[k]=(aggW[k]||0)+v;
      for(const n of this.nodes){
        const v=g.values[n.id];if(Number.isFinite(v))nodeAbs[n.id]=(nodeAbs[n.id]||0)+Math.abs(v);
        if(Number.isFinite(g.grads[n.id]))nodeGrad[n.id]=(nodeGrad[n.id]||0)+Math.abs(g.grads[n.id]);
        if(n.type==="neuron"&&n.activation==="relu"){reluCount[n.id]=(reluCount[n.id]||0)+1;if(v===0)reluDead[n.id]=(reluDead[n.id]||0)+1}
      }
    }
    if(!count)return{ok:false,error:"训练集为空"};
    for(const n of this.nodes){
      if((n.type==="neuron"||n.type==="output")&&!n.frozen){
        n.bias-=lr*(aggB[n.id]||0)/count;
        for(const e of this.incoming(n.id))n.weights[e.id]-=lr*(aggW[e.id]||0)/count;
      }
    }
    if(!isFiniteDeep(this.toJSON()))return{ok:false,error:"参数产生 NaN / Inf"};
    const diagnostics={};
    for(const n of this.nodes)diagnostics[n.id]={
      activation:(nodeAbs[n.id]||0)/count,
      gradient:(nodeGrad[n.id]||0)/count,
      deadFraction:reluCount[n.id]?(reluDead[n.id]||0)/reluCount[n.id]:0
    };
    return{ok:true,loss:loss/count,acc:correct/count,diagnostics};
  }
  train(data,epochs=1,lr=.08){
    let last;const history=[];
    for(let i=0;i<epochs;i++){last=this.trainEpoch(data,lr);if(!last.ok)return last;if(i===0||i===epochs-1||i%5===0)history.push(last.loss)}
    return{...last,history};
  }
  metrics(data){
    let loss=0,correct=0,n=0;const errors=[];
    for(const s of data){const f=this.forward(s);if(!f.ok)return{ok:false,error:f.error,loss:Infinity,acc:0,errors:[]};const l=bce(f.p,s.label);loss+=l;const good=(f.p>=.5)==s.label;correct+=good;n++;if(!good)errors.push({...s,p:f.p,loss:l})}
    return{ok:true,loss:loss/n,acc:correct/n,errors};
  }
  diagnostics(data){
    const ds=data.slice(0,Math.min(64,data.length));const node={};for(const n of this.nodes)node[n.id]={activation:0,gradient:0,dead:0,count:0};
    for(const s of ds){const g=this.gradients(s);if(!g.ok)continue;for(const n of this.nodes){node[n.id].activation+=Math.abs(g.values[n.id]||0);node[n.id].gradient+=Math.abs(g.grads[n.id]||0);node[n.id].count++;if(n.type==="neuron"&&n.activation==="relu"&&g.values[n.id]===0)node[n.id].dead++}}
    for(const x of Object.values(node)){x.activation/=x.count||1;x.gradient/=x.count||1;x.deadFraction=x.dead/(x.count||1)}
    return node;
  }
  structure(){
    const c={neurons:0,relu:0,tanh:0,sigmoid:0,mul:0,square:0,abs:0,add:0,edges:this.edges.length,params:0};
    for(const n of this.nodes){
      if(n.type==="neuron"){c.neurons++;c[n.activation||"tanh"]=(c[n.activation||"tanh"]||0)+1;c.params+=1+this.incoming(n.id).length}
      if(n.type==="output")c.params+=1+this.incoming(n.id).length;
      if(n.type==="op")c[n.op]=(c[n.op]||0)+1;
    }
    return c;
  }
}
