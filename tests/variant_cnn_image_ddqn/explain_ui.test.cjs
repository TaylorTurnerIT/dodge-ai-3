// Pure UI contracts with a small DOM stub; not a visual browser test.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname,
  '../../src/dodge_native_game/variants/cnn_image_ddqn/explain.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1]
  .replace('loadEpisodesThenTrace().catch((error) => showError(error.message));', '');
const nodes = new Map();
const observers = new Map();
function node() {
  const canvasContext=new Proxy({createImageData(w,h){return {data:new Uint8ClampedArray(w*h*4)};},
    measureText(){return {width:10};}}, {get(target,key){return key in target?target[key]:()=>{};}});
  const classes = new Set();
  const listeners = new Map();
  const notify = (target) => (observers.get(target) || []).forEach((callback) => callback([{type:'childList',target}]));
  return {style:{}, children:[], textContent:'', hidden:false, clientWidth:800,
    parentElement:{clientWidth:800}, dataset:{}, value:'value', listeners,
    classList:{add(...names){names.forEach((name)=>classes.add(name));},
      remove(...names){names.forEach((name)=>classes.delete(name));},
      toggle(name, force){const active=force === undefined ? !classes.has(name) : Boolean(force);if(active)classes.add(name);else classes.delete(name);return active;},
      contains(name){return classes.has(name);}},
    addEventListener(type,callback){listeners.set(type,callback);},
    setAttribute(name,value){this[name]=String(value);},removeAttribute(name){delete this[name];},
    replaceChildren(...items){this.children=items;notify(this);},
    append(...items){this.children.push(...items);notify(this);},
    appendChild(item){this.children.push(item);notify(this);return item;},
    focus(){},getContext(){return canvasContext;}};
}
const shell = node();
const inspectorTabs = ['heatmaps','conv','features','help'].map((name) => {
  const tab = node(); tab.dataset.inspectorTab = name; return tab;
});
const inspectorPanels = ['heatmaps','conv','features','help'].map((name) => {
  const panel = node(); panel.dataset.inspectorPanel = name; return panel;
});
const mobileTabs = ['replay','inspector'].map((name) => {
  const tab = node(); tab.dataset.mobileView = name; return tab;
});
['inspector-heatmaps-tab','inspector-conv-tab','inspector-features-tab','inspector-help-tab']
  .forEach((id,index) => nodes.set(id, inspectorTabs[index]));
['inspector-panel-heatmaps','inspector-panel-conv','inspector-panel-features','inspector-panel-help']
  .forEach((id,index) => nodes.set(id, inspectorPanels[index]));
nodes.set('mobile-replay-tab', mobileTabs[0]);
nodes.set('mobile-inspector-tab', mobileTabs[1]);
const document = {
  getElementById(id){if(!nodes.has(id))nodes.set(id,node());return nodes.get(id);},
  createElement:node,
  querySelector(selector){return selector === '.viewport-shell' ? shell : null;},
  querySelectorAll(selector){
    if(selector === '[data-inspector-tab]') return inspectorTabs;
    if(selector === '[data-inspector-panel]') return inspectorPanels;
    if(selector === '[data-mobile-view]') return mobileTabs;
    return [];
  }
};
class MutationObserver {
  constructor(callback){this.callback=callback;}
  observe(target){const callbacks=observers.get(target) || [];callbacks.push(this.callback);observers.set(target,callbacks);}
}
let nextTimer=0;
const timers = new Map();
const context = vm.createContext({assert,document,URLSearchParams,AbortController,
  Uint8Array,console,MutationObserver,Image:class {set src(value){this._src=value;}},
  window:{location:{search:''},innerWidth:1280,requestAnimationFrame(callback){callback();},addEventListener(){},
    atob:(value)=>Buffer.from(value,'base64').toString('binary'),
    setTimeout(callback,delay){timers.set(++nextTimer,{callback,delay});return nextTimer;},
    clearTimeout(id){timers.delete(id);}}, timers,
  packed:Buffer.alloc(84*84*4,128).toString('base64')});
vm.runInContext(script,context);
vm.runInContext(`
  assert.equal(finiteNumber(null),null);
  assert.equal(finiteNumber(undefined),null);
  assert.equal(finiteNumber(''),null);
  assert.equal(finiteNumber(0),0);
  assert.equal(actionLabel(1), 'Left');
  assert.equal(actionLabel(3), 'Up');
  assert.equal(actionLabel(6), 'Up-right');
  runMetrics = {run_id:'rgb',rows:[{step:1000,loss:2},{step:2000,loss:1}],source_rows:2,gate_reasons:[]};
  metricPlot('metric-loss','Training loss','loss','Recorded loss');
  assert.equal(document.getElementById('metric-loss').children[0].textContent, 'Training loss');
  assert.ok(document.getElementById('metric-loss').children[1].innerHTML.includes('<svg'));
  metricPlot('metric-gradient','Gradient','pre_clip_grad_norm','Sparse');
  assert.equal(document.getElementById('metric-gradient').children[1].textContent, 'Not recorded in this run.');
  const trace=normaliseTrace({step_frames:4,frames:[
    {index:0,action:0,q:[9,8,7,6,5,4,3,2,1],value:null,input_base64:packed,
     input_shape:[4,84,84],native_frame:0,reward:4,events:{death:false,collision:true}},
    {index:1,action:0,q:[9,8,7,6,5,4,3,2,1],value:null,native_frame:4,reward:4}
  ]});
  assert.equal(trace.frames[0].value,null);
  const decoded=decodePackedInput(trace.frames[0]);
  assert.equal(decoded.count,4);
  assert.equal(decoded.bytes.length,4*84*84);
  assert.equal(decoded.bytes[0],128);
  assert.equal(decoded.bytes[decoded.bytes.length-1],128);
  assert.ok(decodePackedInput({input_shape:[4,84,84],input_base64:'AA=='}).error);
  const positive=divergingColor(1,1),negative=divergingColor(-1,1),zero=divergingColor(0,1);
  assert.ok(positive[0]>positive[2]);
  assert.ok(negative[2]>negative[0]);
  assert.ok(Math.max(...zero.slice(0,3))<30);
  state.trace=trace;
  state.index=1;
  assert.equal(inputNativeFrame(0,4),0,'reset-padded input uses initial frame');
  assert.equal(inputNativeFrame(3,4),4,'newest input matches current native frame');
  state.index=0;
  renderFrameStats(trace.frames[0]);
  assert.ok(document.getElementById('stat-event').textContent.includes('collision'));
  assert.ok(!document.getElementById('stat-event').textContent.includes('death'));
  renderFrame();
  state.explain={chosen:0,alternative:1,value_map:[[0,1],[-1,0]],
    decision_map:[[0,-1],[1,0]],conv_maps:Array.from({length:64},()=>Array.from({length:7},()=>Array(7).fill(1))),
    contributions:Array(512).fill(.001),bias_difference:0,
    ablation:{q:trace.frames[0].q,gap_delta:.01},grid_positions:[[6,6],[18,6],[6,18],[18,18]]};
  renderExplanation();
  assert.equal(document.getElementById('conv-grid').children.length,64);
  assert.equal(document.getElementById('contribution-list').children.length,512);
  assert.equal(document.getElementById('conv-page-label').textContent,'Channels 1–16 / 64');
  assert.equal(document.getElementById('contribution-page-label').textContent,'Features 1–16 / 512');
  assert.equal(document.getElementById('conv-pagination').hidden,false);
  assert.equal(document.getElementById('contribution-pagination').hidden,false);
  document.getElementById('conv-page-next').listeners.get('click')();
  assert.equal(document.getElementById('conv-page-label').textContent,'Channels 17–32 / 64');
  assert.equal(document.getElementById('conv-grid').children[0].hidden,true);
  assert.equal(document.getElementById('conv-grid').children[16].hidden,false);
  setInspectorTab('conv');
  assert.equal(document.getElementById('inspector-panel-conv').hidden,false);
  assert.equal(document.getElementById('inspector-panel-heatmaps').hidden,true);
  assert.equal(document.querySelectorAll('[data-inspector-tab]')[1]['aria-selected'],'true');
  setInspectorTab('features');
  assert.equal(document.getElementById('inspector-panel-features').hidden,false);
  assert.equal(document.getElementById('inspector-panel-conv').hidden,true);
  let explanations=0;
  requestExplanation=()=>{explanations++;};
  renderFrame=()=>{};
  setPlaying(true);
  assert.equal(timers.size,1);
  const scheduled=[...timers.values()][0];
  assert.ok(Math.abs(scheduled.delay - 1000*4/60)<1);
  timers.clear();scheduled.callback();
  assert.equal(explanations,0,'playback must not request heatmaps');
  setPlaying(false);
  assert.equal(timers.size,0,'pause cancels pending timer');
`,context);
console.log('PASS UI null handling, packed inputs, signed colors, events, playback cadence/pause');
