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
function node() {
  const canvasContext=new Proxy({createImageData(w,h){return {data:new Uint8ClampedArray(w*h*4)};},
    measureText(){return {width:10};}}, {get(target,key){return key in target?target[key]:()=>{};}});
  return {style:{}, children:[], textContent:'', hidden:false, clientWidth:800,
    parentElement:{clientWidth:800}, dataset:{}, value:'value',
    classList:{add(){},remove(){},toggle(){}},
    addEventListener(){},setAttribute(){},removeAttribute(){},replaceChildren(){this.children=[];},
    append(...items){this.children.push(...items);},
    appendChild(item){this.children.push(item);},getContext(){return canvasContext;}};
}
const document = {getElementById(id){if(!nodes.has(id))nodes.set(id,node());return nodes.get(id);},
  createElement:node,querySelectorAll(){return [];}};
let nextTimer=0;
const timers = new Map();
const context = vm.createContext({assert,document,URLSearchParams,AbortController,
  Uint8Array,console,Image:class {set src(value){this._src=value;}},
  window:{location:{search:''},addEventListener(){},
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
