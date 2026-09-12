// Controlled slow image loads: no network, wall-clock sleeps, or policy inference.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname,
  '../../src/dodge_native_game/variants/cnn_image_ddqn/explain.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1]
  .replace('    initialiseLayout();', '')
  .replace('loadEpisodesThenTrace().catch((error) => showError(error.message));', '');
const nodes = new Map(), timers = new Map(), images = new Map();
let timerId = 0;
function node() {
  const context = new Proxy({}, {get: () => () => {}});
  return {style: {}, dataset: {}, children: [], hidden: false, value: '',
    classList: {add() {}, remove() {}, toggle() {}},
    addEventListener() {}, setAttribute() {}, removeAttribute() {},
    append() {}, appendChild() {}, replaceChildren() {}, getContext: () => context};
}
const document = {getElementById(id) {
  if (!nodes.has(id)) nodes.set(id, node());
  return nodes.get(id);
}, createElement: node, querySelectorAll: () => []};
const context = vm.createContext({assert, document, URLSearchParams, AbortController,
  console, Uint8Array, Image: class {
    set src(url) { this.url = url; if (url) images.set(url, this); }
  }, window: {location: {search: ''}, addEventListener() {},
    setTimeout(callback, delay) {timers.set(++timerId, {callback, delay}); return timerId;},
    clearTimeout(id) {timers.delete(id);}}, timers, images});
vm.runInContext(script, context);
const run = (source) => vm.runInContext(source, context);
const flush = async () => { await Promise.resolve(); await Promise.resolve(); };
function load(url, failed = false) {
  const image = images.get(url);
  assert.ok(image, `requested ${url}`);
  const callback = failed ? image.onerror : image.onload;
  assert.equal(typeof callback, 'function');
  callback();
}
function tick() {
  const timer = [...timers].find(([, value]) => value.delay < 1000);
  assert.ok(timer, 'playback timer scheduled');
  timers.delete(timer[0]);
  return timer[1].callback();
}
async function main() {
  run(`
    state.trace = {step_frames:4,frames:Array.from({length:40},(_,index)=>
      ({index,native_frame:index*4,game_url:'/game/'+index}))};
    requestExplanation = () => {};
    renderFrame = () => renderGameImage(currentFrame());
    renderFrame();
  `);
  load('/game/0');
  await flush();
  assert.equal(nodes.get('game-image').dataset.frameIndex, '0');
  run('setPlaying(true)');
  const pending = tick(); // next image still pending after the nominal 67ms tick
  await flush();
  assert.equal(run('state.index'), 0, 'cursor waits for native image');
  assert.equal(nodes.get('game-image').hidden, false, 'old matching frame stays visible');
  assert.equal(nodes.get('game-image').dataset.frameIndex, '0');
  assert.equal(images.get('/game/1').url, '/game/1', 'pending image not cancelled');
  load('/game/1');
  await pending;
  assert.equal(run('state.index'), 1);
  assert.equal(nodes.get('game-image').dataset.frameIndex, '1');
  assert.equal(nodes.get('game-image').hidden, false);

  const paused = tick();
  run('setPlaying(false)');
  load('/game/2');
  await paused;
  assert.equal(run('state.index'), 1, 'pause invalidates pending playback');

  run('setPlaying(true)');
  await tick(); // frame 2 now cached
  const seeking = tick();
  run('setIndex(8)');
  load('/game/3');
  await seeking;
  assert.equal(run('state.index'), 8, 'seek invalidates pending playback');
  load('/game/8');
  await flush();
  assert.equal(nodes.get('game-image').dataset.frameIndex, '8');

  run('setPlaying(true)');
  const failed = tick();
  load('/game/9', true);
  await failed;
  assert.equal(run('state.index'), 8, 'failed load must not advance');
  assert.equal(run('state.playing'), false, 'failed load pauses for retry');
  assert.equal(nodes.get('game-image').hidden, false);
  run('setPlaying(true)');
  const retry = tick();
  load('/game/9');
  await retry;
  assert.equal(run('state.index'), 9, 'failed image can be retried');

  const replaced = tick();
  run('setPlaying(false); state.trace = {...state.trace}; state.index = 0;');
  load('/game/10');
  await replaced;
  assert.equal(run('state.index'), 0, 'old trace cannot advance new episode');
  run('for (let i=0;i<100;i++) nativeImageEntry("/cache/"+i);');
  assert.equal(run('state.gameCache.size'), 24, 'decoded/pending cache is bounded');
  assert.equal(images.get('/cache/0').url, '', 'evicted pending image cancelled');
  console.log('PASS slow native playback, alignment, pause, seek, episode replacement, retry, cache bound');
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
