"""Small Tailscale-facing browser server for native Dodge replays."""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import threading
from collections import OrderedDict
from collections.abc import Iterable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final
from urllib.parse import unquote, urlsplit

from .native_replay import (
    DEFAULT_REPLAY_STEPS,
    MAX_REPLAY_STEPS,
    NativeReplay,
    generate_native_replay,
)
from .run_replay import EPISODE_KEYS, generate_run_comparison, run_context

DEFAULT_REPLAY_HOST: Final = "127.0.0.1"
DEFAULT_REPLAY_PORT: Final = 8890
MAX_REPLAYS: Final = 8


def tailscale_ipv4() -> str | None:
    """Return the local Tailscale IPv4 address when the CLI is available."""

    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in result.stdout.splitlines():
        value = line.strip()
        if value and "." in value:
            return value
    return None


class ReplayStore:
    """Bounded in-memory store; no live trainer state is served."""

    def __init__(self, *, max_replays: int = MAX_REPLAYS) -> None:
        if isinstance(max_replays, bool) or max_replays < 1:
            raise ValueError("max_replays must be positive")
        self._max_replays = int(max_replays)
        self._items: OrderedDict[str, NativeReplay] = OrderedDict()
        self._lock = threading.RLock()

    def add(self, replay: NativeReplay) -> str:
        if not 1 <= replay.frame_count <= MAX_REPLAY_STEPS + 1:
            raise ValueError("replay frame count exceeds the bounded store limit")
        token = secrets.token_urlsafe(9)
        with self._lock:
            self._items[token] = replay
            while len(self._items) > self._max_replays:
                self._items.popitem(last=False)
        return token

    def get(self, token: str) -> NativeReplay | None:
        with self._lock:
            return self._items.get(token)

    def items(self) -> Iterable[tuple[str, NativeReplay]]:
        with self._lock:
            return tuple(
                sorted(
                    self._items.items(),
                    key=lambda item: _checkpoint_mtime(item[1]),
                    reverse=True,
                )
            )


def _page_html(token: str) -> bytes:
    token_json = json.dumps(token)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dodge Native Replay</title>
  <style>
    :root {{ color-scheme: dark; --bg:#171615; --panel:#0e0f0f; --border:#393733;
      --text:#e0dfdb; --muted:#8b8982; --cyan:#15c2eb; --green:#71be59;
      --yellow:#f4bd2f; --red:#ff4c4c; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; min-height:100vh; background:var(--bg); color:var(--text);
      font:14px/1.4 system-ui, sans-serif; }}
    body::before {{ content:""; position:fixed; inset:0; pointer-events:none; opacity:.35;
      background-image:radial-gradient(#312f2c 1px, transparent 1px); background-size:20px 20px; }}
    main {{ position:relative; max-width:1120px; margin:0 auto; padding:22px; }}
    header {{ display:flex; justify-content:space-between; align-items:end; gap:18px;
      margin-bottom:18px; }}
    h1 {{ margin:0; font-size:22px; font-weight:650; }}
    h1 span {{ color:var(--cyan); }}
    .subtitle {{ color:var(--muted); margin-top:4px; }}
    .badge {{ border:1px solid var(--border); border-radius:999px; padding:6px 11px;
      color:var(--green); white-space:nowrap; }}
    .layout {{ display:grid; grid-template-columns:minmax(320px, 700px) minmax(250px,1fr);
      gap:18px; align-items:start; }}
    .card {{ background:var(--panel); border:1px solid var(--border); border-radius:9px; padding:16px; }}
    .screen {{ display:flex; justify-content:center; align-items:center; min-height:420px;
      background:#000; border-radius:6px; border:1px solid #292826; overflow:hidden; }}
    canvas {{ width:min(84vw, 640px); height:min(84vw, 640px); max-width:100%; image-rendering:pixelated;
      image-rendering:crisp-edges; background:#000; }}
    .view-toggle {{ display:flex; gap:8px; margin-top:10px; }}
    .view-toggle button.active {{ border-color:var(--cyan); color:var(--cyan); }}
    .controls {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-top:14px; }}
    button {{ border:1px solid var(--border); border-radius:5px; padding:8px 12px; color:var(--text);
      background:#242321; cursor:pointer; }}
    button:hover {{ border-color:var(--cyan); }}
    button.primary {{ background:#16586a; border-color:#1d7890; }}
    input[type=range] {{ flex:1; min-width:180px; accent-color:var(--cyan); }}
    .frame-label {{ min-width:72px; text-align:right; color:var(--muted); }}
    .stats {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
    .stat {{ border:1px solid #2a2926; border-radius:6px; padding:10px; }}
    .stat label {{ display:block; color:var(--muted); font-size:12px; margin-bottom:3px; }}
    .stat strong {{ font-size:17px; font-weight:600; }}
    .meta {{ margin-top:14px; color:var(--muted); overflow-wrap:anywhere; }}
    .meta div {{ margin-top:7px; }}
    .meta code {{ color:var(--text); }}
    .legend {{ margin-top:16px; color:var(--muted); }}
    .legend b {{ color:var(--text); font-weight:550; }}
    .loading {{ color:var(--muted); padding:32px; text-align:center; }}
    @media (max-width:760px) {{ main {{ padding:12px; }} .layout {{ grid-template-columns:1fr; }}
      .screen {{ min-height:0; }} header {{ align-items:start; flex-direction:column; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <div><h1>Dodge <span>Native Replay</span></h1>
        <div class="subtitle">Game-view playback from a saved DDQN checkpoint</div></div>
      <div class="badge">Rust lane · game view</div>
    </header>
    <div id="loading" class="card loading">Loading replay…</div>
    <section id="content" class="layout" hidden>
      <div class="card">
        <div class="screen"><canvas id="screen" width="128" height="128"></canvas></div>
        <div class="view-toggle">
          <button id="view-game" class="active">Game view</button>
          <button id="view-collision">Collision view</button>
        </div>
        <div class="controls">
          <button id="restart">Restart</button><button id="play" class="primary">Play</button>
          <button id="back">Step −</button><button id="forward">Step +</button>
          <input id="timeline" type="range" min="0" max="0" value="0" step="1" aria-label="Replay frame">
          <span id="frame-label" class="frame-label">0 / 0</span>
        </div>
      </div>
      <aside class="card">
        <div class="stats">
          <div class="stat"><label>Native frame</label><strong id="native-frame">-</strong></div>
          <div class="stat"><label>Replay step</label><strong id="replay-step">-</strong></div>
          <div class="stat"><label>Action</label><strong id="action">-</strong></div>
          <div class="stat"><label>Reward</label><strong id="reward">-</strong></div>
          <div class="stat"><label>Terminal</label><strong id="terminal">-</strong></div>
          <div class="stat"><label>Decision interval</label><strong id="step-frames">-</strong></div>
        </div>
        <div class="meta"><div>Checkpoint: <code id="checkpoint">-</code></div>
          <div>Seed: <code id="seed">-</code></div><div>Generated: <code id="created">-</code></div>
          <div id="run-context">Run best: <code>-</code></div></div>
        <div class="legend"><b>Game view</b><br>the native 128×128 framebuffer in PICO-8 colors.
          Collision view shows the 84×84 model input: black = empty · gray = hazard area ·
          bright = hostile · white = player.</div>
      </aside>
    </section>
  </main>
  <script>
    const TOKEN = {token_json};
    const $ = (id) => document.getElementById(id);
    const state = {{ meta:null, index:0, playing:false, loadSerial:0, view:"game" }};
    const canvas = $("screen"), context = canvas.getContext("2d");
    context.imageSmoothingEnabled = false;
    function frameUrl(frame) {{ return state.view === "collision" ? frame.collision_url : frame.frame_url; }}
    function setView(view) {{
      state.view = view;
      $("view-game").classList.toggle("active", view === "game");
      $("view-collision").classList.toggle("active", view === "collision");
      canvas.width = view === "collision" ? 84 : 128;
      canvas.height = view === "collision" ? 84 : 128;
      showFrame(state.index);
    }}
    function clamp(value, low, high) {{ return Math.max(low, Math.min(high, value)); }}
    function setPlaying(value) {{ state.playing=value; $("play").textContent=value ? "Pause" : "Play"; }}
    function frameData() {{ return state.meta.frames[state.index]; }}
    async function showFrame(index) {{
      if (!state.meta) return;
      state.index=clamp(index, 0, state.meta.frames.length-1);
      const frame=frameData(), serial=++state.loadSerial;
      $("timeline").value=state.index;
      $("frame-label").textContent=`${{state.index}} / ${{state.meta.frames.length-1}}`;
      $("native-frame").textContent=frame.native_frame.toLocaleString();
      $("replay-step").textContent=state.index.toLocaleString();
      $("action").textContent=frame.action_name || "reset";
      $("reward").textContent=Number(frame.reward).toFixed(2);
      $("terminal").textContent=frame.done ? "yes" : "no";
      const size = state.view === "collision" ? 84 : 128;
      const image=new Image(); image.src=frameUrl(frame);
      image.onload=()=>{{ if(serial !== state.loadSerial) return; context.clearRect(0,0,size,size); context.drawImage(image,0,0,size,size); }};
      if(state.playing && state.index < state.meta.frames.length-1) setTimeout(()=>showFrame(state.index+1), 1000/15);
      else if(state.playing) setPlaying(false);
    }}
    async function load() {{
      try {{
        const response=await fetch(`/api/replay/${{TOKEN}}`); if(!response.ok) throw new Error("Replay is unavailable");
        state.meta=await response.json(); $("loading").hidden=true; $("content").hidden=false;
        $("timeline").max=state.meta.frames.length-1; $("checkpoint").textContent=state.meta.checkpoint;
        $("seed").textContent=state.meta.seed; $("created").textContent=state.meta.created_at;
        $("step-frames").textContent=state.meta.step_frames; await showFrame(0);
        if (state.meta.run_id) {{
          fetch(`/api/run/${{encodeURIComponent(state.meta.run_id)}}/curves`).then((r) => r.json()).then((curves) => {{
            const best = curves.best_eval_episode;
            $("run-context").innerHTML = best
              ? `Run best: <code>${{curves.run_id}} seed ${{best.seed}} (eval ${{Number(best.eval_reward).toFixed(1)}})</code> · `
                + `training max <code>${{curves.train_max.reward}} @ step ${{curves.train_max.step}}</code> (exploration-era)`
              : `Run best: <code>unavailable</code>`;
          }}).catch(() => {{ $("run-context").textContent = "Run best: unavailable"; }});
        }} else {{ $("run-context").textContent = "Run best: ad-hoc replay (no run attached)"; }}
      }} catch(error) {{ $("loading").textContent=error.message; }}
    }}
    $("play").onclick=()=>state.meta && (state.playing ? setPlaying(false) : (setPlaying(true), showFrame(state.index)));
    $("restart").onclick=()=>{{ setPlaying(false); showFrame(0); }};
    $("back").onclick=()=>{{ setPlaying(false); showFrame(state.index-1); }};
    $("forward").onclick=()=>{{ setPlaying(false); showFrame(state.index+1); }};
    $("timeline").oninput=(event)=>{{ setPlaying(false); showFrame(Number(event.target.value)); }};
    $("view-game").onclick=()=>setView("game");
    $("view-collision").onclick=()=>setView("collision");
    load();
  </script>
</body>
</html>"""
    return html.encode("utf-8")


def _index_html(items: Iterable[tuple[str, NativeReplay]]) -> bytes:
    rows = []
    for token, replay in items:
        label = replay.label or "-"
        run = replay.run_id or "-"
        source = replay.source or "-"
        eval_reward = (
            f"{replay.eval_reward:.1f}"
            if replay.eval_reward is not None
            else "-"
        )
        rows.append(
            f'<li><a href="/replay/{token}">{replay.checkpoint.name}</a> '
            f'run <b>{run}</b> · {label} · {source} '
            f'(eval {eval_reward}, replay {replay.total_reward:.1f}, '
            f'{replay.frame_count} frames, seed {replay.seed}) '
            f'<a href="/compare/{token}">compare</a></li>'
        )
    body = "".join(rows) or "<li>No replay has been generated yet.</li>"
    return (
        "<!doctype html><meta charset='utf-8'><title>Dodge Native Replays</title>"
        "<style>body{background:#171615;color:#e0dfdb;font:16px system-ui;padding:28px}"
        "a{color:#15c2eb}b{color:#fff}</style><h1>Dodge Native Replays</h1>"
        "<p>Generate a run comparison with "
        "<code>dodge-cnn-image-replay-server --run-id &lt;run&gt;</code> "
        "to get best / median / worst games side by side.</p><ul>"
        + body
        + "</ul>"
    ).encode("utf-8")


def _compare_html(tokens: list[str], run_ids: list[str] | None = None) -> bytes:
    tokens_json = json.dumps(tokens)
    runs_json = json.dumps(run_ids or [])
    cards = "".join(
        f'<div class="card" id="panel-{index}">'
        f'<h2>Loading…</h2>'
        f'<div class="screen"><canvas width="128" height="128"></canvas></div>'
        '<div class="controls"><button data-act="restart">Restart</button>'
        '<button data-act="play" class="primary">Play</button>'
        '<button data-act="back">−</button><button data-act="forward">+</button>'
        '<input type="range" min="0" max="0" value="0" step="1" aria-label="Replay frame">'
        '<span class="frame-label">0 / 0</span></div>'
        '<div class="meta"></div></div>'
        for index in range(len(tokens))
    )
    context_cards = "".join(
        f'<div class="card" id="context-{index}"><h2>Run context loading…</h2></div>'
        for index in range(len(run_ids or []))
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dodge Replay Comparison</title>
  <style>
    :root {{ color-scheme: dark; --bg:#171615; --panel:#0e0f0f; --border:#393733;
      --text:#e0dfdb; --muted:#8b8982; --cyan:#15c2eb; --green:#71be59;
      --yellow:#f4bd2f; --red:#ff4c4c; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.4 system-ui, sans-serif; }}
    main {{ max-width:1500px; margin:0 auto; padding:22px; }}
    h1 {{ margin:0; font-size:22px; }} h1 span {{ color:var(--cyan); }}
    h3 {{ margin:0 0 6px; font-size:14px; }}
    .subtitle {{ color:var(--muted); margin:4px 0 14px; }}
    .global {{ display:flex; gap:8px; margin-bottom:14px; }}
    button {{ border:1px solid var(--border); border-radius:5px; padding:8px 12px;
      color:var(--text); background:#242321; cursor:pointer; }}
    button.primary {{ background:#16586a; border-color:#1d7890; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(340px, 1fr)); gap:16px; }}
    .card {{ background:var(--panel); border:1px solid var(--border); border-radius:9px; padding:14px; }}
    .card h2 {{ margin:0 0 4px; font-size:16px; }}
    .card .sub {{ color:var(--muted); margin-bottom:8px; }}
    .screen {{ display:flex; justify-content:center; background:#000; border-radius:6px;
      border:1px solid #292826; overflow:hidden; }}
    canvas {{ width:100%; max-width:420px; aspect-ratio:1; image-rendering:pixelated; background:#000; }}
    .controls {{ display:flex; align-items:center; gap:6px; margin-top:10px; flex-wrap:wrap; }}
    input[type=range] {{ flex:1; min-width:120px; accent-color:var(--cyan); }}
    .frame-label {{ color:var(--muted); min-width:64px; text-align:right; }}
    .meta {{ color:var(--muted); margin-top:8px; }} .meta code {{ color:var(--text); }}
    svg.plot {{ width:100%; height:auto; display:block; background:#000;
      border:1px solid #292826; border-radius:6px; margin-top:8px; }}
    .caption {{ color:var(--muted); margin-top:6px; }} .caption code {{ color:var(--text); }}
    .section-title {{ margin:22px 0 10px; font-size:17px; }}
  </style>
</head>
<body>
  <main>
    <h1>Dodge <span>Replay Comparison</span></h1>
    <div class="subtitle">Best / median / worst final-eval games from one run, same checkpoint and game settings.</div>
    <div class="global"><button id="play-all" class="primary">Play all</button>
      <button id="pause-all">Pause all</button><button id="restart-all">Restart all</button></div>
    <div class="grid" id="replays">{cards}</div>
    <h2 class="section-title">Run context <span style="color:var(--muted);font-weight:400;">- final results plotted, not the training max</span></h2>
    <div class="grid" id="context">{context_cards}</div>
  </main>
  <script>
    const TOKENS = {tokens_json};
    const RUNS = {runs_json};
    const players = [];
    function makePlayer(panel, token) {{
      const canvas = panel.querySelector("canvas"), ctx = canvas.getContext("2d");
      ctx.imageSmoothingEnabled = false;
      const slider = panel.querySelector("input"), label = panel.querySelector(".frame-label");
      const meta = panel.querySelector(".meta"), title = panel.querySelector("h2");
      const sub = document.createElement("div"); sub.className = "sub"; title.after(sub);
      const player = {{ meta:null, index:0, playing:false, serial:0, canvas, ctx, slider, label, meta, title, sub }};
      async function show(i) {{
        if (!player.meta) return;
        player.index = Math.max(0, Math.min(player.meta.frames.length - 1, i));
        const frame = player.meta.frames[player.index], serial = ++player.serial;
        slider.max = player.meta.frames.length - 1; slider.value = player.index;
        label.textContent = `${{player.index}} / ${{player.meta.frames.length - 1}}`;
        meta.innerHTML = `action <code>${{frame.action_name || "reset"}}</code> · reward `
          + `<code>${{Number(frame.reward).toFixed(2)}}</code> · native <code>${{frame.native_frame}}</code>`
          + (frame.done ? " · <code>done</code>" : "");
        const img = new Image(); img.src = frame.frame_url;
        img.onload = () => {{ if (serial !== player.serial) return;
          ctx.clearRect(0, 0, 128, 128); ctx.drawImage(img, 0, 0, 128, 128); }};
        if (player.playing && player.index < player.meta.frames.length - 1)
          setTimeout(() => show(player.index + 1), 1000 / 15);
        else if (player.playing) setPlaying(false);
      }}
      function setPlaying(v) {{
        player.playing = v;
        panel.querySelector('[data-act="play"]').textContent = v ? "Pause" : "Play";
      }}
      panel.querySelector('[data-act="play"]').onclick = () =>
        player.playing ? setPlaying(false) : (setPlaying(true), show(player.index));
      panel.querySelector('[data-act="restart"]').onclick = () => {{ setPlaying(false); show(0); }};
      panel.querySelector('[data-act="back"]').onclick = () => {{ setPlaying(false); show(player.index - 1); }};
      panel.querySelector('[data-act="forward"]').onclick = () => {{ setPlaying(false); show(player.index + 1); }};
      slider.oninput = (e) => {{ setPlaying(false); show(Number(e.target.value)); }};
      player.show = show; player.setPlaying = setPlaying;
      fetch(`/api/replay/${{token}}`).then((r) => r.json()).then((data) => {{
        player.meta = data;
        const tag = data.label ? data.label.toUpperCase() : "REPLAY";
        title.textContent = `${{tag}} - seed ${{data.seed}}`;
        const evalText = data.eval_reward == null ? "-" : Number(data.eval_reward).toFixed(1);
        sub.textContent = `run ${{data.run_id || data.checkpoint}} · ${{data.source || "seed"}} · `
          + `eval ${{evalText}} · replay ${{Number(data.total_reward).toFixed(1)}} · ${{data.frame_count}} frames`;
        show(0);
      }});
      return player;
    }}
    document.querySelectorAll("#replays .card").forEach((panel, i) => players.push(makePlayer(panel, TOKENS[i])));
    document.getElementById("play-all").onclick = () => players.forEach((p) => p.meta && (p.setPlaying(true), p.show(p.index)));
    document.getElementById("pause-all").onclick = () => players.forEach((p) => p.setPlaying(false));
    document.getElementById("restart-all").onclick = () => players.forEach((p) => {{ p.setPlaying(false); p.show(0); }});
    function lineChart(points, trainMax) {{
      const W = 600, H = 200, pad = 34;
      if (!points.length) return `<svg class="plot" viewBox="0 0 ${{W}} ${{H}}"><text x="12" y="20" fill="#8b8982">no training rows</text></svg>`;
      const xs = points.map((p) => p.step), ys = points.map((p) => p.reward);
      const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
      const y1 = Math.max(...ys, 1);
      const X = (v) => pad + ((v - x0) / (x1 - x0)) * (W - pad - 8);
      const Y = (v) => H - pad + 8 - (Math.max(v, 0) / y1) * (H - pad - 16);
      const line = points.map((p) => `${{X(p.step).toFixed(1)}},${{Y(p.reward).toFixed(1)}}`).join(" ");
      const mx = X(trainMax.step), my = Y(trainMax.reward);
      return `<svg class="plot" viewBox="0 0 ${{W}} ${{H}}">`
        + `<polyline points="${{line}}" fill="none" stroke="#15c2eb" stroke-width="1.5"/>`
        + `<circle cx="${{mx}}" cy="${{my}}" r="4" fill="#f4bd2f"/>`
        + `<text x="${{Math.min(mx + 7, W - 190)}}" y="${{Math.max(my - 7, 12)}}" fill="#f4bd2f" font-size="12">train max ${{trainMax.reward}} @ step ${{trainMax.step}}</text>`
        + `<text x="${{pad}}" y="${{H - 8}}" fill="#8b8982" font-size="11">train step</text></svg>`;
    }}
    function evalStrip(evalData, picks) {{
      const W = 600, H = 150, pad = 34;
      const rows = ["inner", "holdout"].filter((s) => evalData[s] && evalData[s].rewards.length);
      if (!rows.length) return `<svg class="plot" viewBox="0 0 ${{W}} ${{H}}"><text x="12" y="20" fill="#8b8982">no eval table</text></svg>`;
      const all = rows.flatMap((s) => evalData[s].rewards);
      const y1 = Math.max(...all, 1);
      const pickSeeds = {{}};
      Object.entries(picks || {{}}).forEach(([key, ep]) => {{ pickSeeds[ep.seed] = key; }});
      let svg = "";
      rows.forEach((source, r) => {{
        const rewards = evalData[source].rewards, seeds = evalData[source].seeds;
        const cy = 34 + r * 52;
        const mean = rewards.reduce((a, b) => a + b, 0) / rewards.length;
        const my = 20 + ((H - 40) - (mean / y1) * (H - 60));
        svg += `<line x1="${{pad}}" y1="${{my}}" x2="${{W - 8}}" y2="${{my}}" stroke="#393733" stroke-dasharray="4 3"/>`;
        svg += `<text x="4" y="${{cy + 4}}" fill="#8b8982" font-size="11">${{source}}</text>`;
        rewards.forEach((reward, i) => {{
          const cx = pad + (rewards.length < 2 ? 0.5 : i / (rewards.length - 1)) * (W - pad - 30);
          const cyDot = 20 + ((H - 40) - (reward / y1) * (H - 60));
          const pick = pickSeeds[seeds[i]];
          const color = pick === "best" ? "#71be59" : pick === "worst" ? "#ff4c4c" : pick === "median" ? "#f4bd2f" : "#15c2eb";
          svg += `<circle cx="${{cx}}" cy="${{cyDot}}" r="${{pick ? 6 : 4}}" fill="${{color}}"${{pick ? ' stroke="#e0dfdb" stroke-width="1.5"' : ""}}/>`;
          if (pick) svg += `<text x="${{cx + 8}}" y="${{cyDot + 4}}" fill="${{color}}" font-size="11">${{pick}} (seed ${{seeds[i]}})</text>`;
        }});
      }});
      return `<svg class="plot" viewBox="0 0 ${{W}} ${{H}}">${{svg}}</svg>`;
    }}
    RUNS.forEach((runId, i) => {{
      const panel = document.getElementById(`context-${{i}}`);
      fetch(`/api/run/${{encodeURIComponent(runId)}}/curves`).then((r) => {{
        if (!r.ok) throw new Error("unavailable");
        return r.json();
      }}).then((data) => {{
        const best = data.best_eval_episode;
        const bestText = best ? `seed ${{best.seed}} (${{best.source}}, eval ${{best.eval_reward}})` : "-";
        panel.innerHTML = `<h2>Run <code style="color:var(--cyan)">${{data.run_id}}</code></h2>`
          + `<div class="sub">final best ${{bestText}} · training max ${{data.train_max.reward}} @ step ${{data.train_max.step}} (exploration-era, not replayable)</div>`
          + `<h3>Training episode reward</h3>` + lineChart(data.train_curve, data.train_max)
          + `<h3 style="margin-top:10px">Final eval rewards</h3>` + evalStrip(data.eval, data.picks)
          + `<div class="caption">Rings mark the replayed best / median / worst games. Dashed lines are split means.</div>`;
      }}).catch(() => {{ panel.innerHTML = `<h2>${{runId}}</h2><div class="sub">Run context unavailable.</div>`; }});
    }});
  </script>
</body>
</html>"""
    return html.encode("utf-8")


def _checkpoint_mtime(replay: NativeReplay) -> float:
    """Training recency: newest checkpoint file first, unknown last."""

    try:
        return float(replay.checkpoint.stat().st_mtime)
    except OSError:
        return 0.0


class _ReplayHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        store: ReplayStore,
        history_root: Path | None = None,
    ) -> None:
        self.store = store
        self.history_root = history_root
        super().__init__(address, _ReplayRequestHandler)


class _ReplayRequestHandler(BaseHTTPRequestHandler):
    server: _ReplayHTTPServer
    protocol_version = "HTTP/1.0"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler hook
        path = unquote(urlsplit(self.path).path)
        segments = [segment for segment in path.split("/") if segment]
        try:
            if not segments:
                self._send(HTTPStatus.OK, _index_html(self.server.store.items()), "text/html")
                return
            if segments[0] == "replay" and len(segments) == 2:
                self._serve_page(segments[1])
                return
            if segments[0] == "compare" and len(segments) == 2:
                self._serve_compare(segments[1])
                return
            if segments[:2] == ["api", "replays"] and len(segments) == 2:
                self._send_json(
                    [self._metadata(token, replay) for token, replay in self.server.store.items()]
                )
                return
            if segments[:2] == ["api", "replay"] and len(segments) == 3:
                replay = self._get_replay(segments[2])
                self._send_json(self._metadata(segments[2], replay))
                return
            if segments[:2] == ["api", "run"] and len(segments) == 4:
                self._serve_curves(segments[2], segments[3])
                return
            if (
                segments[:2] == ["api", "replay"]
                and len(segments) == 5
                and segments[3] in ("frame", "collision")
            ):
                replay = self._get_replay(segments[2])
                if not segments[4].endswith(".png"):
                    raise KeyError
                index = int(segments[4][:-4])
                if index < 0:
                    raise KeyError
                frame = replay.frames[index]
                body = frame.png if segments[3] == "frame" else frame.collision_png
                self._send(HTTPStatus.OK, body, "image/png", cache=True)
                return
        except (IndexError, KeyError, ValueError):
            pass
        self._send(HTTPStatus.NOT_FOUND, b"not found\n", "text/plain")

    def _serve_page(self, token: str) -> None:
        self._get_replay(token)
        self._send(HTTPStatus.OK, _page_html(token), "text/html")

    def _serve_compare(self, raw: str) -> None:
        tokens = [part for part in raw.split(",") if part]
        if not tokens or len(tokens) > MAX_REPLAYS:
            raise KeyError(raw)
        run_ids: list[str] = []
        for token in tokens:
            replay = self._get_replay(token)
            if replay.run_id and replay.run_id not in run_ids:
                run_ids.append(replay.run_id)
        self._send(HTTPStatus.OK, _compare_html(tokens, run_ids), "text/html")

    def _serve_curves(self, run_id: str, leaf: str) -> None:
        if leaf != "curves":
            raise KeyError(leaf)
        history_root = self.server.history_root
        if history_root is None:
            raise KeyError(run_id)
        try:
            self._send_json(run_context(history_root, run_id))
        except (FileNotFoundError, ValueError):
            raise KeyError(run_id) from None

    def _get_replay(self, token: str) -> NativeReplay:
        replay = self.server.store.get(token)
        if replay is None:
            raise KeyError(token)
        return replay

    @staticmethod
    def _metadata(token: str, replay: NativeReplay) -> dict[str, object]:
        return {
            "token": token,
            "checkpoint": replay.checkpoint.name,
            "seed": replay.seed,
            "step_frames": replay.step_frames,
            "created_at": replay.created_at,
            "frame_count": replay.frame_count,
            "run_id": replay.run_id,
            "label": replay.label,
            "source": replay.source,
            "eval_reward": replay.eval_reward,
            "total_reward": replay.total_reward,
            "survival_frames": replay.survival_frames,
            "terminated": replay.terminated,
            "frames": [
                {
                    "index": frame.index,
                    "native_frame": frame.native_frame,
                    "action": frame.action,
                    "action_name": frame.action_name,
                    "reward": frame.reward,
                    "done": frame.done,
                    "frame_url": f"/api/replay/{token}/frame/{frame.index}.png",
                    "collision_url": f"/api/replay/{token}/collision/{frame.index}.png",
                }
                for frame in replay.frames
            ],
        }

    def _send_json(self, payload: object) -> None:
        self._send(
            HTTPStatus.OK,
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            "application/json",
        )

    def _send(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
        *,
        cache: bool = False,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=31536000" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class ReplayServer:
    """Background HTTP server used only for generated replay snapshots."""

    def __init__(
        self,
        *,
        store: ReplayStore | None = None,
        host: str = DEFAULT_REPLAY_HOST,
        port: int = DEFAULT_REPLAY_PORT,
        public_host: str | None = None,
        history_root: Path | str | None = None,
    ) -> None:
        self.store = store or ReplayStore()
        self.host = host
        self.port = int(port)
        self.public_host = public_host or host
        self.history_root = Path(history_root).expanduser() if history_root else None
        self._server: _ReplayHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = _ReplayHTTPServer(
            (self.host, self.port), self.store, self.history_root
        )
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="dodge-native-replay-server",
            daemon=True,
        )
        self._thread.start()

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("replay server has not started")
        return f"http://{self.public_host}:{self.port}"

    def url_for(self, token: str) -> str:
        return f"{self.base_url}/replay/{token}"

    def url_for_compare(self, tokens: list[str]) -> str:
        return f"{self.base_url}/compare/{','.join(tokens)}"

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=None, help="bind address; defaults to Tailscale IPv4")
    parser.add_argument("--port", type=int, default=DEFAULT_REPLAY_PORT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--history-root", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--episodes",
        type=str,
        default="best,median,worst",
        help="comma-separated subset of best,median,worst (only with --run-id)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="comma-separated replay seeds; defaults to --seed alone",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_REPLAY_STEPS)
    parser.add_argument(
        "--full",
        action="store_true",
        help="replay each episode until death (up to 600 decisions)",
    )
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--difficulty", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--patterns", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--powerups", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    host = args.host or tailscale_ipv4() or DEFAULT_REPLAY_HOST
    store = ReplayStore()
    history_root = args.history_root or Path("history/dodge/gymnasium")
    initial_url = None
    compare_url = None
    if args.run_id is not None:
        wanted = [
            part.strip()
            for part in str(args.episodes).split(",")
            if part.strip() in EPISODE_KEYS
        ]
        if not wanted:
            parser.error(f"--episodes must be a subset of {','.join(EPISODE_KEYS)}")
        # --run-id replays the run's own eval seeds to full episodes by
        # default; an explicit --steps still caps the length.
        steps = MAX_REPLAY_STEPS if args.steps == DEFAULT_REPLAY_STEPS else args.steps
        server = ReplayServer(
            store=store, host=host, port=args.port, history_root=history_root
        )
        server.start()
        try:
            context = run_context(history_root, args.run_id)
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        best = context["best_eval_episode"] or {}
        print(
            f"Run {args.run_id} final best: seed {best.get('seed')} "
            f"({best.get('source')}, eval {best.get('eval_reward')}); "
            f"training max {context['train_max']['reward']:.1f} "
            f"@ step {int(context['train_max']['step'])} "
            f"(exploration-era, not replayable)",
            flush=True,
        )
        pairs = generate_run_comparison(
            history_root,
            args.run_id,
            episodes=wanted,
            steps=steps,
            device=args.device,
            epsilon=args.epsilon,
        )
        tokens = []
        for episode, replay in pairs[:MAX_REPLAYS]:
            url = server.url_for(store.add(replay))
            tokens.append(url.rsplit("/", 1)[-1])
            print(
                f"Run {args.run_id} {episode.key} "
                f"(seed {episode.seed}, {episode.source}, "
                f"eval {episode.eval_reward:.1f}, "
                f"replay {replay.total_reward:.1f}): {url}",
                flush=True,
            )
            if initial_url is None:
                initial_url = url
        if tokens:
            compare_url = server.url_for_compare(tokens)
    elif args.checkpoint is not None:
        seeds = [args.seed]
        if args.seeds:
            seeds = [int(part) for part in args.seeds.split(",") if part.strip() != ""]
        server = ReplayServer(
            store=store, host=host, port=args.port, history_root=history_root
        )
        server.start()
        tokens = []
        for seed in seeds[:MAX_REPLAYS]:
            replay = generate_native_replay(
                args.checkpoint,
                seed=seed,
                steps=args.steps,
                device=args.device,
                difficulty=args.difficulty,
                patterns=args.patterns,
                powerups=args.powerups,
                epsilon=args.epsilon,
            )
            url = server.url_for(store.add(replay))
            tokens.append(url.rsplit("/", 1)[-1])
            print(f"Native replay page (seed {seed}): {url}", flush=True)
            if initial_url is None:
                initial_url = url
        if len(tokens) > 1:
            compare_url = server.url_for_compare(tokens)
    else:
        server = ReplayServer(
            store=store, host=host, port=args.port, history_root=history_root
        )
        server.start()
    print(f"Native replay server: {server.base_url}", flush=True)
    if initial_url:
        print(f"Native replay page: {initial_url}", flush=True)
    if compare_url:
        print(f"Native replay comparison: {compare_url}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


__all__ = [
    "DEFAULT_REPLAY_HOST",
    "DEFAULT_REPLAY_PORT",
    "ReplayServer",
    "ReplayStore",
    "main",
    "tailscale_ipv4",
]


if __name__ == "__main__":
    raise SystemExit(main())
