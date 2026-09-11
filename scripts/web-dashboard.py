#!/usr/bin/env python3
"""Read-only browser mirror of the Pygame Dodge DDQN Training dashboard.

Same panels, same order, same math as DashboardRenderer.render:
sidebar Learning Control / Learning Statistics / System, main Episode Reward /
Progress / Achievements / Policy Entropy / Survival Time / Enemies Killed.
Series use the RunSeries semantics (EMA trend 0.12, first-5 baseline,
run-wide bounds). Reads history artifacts only; controls are shown disabled
except Watch Agent, which links the replay server.
"""
import contextlib
import html
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

HISTORY = os.path.join(os.getcwd(), "history/dodge/gymnasium/cnn-image-ddqn")
REPLAY_BASE = "http://100.100.169.122:8890"

TEXT, MUTED = "#e0dfdb", "#8b8982"
CYAN, BLUE, GREEN, YELLOW, ORANGE, RED, WHITE = (
    "#15c2eb", "#4191dc", "#71be59", "#f4bd2f",
    "#e07441", "#ff4c4c", "#f4f4f1",
)
BG, PANEL, GRID = "#1e1e1c", "#0e0f0f", "#2d2c29"
# One cockpit type scale shared with the Pygame board: 12 caption, 13 body,
# 20 hero values, monospace numerals everywhere so columns align.
BASE_FONT = "font-family:-apple-system,'Segoe UI',system-ui,sans-serif;font-size:13px"
MONO_FONT = "font-family:ui-monospace,'Cascadia Mono',Consolas,monospace;font-variant-numeric:tabular-nums"

EVENT_SPECS = (
    ("survival_frames", "Survival frames", CYAN),
    ("enemies_destroyed", "Enemies destroyed", YELLOW),
    ("explosion_pickups", "Explosion pickups", ORANGE),
    ("freeze_pickups", "Freeze pickups", "#5eb5e1"),
    ("shrink_pickups", "Shrink pickups", "#b182e0"),
    ("patterns_survived", "Patterns survived", GREEN),
    ("deaths", "Deaths", RED),
    ("lives_spent", "Lives spent", "#b182e0"),
)
PROGRESS_ROWS = (
    ("reward", "Episode reward", "", True),
    ("episode_length", "Survival time", "s", True),
    ("enemies_killed", "Enemies killed", "", True),
    ("entropy", "Policy entropy", "", False),
)
STATE_COLORS = {
    "queued": "#6b7280", "running": BLUE, "paused": YELLOW,
    "stale": YELLOW, "completed": GREEN, "failed": RED, "stopped": RED,
}


def load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def read_metrics(run_dir):
    rows = []
    try:
        with open(os.path.join(run_dir, "metrics.jsonl")) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    with contextlib.suppress(ValueError):
                        rows.append(json.loads(line))
    except OSError:
        pass
    return rows


def ema(values, smoothing=0.12):
    out, run = [], None
    for v in values:
        run = v if run is None else run + smoothing * (v - run)
        out.append(run)
    return out


def bounds(values, trend):
    lo = min(min(values), min(trend))
    hi = max(max(values), max(trend))
    margin = max((hi - lo) * 0.22, 0.5)
    return lo - margin, hi + margin


def stride(values, cap=240):
    return values[:: max(1, len(values) // cap)] if len(values) > cap else values


def delta_tag(delta, rising):
    if abs(delta) < 1e-6:
        return f"<span style='color:{MUTED}'>= +0.0 since start</span>"
    improved = delta > 0 if rising else delta < 0
    arrow = "▲" if delta > 0 else "▼"
    color = GREEN if improved else RED
    return f"<span style='color:{color}'>{arrow} {delta:+.1f} since start</span>"


def fmt_duration(seconds):
    seconds = max(0, int(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def card(title, inner, title_color=TEXT, link=None, name=None):
    tag = f" data-panel='{name}'" if name else ""
    box = (
        f"<div{tag} style='background:{PANEL};border-radius:8px;padding:10px 12px;"
        f"height:100%;box-sizing:border-box;overflow:hidden'>"
        f"<div style='font-weight:bold;color:{title_color};font-size:13px;"
        f"margin-bottom:6px'>{html.escape(title)}</div>{inner}</div>"
    )
    if link:
        return (
            f"<a href='{link}' style='display:block;height:100%;text-decoration:none;"
            f"color:inherit'>{box}</a>"
        )
    return box


def chart_panel(title, raw, color, suffix="", rising=True, elapsed=0.0,
                latest=None, link=None, big=False, pname=None):
    if len(raw) < 2:
        return card(title, "<p style='color:" + MUTED + "'>collecting…</p>", color, link, pname)
    cap = 4000 if big else 240
    raw = stride(raw, cap)
    trend_full = ema(raw)
    trend = stride(trend_full, cap)
    lo, hi = bounds(raw, trend_full)
    w, h, pad = (1100, 420, 10) if big else (560, 96, 6)
    span = max(hi - lo, 1e-9)
    n = len(raw)

    def pts(vals):
        return " ".join(
            f"{pad + i * (w - 2 * pad) / max(1, n - 1):.1f},"
            f"{h - pad - (v - lo) / span * (h - 2 * pad):.1f}"
            for i, v in enumerate(vals)
        )

    base = float(sum(raw[:5]) / min(5, len(raw)))
    delta = trend_full[-1] - base
    head = latest if latest is not None else raw[-1]
    return card(
        title,
        f"<div style='text-align:right;color:{color};font-size:20px;{MONO_FONT}'>"
        f"{head:.1f}{html.escape(suffix)}</div>"
        f"<svg width='100%' viewBox='0 0 {w} {h}' style='background:{PANEL}'>"
        f"<polygon points='{pad},{h - pad} {pts(raw)} {w - pad},{h - pad}' "
        f"fill='{color}' opacity='0.15'/>"
        f"<polyline points='{pts(raw)}' fill='none' stroke='{color}' "
        "stroke-width='1' opacity='0.4'/>"
        f"<polyline points='{pts(trend)}' fill='none' stroke='{color}' "
        "stroke-width='2.5'/></svg>"
        f"<div style='display:flex;justify-content:space-between;color:{MUTED};"
        f"font-size:12px'><span>run start → now ({fmt_duration(elapsed)})</span>"
        f"<span>{delta_tag(delta, rising)}</span></div>",
        color,
        link,
        pname,
    )


def progress_panel(rows, link=None, pname=None):
    if not rows:
        return card("Progress", "<p style='color:" + MUTED + "'>collecting…</p>", link=link)
    out = "<div style='text-align:right;color:" + MUTED + ";font-size:12px'>since run start</div>"
    for key, label, unit, rising in PROGRESS_ROWS:
        vals = [float(r.get(key, 0.0) or 0.0) for r in rows]
        if len(vals) < 2:
            out += f"<div style='{MONO_FONT}'>{html.escape(label)}: collecting…</div>"
            continue
        trend = ema(vals)
        base, smooth, delta = sum(vals[:5]) / 5, trend[-1], trend[-1] - sum(vals[:5]) / 5
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
        color = GREEN if (delta > 0) == rising and abs(delta) >= 1e-6 else (MUTED if abs(delta) < 1e-6 else RED)
        out += (
            f"<div style='margin:8px 0'><div style='display:flex;justify-content:space-between'>"
            f"<span style='color:{MUTED};font-size:12px'>{html.escape(label)}</span>"
            f"<b style='color:{color};{MONO_FONT}'>{arrow} {delta:+.1f}{html.escape(unit)}</b></div>"
            f"<div style='{MONO_FONT}'>{base:.1f} → {smooth:.1f}{html.escape(unit)}</div></div>"
        )
    return card("Progress", out, link=link, name=pname)


def final_best_reward(rep):
    """Reported best: best frozen final-eval episode, never the training max."""

    final = rep.get("final_metrics") or {}
    if isinstance(final.get("best_eval_reward"), (int, float)):
        return float(final["best_eval_reward"])
    ev = rep.get("evaluation") or {}
    best = None
    for source in ("inner", "holdout"):
        section = ev.get(source) if isinstance(ev, dict) else None
        if not isinstance(section, dict):
            continue
        seeds = section.get("seeds") or []
        rewards = section.get("rewards") or []
        survival = section.get("survival_frames") or []
        for i, (seed, reward) in enumerate(zip(seeds, rewards)):
            try:
                key = (float(reward), float(survival[i]) if i < len(survival) else 0.0, -float(seed))
            except (TypeError, ValueError):
                continue
            if best is None or key > best[0]:
                best = (key, float(reward))
    return best[1] if best else None


def stats_panel(st, manifest, cfg, latest, elapsed, rep=None, link=None, pname=None):
    ckpts, ckdir = [], os.path.join(os.path.dirname(st.get("_dir", "")), "checkpoints")
    with contextlib.suppress(OSError):
        ckpts = sorted(os.listdir(ckdir))
    model = ckpts[-1].replace(".pt", "") if ckpts else "None"
    upd = latest.get("optimizer_step", "?")
    best_eval = final_best_reward(rep or {})
    dead = latest.get("dead_units")
    rows = (
        ("Status", str(st.get("state", "?")).title()),
        ("Model", model),
        ("Update", f"{upd:,}" if isinstance(upd, int) else str(upd)),
        ("Seed", str(manifest.get("seed", "?"))),
        ("Steps / second", f"{float(latest.get('throughput', 0.0)):,.0f}"),
        ("Best episode (train)", f"{float(latest.get('best_score', 0.0)):.1f}"),
        ("Best episode (final)", f"{best_eval:.1f}" if best_eval is not None else "-"),
        ("Learning time", fmt_duration(elapsed)),
        ("Dead units", f"{float(dead):.0%}" if isinstance(dead, (int, float)) else "-"),
    )
    out = "".join(
        f"<div style='display:flex;justify-content:space-between'>"
        f"<span style='color:{MUTED}'>{k}</span><span style='{MONO_FONT}'>{html.escape(str(v))}</span></div>"
        for k, v in rows
    )
    total = (cfg.get("run") or {}).get("steps", 0) if isinstance(cfg, dict) else 0
    frac = min(1.0, (st.get("step", 0) or 0) / total) if total else 0.0
    out += (
        f"<div style='color:{MUTED};font-size:12px;margin-top:6px'>Rollout</div>"
        f"<div style='background:{GRID};border-radius:2px;height:5px'>"
        f"<div style='width:{frac:.0%};background:{GREEN};height:5px'></div></div>"
    )
    return card("Learning Statistics", out, link=link, name=pname)


def system_panel(device, link=None, pname=None):
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        proc = psutil.Process().memory_info().rss / (1024 * 1024)
    except (ImportError, OSError):
        cpu, mem, proc = 0.0, 0.0, 0.0
    accel = {"cuda": "CUDA", "cpu": "CPU"}.get(str(device), str(device))
    out = ""
    for label, text, color, frac in (
        ("CPU", f"{cpu:.0f}%", CYAN, cpu / 100),
        ("Memory", f"{proc:.0f} MB", GREEN, mem / 100),
        ("Accelerator", accel, YELLOW, 0.18),
    ):
        out += (
            f"<div style='display:flex;justify-content:space-between'>"
            f"<span style='color:{MUTED}'>{label}</span><span style='{MONO_FONT}'>{text}</span></div>"
            f"<div style='background:{GRID};border-radius:2px;height:4px;margin:2px 0 8px'>"
            f"<div style='width:{frac:.0%};background:{color};height:4px'></div></div>"
        )
    return card("System", out, link=link, name=pname)


def events_panel(rows, link=None, pname=None):
    if not rows:
        return card("Achievements", "collecting…", link=link, name=pname)
    out = ("<table style='width:100%'><tr style='color:" + MUTED + ";font-size:12px'>"
           "<td>Event</td><td align='right'>Rollout</td><td align='right'>Total</td>"
           "<td align='right'>Recent rollouts</td></tr>")
    for key, label, color in EVENT_SPECS:
        if key == "survival_frames":
            hist = [int(r.get("survival_frames", 0) or 0) for r in rows[-18:]]
        elif key == "deaths":
            hist = [int(r.get("deaths", 0) or 0) for r in rows[-18:]]
        else:
            hist = [int(r.get(key, 0) or 0) for r in rows[-18:]]
        rollout, total = hist[-1], sum(int(r.get(key, 0) or 0) for r in rows
                                       if key not in ("survival_frames",)) if key not in ("survival_frames",) else sum(hist)
        hi = max(max(hist), 1)
        bars = "".join(
            f"<span title='{v}' style='display:inline-block;width:5px;margin-right:1px;"
            f"height:{max(2, round(v / hi * 22))}px;background:{color};"
            "vertical-align:bottom'></span>" for v in hist
        )
        out += (f"<tr><td style='color:{color}'>{label}</td>"
                f"<td align='right' style='{MONO_FONT}'>{rollout:,}</td>"
                f"<td align='right' style='{MONO_FONT}'>{total:,}</td>"
                f"<td align='right'>{bars}</td></tr>")
    return card("Achievements", out + "</table>", link=link, name=pname)


def controls_panel(state, link=None, pname=None, board=None):
    # One neutral chrome system; only Watch Agent takes the cyan accent.
    btns = []
    for name in ("Pause", "Rewards", "Game Config", "Models"):
        btns.append(
            f"<div style='background:#242321;border:1px solid #393733;color:{TEXT};"
            f"border-radius:6px;text-align:center;padding:10px;margin:5px 0'>{name} (live session only)</div>")
    watch = board or (REPLAY_BASE + "/")
    btns.append(
        f"<a href='{watch}' style='display:block;background:#16586a;"
        f"color:{TEXT};border-radius:6px;text-align:center;padding:10px;margin:5px 0;"
        "text-decoration:none;font-weight:bold'>Watch Agent ↗</a>")
    return card("Learning Control", "".join(btns), link=link, name=pname)


def config_panel(cfg, link=None, pname=None):
    if not isinstance(cfg, dict):
        return ""
    game = cfg.get("dashboard_game_controls", cfg.get("game", {}))
    model = cfg.get("model", {})
    items = "".join(
        f"<div style='display:flex;justify-content:space-between;font-size:13px'>"
        f"<span style='color:{MUTED}'>{html.escape(str(k))}</span>"
        f"<span>{html.escape(str(v))}</span></div>"
        for k, v in list(game.items())[:8] + [("architecture", model.get("architecture", "?")),
                                              ("learning_rate", model.get("learning_rate", "?")),
                                              ("gamma", model.get("gamma", "?"))]
    )
    return card("Game Config / Model (read-only)", items, link=link, name=pname)


def run_page(rid):
    d = os.path.join(HISTORY, rid)
    if not os.path.isdir(d):
        return None
    st = load_json(os.path.join(d, "status.json")) or {}
    st["_dir"] = d
    manifest = load_json(os.path.join(d, "manifest.json")) or {}
    cfg = load_json(os.path.join(d, "config.json")) or {}
    rep = load_json(os.path.join(d, "report.json")) or {}
    ev = load_json(os.path.join(d, "evaluation.json")) or rep.get("evaluation") or {}
    rows = read_metrics(d)
    latest = rows[-1] if rows else {}
    step = latest.get("step", st.get("step", 0)) or 0
    elapsed = step / max(float(latest.get("throughput", 1) or 1), 1e-9)

    reward = [float(r.get("reward", 0.0) or 0.0) for r in rows]
    entropy = [float(r.get("policy_entropy", r.get("entropy", 0.0)) or 0.0) for r in rows]
    length = [float(r.get("episode_length", r.get("survival_frames", 0.0) / 60.0) or 0.0) for r in rows]
    kills = [float(r.get("enemies_killed", 0.0) or 0.0) for r in rows]
    for r in rows:
        r["entropy"] = float(r.get("policy_entropy", r.get("entropy", 0.0)) or 0.0)

    state = st.get("state", "?")
    ev_html = (f"Eval: mean reward {ev.get('mean_reward', '?')}, mean survival "
               f"{ev.get('mean_survival_frames', '?')}" if ev else "")
    base = f"/run/{html.escape(rid)}/p"
    ctx = {
        "rid": rid, "st": st, "manifest": manifest, "cfg": cfg, "rows": rows,
        "latest": latest, "elapsed": elapsed, "reward": reward, "entropy": entropy,
        "length": length, "kills": kills, "state": state, "ev_html": ev_html,
        "base": base, "rep": rep,
    }
    return board_page(ctx)


def board_panels(ctx):
    st, manifest, cfg = ctx["st"], ctx["manifest"], ctx["cfg"]
    rows, latest, elapsed = ctx["rows"], ctx["latest"], ctx["elapsed"]
    base = ctx["base"]
    return {
        "controls": controls_panel(ctx["state"], link=f"{base}/controls", pname="controls",
                                    board=f"/run/{html.escape(ctx['rid'])}"),
        "stats": stats_panel(st, manifest, cfg, latest, elapsed, ctx.get("rep"), link=f"{base}/stats", pname="stats"),
        "system": system_panel(manifest.get("device", "?"), link=f"{base}/system", pname="system"),
        "replay": replay_panel(ctx),
        "reward": chart_panel("Episode Reward", ctx["reward"], CYAN, elapsed=elapsed, link=f"{base}/reward", pname="reward"),
        "progress": progress_panel(rows, link=f"{base}/progress", pname="progress"),
        "events": events_panel(rows, link=f"{base}/events", pname="events"),
        "entropy": chart_panel("Policy Entropy", ctx["entropy"], RED, rising=False, elapsed=elapsed, link=f"{base}/entropy", pname="entropy"),
        "length": chart_panel("Survival Time", ctx["length"], YELLOW, suffix=" s", elapsed=elapsed, link=f"{base}/length", pname="length"),
        "enemies": chart_panel("Enemies Killed", ctx["kills"], ORANGE, elapsed=elapsed, link=f"{base}/enemies", pname="enemies"),
    }


def board_page(ctx):
    rid = ctx["rid"]
    panels = board_panels(ctx)
    board_css = (
        "margin:0;height:100vh;overflow:hidden;box-sizing:border-box;padding:6px 10px;"
        f"background:{BG};color:{TEXT};font-family:sans-serif;font-size:11px"
    )
    cell = "min-height:0;min-width:0"
    return (
        f"<html><head><title>{html.escape(rid)}</title>"
        "<script>"
        f"const RID={json.dumps(rid)};let curV=null,watchdog=null;"
        "function armWatchdog(){clearTimeout(watchdog);watchdog=setTimeout(()=>location.reload(),90000);}"
        "async function refreshFrags(){"
        "const r=await fetch(`/run/${RID}/frags`);if(!r.ok)return;"
        "const j=await r.json();"
        "for(const [n,h] of Object.entries(j.panels)){"
        "const el=document.querySelector(`[data-panel=\"${n}\"]`);"
        "if(el&&n!=='replay')el.outerHTML=h;}"
        "if(j.header){document.getElementById('board-head').outerHTML=j.header;}}"
        "function connect(){"
        "const ws=new WebSocket(`ws://${location.host}/ws/${RID}`);"
        "ws.onmessage=(e)=>{const v=JSON.parse(e.data);"
        "if(curV!==null&&v.v!==curV)refreshFrags();curV=v.v;armWatchdog();};"
        "ws.onclose=()=>setTimeout(connect,5000);}"
        "connect();armWatchdog();"
        "</script></head>"
        f"<body style='{board_css}'>"
        f"{board_header(ctx)}"
        "<div style='display:grid;grid-template-columns:200px 1fr;gap:8px;"
        "height:calc(100vh - 36px)'>"
        "<div style='display:grid;grid-template-rows:auto auto 1fr;gap:8px;min-height:0'>"
        f"<div style='{cell}'>{panels['controls']}</div>"
        f"<div style='{cell}'>{panels['stats']}</div>"
        f"<div style='{cell}'>{panels['system']}</div>"
        "</div>"
        "<div style='display:grid;grid-template-columns:44% 1fr;gap:8px;min-height:0'>"
        "<div style='display:grid;grid-template-rows:auto 1fr 1fr;gap:8px;min-height:0'>"
        f"<div style='{cell}'>{panels['replay']}</div>"
        f"<div style='{cell}'>{panels['reward']}</div>"
        f"<div style='{cell}'>{panels['events']}</div>"
        "</div>"
        "<div style='display:grid;grid-template-rows:auto 1fr 1fr 1fr;gap:8px;min-height:0'>"
        f"<div style='{cell}'>{panels['progress']}</div>"
        f"<div style='{cell}'>{panels['entropy']}</div>"
        f"<div style='{cell}'>{panels['length']}</div>"
        f"<div style='{cell}'>{panels['enemies']}</div>"
        "</div></div></div></body></html>"
    )


def board_header(ctx):
    rid, state = ctx["rid"], ctx["state"]
    st, ev_html = ctx["st"], ctx["ev_html"]
    return (
        f"<div id='board-head' style='display:flex;gap:10px;align-items:center;height:24px;"
        "white-space:nowrap;overflow:hidden'>"
        f"<a href='/' style='color:{CYAN}'>← runs</a>"
        f"<b>Dodge DDQN Training</b><span style='color:{MUTED}'>{html.escape(rid)}</span>"
        f"<span style='background:{STATE_COLORS.get(state, '#6b7280')};color:#fff;"
        f"padding:1px 10px;border-radius:10px'>● {html.escape(str(state))}</span>"
        f"<span style='color:{MUTED}'>{html.escape(str(st.get('message', '')))}"
        + (f" · {html.escape(ev_html)}" if ev_html else "") + "</span>"
        "<span style='color:" + MUTED + "'> · live · click a panel to zoom</span></div>"
    )


REPLAY_CACHE = "/tmp/opencode/replay-cache"
REPLAY_STEPS = 600
REPLAY_MAX_SEEDS = 8
# Showcase rollouts run slightly exploratory: the pure greedy argmax has
# collapsed onto one action, so eps 0 shows the lock-in, not the skill.
REPLAY_EPSILON = 0.1
_gen_lock = None


def _replay_gen_lock():
    global _gen_lock
    if _gen_lock is None:
        import threading

        _gen_lock = threading.Lock()
    return _gen_lock


def latest_checkpoint(run_dir):
    ckdir = os.path.join(run_dir, "checkpoints")
    try:
        names = sorted(
            (n for n in os.listdir(ckdir) if n.endswith(".pt")),
            key=lambda n: os.stat(os.path.join(ckdir, n)).st_mtime,
            reverse=True,
        )
    except OSError:
        return None
    return os.path.join(ckdir, names[0]) if names else None


def replay_seeds(run_dir):
    """Seeds from this run: measured eval seeds first, then train seeds."""

    seeds = []
    for name in ("evaluation.json", "report.json"):
        try:
            with open(os.path.join(run_dir, name)) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        ev = doc.get("evaluation", doc) if isinstance(doc, dict) else {}
        if isinstance(ev, dict):
            for seed in ev.get("seeds", []):
                if isinstance(seed, int) and seed not in seeds:
                    seeds.append(seed)
    try:
        with open(os.path.join(run_dir, "manifest.json")) as fh:
            base_seed = int(json.load(fh).get("seed", 42))
    except (OSError, ValueError):
        base_seed = 42
    for episode in range(REPLAY_MAX_SEEDS):
        seed = base_seed + episode
        if seed not in seeds and len(seeds) < REPLAY_MAX_SEEDS:
            seeds.append(seed)
    return seeds[:REPLAY_MAX_SEEDS]


def ensure_replay(rid):
    """Return cache state; spawn a background generator when stale.

    State is one of: missing checkpoint | generating(done, total) |
    ready(replays=[{seed, frames, ...}] longest first) | failed(reason).
    """

    run_dir = os.path.join(HISTORY, rid)
    checkpoint = latest_checkpoint(run_dir)
    if checkpoint is None:
        return {"status": "empty"}
    out_dir = os.path.join(REPLAY_CACHE, rid)
    meta_path = os.path.join(out_dir, "meta.json")
    fresh_meta = None
    try:
        with open(meta_path) as fh:
            meta = json.load(fh)
        if (
            meta.get("status") == "ready"
            and isinstance(meta.get("replays"), list)
            and meta.get("replays")
            and meta.get("epsilon", 0.0) == REPLAY_EPSILON
            and meta.get("checkpoint_mtime")
            == os.stat(checkpoint).st_mtime
        ):
            fresh_meta = meta
    except (OSError, ValueError):
        pass
    if fresh_meta is not None:
        with contextlib.suppress(OSError):
            os.remove(os.path.join(out_dir, ".generating"))
        fresh_meta["cache_dir"] = out_dir
        return fresh_meta
    try:
        with open(meta_path) as fh:
            generating = json.load(fh)
        if generating.get("status") == "generating":
            return generating
    except (OSError, ValueError):
        pass
    with _replay_gen_lock():
        import time

        marker = os.path.join(out_dir, ".generating")
        try:
            if os.path.isfile(marker):
                age = time.time() - os.stat(marker).st_mtime
                if age < 1800:
                    return {"status": "generating", "done": 0, "total": REPLAY_STEPS}
                os.remove(marker)
                with contextlib.suppress(OSError):
                    os.remove(os.path.join(out_dir, "meta.json"))
            os.makedirs(out_dir, exist_ok=True)
            with open(marker, "w") as fh:
                fh.write("1")
        except OSError:
            return {"status": "failed", "reason": "cache dir not writable"}
    import glob
    import subprocess

    env = dict(os.environ)
    ld_path = env.get("LD_LIBRARY_PATH", "")
    for pattern in ("/nix/store/*-gcc-*-lib/lib", "/nix/store/*-zlib-*/lib"):
        for candidate in sorted(glob.glob(pattern)):
            if os.path.isfile(os.path.join(candidate, "libstdc++.so.6")) or os.path.isfile(
                os.path.join(candidate, "libz.so.1")
            ):
                if candidate not in ld_path:
                    ld_path = f"{candidate}:{ld_path}" if ld_path else candidate
                break
    env["LD_LIBRARY_PATH"] = ld_path
    venv_python = os.path.join(os.getcwd(), ".venv", "bin", "python")
    seeds_csv = ",".join(str(s) for s in replay_seeds(run_dir))
    try:
        subprocess.Popen(
            [
                venv_python,
                "scripts/gen-replay-cache.py",
                "--checkpoint",
                checkpoint,
                "--out-dir",
                out_dir,
                "--seeds",
                seeds_csv,
                "--steps",
                str(REPLAY_STEPS),
                "--epsilon",
                str(REPLAY_EPSILON),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
    except OSError:
        return {"status": "failed", "reason": "could not spawn generator"}
    return {"status": "generating", "done": 0, "total": max(1, len(seeds_csv.split(",")))}


def replay_panel(ctx):
    rid = ctx["rid"]
    rep = ensure_replay(rid)
    status = rep.get("status")
    if status == "empty":
        return card("Live Replay", (
            "<p style='color:" + MUTED + "'>no checkpoint in this run yet.</p>"),
            name="replay")
    if status == "failed":
        return card("Live Replay", (
            "<p style='color:" + RED + "'>replay failed: "
            f"{html.escape(str(rep.get('reason', '?')))}</p>"), name="replay")
    if status != "ready":
        done, total = int(rep.get("done", 0)), max(1, int(rep.get("total", 1)))
        pct = min(100, round(done / total * 100))
        return card("Live Replay", (
            f"<p style='color:{MUTED}'>generating replays… {pct}%</p>"
            "<div style='background:" + GRID + ";border-radius:2px;height:5px'>"
            f"<div style='width:{pct}%;background:{CYAN};height:5px'></div></div>"
            "<script>setTimeout(()=>location.reload(),10000);</script>"), name="replay")
    replays = rep["replays"]
    featured = replays[0]
    eps_label = f"eps {rep.get('epsilon', 0.0)} showcase"
    base = f"/run/{html.escape(rid)}/replay"
    seed_btns = "".join(
        f"<button data-seed='{r['seed']}' data-frames='{r['frames']}' "
        f"class='rseed' style='flex:1'>{r['seed']}·{r['frames']}</button>"
        for r in replays
    )
    inner = (
        "<div style='display:flex;gap:8px;align-items:center'>"
        f"<img id='replay-img' src='{base}/seed/{featured['seed']}/frame/0.png' "
        "style='width:132px;height:132px;image-rendering:pixelated;background:#000;"
        "border-radius:4px'>"
        "<div style='flex:1;min-width:0'>"
        f"<div style='font-size:11px'>{html.escape(str(rep.get('checkpoint', '?')))} · {html.escape(eps_label)}</div>"
        f"<div id='replay-label' style='color:{MUTED};font-size:11px'></div>"
        "<div style='display:flex;gap:6px;margin-top:4px'>"
        "<button id='replay-play' style='flex:1'>Pause</button>"
        "<button id='replay-view' style='flex:1'>Collision</button>"
        "</div>"
        f"<div style='display:flex;gap:4px;margin-top:4px;flex-wrap:wrap'>{seed_btns}</div>"
        "</div></div>"
        "<script>"
        f"const RBASE={json.dumps(base)};"
        f"let RSEED={featured['seed']},RN={featured['frames']};"
        "let rIdx=0,rPlay=true,rView=0,rTimer=null;"
        "function rShow(){const el=document.getElementById('replay-img');if(!el)return;"
        "el.src=rView?`${RBASE}/seed/${RSEED}/collision/${rIdx}.png`:`${RBASE}/seed/${RSEED}/frame/${rIdx}.png`;"
        "document.getElementById('replay-label').textContent=`seed ${RSEED} · ${rIdx} / ${RN-1}`;}"
        "function rTick(){rIdx=(rIdx+1)%RN;rShow();}"
        "function rArm(){clearInterval(rTimer);if(rPlay)rTimer=setInterval(rTick,1000/15);}"
        "document.getElementById('replay-play').onclick=(e)=>{rPlay=!rPlay;e.target.textContent=rPlay?'Pause':'Play';rArm();};"
        "document.getElementById('replay-view').onclick=(e)=>{rView=rView?0:1;e.target.textContent=rView?'Game':'Collision';rShow();};"
        "document.querySelectorAll('.rseed').forEach((b)=>{b.onclick=()=>{"
        "RSEED=Number(b.dataset.seed);RN=Number(b.dataset.frames);rIdx=0;rShow();};});"
        "rShow();rArm();"
        "</script>"
    )
    inner += "<div style='display:none'>" + "".join(
        f"<img src='{base}/seed/{featured['seed']}/frame/{i}.png'>"
        for i in range(featured["frames"])) + "</div>"
    return card("Live Replay", inner, name="replay")


PANEL_TITLES = {
    "reward": "Episode Reward", "progress": "Progress", "events": "Achievements",
    "entropy": "Policy Entropy", "length": "Survival Time", "enemies": "Enemies Killed",
    "controls": "Learning Control", "stats": "Learning Statistics", "system": "System",
    "config": "Game Config / Model",
}


def panel_page(rid, name):
    d = os.path.join(HISTORY, rid)
    if not os.path.isdir(d) or name not in PANEL_TITLES:
        return None
    st = load_json(os.path.join(d, "status.json")) or {}
    st["_dir"] = d
    manifest = load_json(os.path.join(d, "manifest.json")) or {}
    cfg = load_json(os.path.join(d, "config.json")) or {}
    rows = read_metrics(d)
    latest = rows[-1] if rows else {}
    step = latest.get("step", st.get("step", 0)) or 0
    elapsed = step / max(float(latest.get("throughput", 1) or 1), 1e-9)
    back = f"<p><a href='/run/{html.escape(rid)}' style='color:{CYAN}'>← board</a></p>"
    title = PANEL_TITLES[name]
    if name == "reward":
        vals = [float(r.get("reward", 0.0) or 0.0) for r in rows]
        body = chart_panel(title, vals, CYAN, elapsed=elapsed, big=True)
    elif name == "entropy":
        vals = [float(r.get("policy_entropy", r.get("entropy", 0.0)) or 0.0) for r in rows]
        body = chart_panel(title, vals, RED, rising=False, elapsed=elapsed, big=True)
    elif name == "length":
        vals = [float(r.get("episode_length", r.get("survival_frames", 0.0) / 60.0) or 0.0) for r in rows]
        body = chart_panel(title, vals, YELLOW, suffix=" s", elapsed=elapsed, big=True)
    elif name == "enemies":
        vals = [float(r.get("enemies_killed", 0.0) or 0.0) for r in rows]
        body = chart_panel(title, vals, ORANGE, elapsed=elapsed, big=True)
    elif name == "progress":
        body = progress_panel(rows)
    elif name == "events":
        body = events_panel(rows)
    elif name == "stats":
        rep = load_json(os.path.join(d, "report.json")) or {}
        body = stats_panel(st, manifest, cfg, latest, elapsed, rep)
    elif name == "system":
        body = system_panel(manifest.get("device", "?"))
    elif name == "controls":
        body = controls_panel(st.get("state", "?"))
    else:
        body = config_panel(cfg)
    return (
        f"<html><head><title>{html.escape(rid)} · {html.escape(title)}</title></head>"
        f"<body style='background:{BG};color:{TEXT};font-family:sans-serif;"
        f"max-width:1200px;margin:0 auto;padding:16px'>{back}<h1>{html.escape(title)} "
        f"<small style='color:{MUTED}'>{html.escape(rid)}</small></h1>{body}</body></html>"
    )


def catalog_entries():
    entries = []
    try:
        names = os.listdir(HISTORY)
    except OSError:
        return entries
    for rid in names:
        d = os.path.join(HISTORY, rid)
        if not os.path.isdir(d) or rid == "models":
            continue
        st = load_json(os.path.join(d, "status.json")) or {}
        rep = load_json(os.path.join(d, "report.json")) or {}
        cur = st.get("current_metrics") or {}
        final = rep.get("final_metrics") or {}
        ev = rep.get("evaluation") or {}
        inner = ev.get("inner", {}) if isinstance(ev, dict) else {}
        try:
            updated = os.stat(os.path.join(d, "manifest.json")).st_mtime
        except OSError:
            try:
                updated = os.stat(d).st_mtime
            except OSError:
                updated = 0.0
        best = cur.get("best_score", final.get("best_score", "?"))
        entries.append({
            "rid": rid, "state": st.get("state", "?"),
            "gate": st.get("gate", rep.get("quality_gate", "?")),
            "step": st.get("step", "?"), "best": best,
            "eval": inner.get("mean_reward", ev.get("mean_reward", "?"))
            if isinstance(ev, dict) else "?",
            "updated": updated,
        })
    entries.sort(key=lambda e: e["updated"], reverse=True)
    try:
        best_val = max(float(e["best"]) for e in entries
                       if isinstance(e["best"], (int, float)))
    except ValueError:
        best_val = None
    for index, entry in enumerate(entries):
        entry["newest"] = index == 0
        entry["top"] = (
            best_val is not None and isinstance(entry["best"], (int, float))
            and float(entry["best"]) == best_val)
    return entries


def catalog_page():
    rows = []
    for e in catalog_entries():
        rid = e["rid"]
        badges = ""
        if e["newest"]:
            badges += f" <b style='color:{CYAN}'>NEWEST</b>"
        if e["top"]:
            badges += f" <b style='color:{YELLOW}'>★ BEST</b>"
        rows.append(
            f"<tr><td><a href='/run/{html.escape(rid)}' style='color:{CYAN}'>"
            f"{html.escape(rid)}</a>{badges}</td>"
            f"<td><span style='background:{STATE_COLORS.get(e['state'], '#6b7280')};"
            f"color:#fff;padding:2px 10px;border-radius:10px'>● {html.escape(str(e['state']))}</span></td>"
            f"<td>{html.escape(str(e['gate']))}</td>"
            f"<td>{e['step']}</td><td>{e['best']}</td><td>{e['eval']}</td></tr>")
    return (
        "<html><head>"
        "<title>Dodge Gymnasium runs</title></head>"
        f"<body style='background:{BG};color:{TEXT};font-family:sans-serif;padding:16px'>"
        "<h1>Dodge DDQN Training — runs, newest first (read-only)</h1>"
        "<table border='1' cellpadding='6' style='border-collapse:collapse'>"
        "<tr><th>run</th><th>state</th><th>gate</th><th>step</th>"
        "<th>best</th><th>eval reward</th></tr>"
        + "".join(rows) + "</table></body></html>")


def serve_replay_file(path):
    """Serve one cached replay PNG or meta JSON for a run."""

    rest = path[len("/run/"):]
    rid, _, tail = rest.partition("/replay/")
    if not rid or not tail or "/" in rid or rid.startswith("."):
        return None
    out_dir = os.path.join(REPLAY_CACHE, rid)
    if tail == "meta":
        try:
            with open(os.path.join(out_dir, "meta.json"), "rb") as fh:
                return fh.read(), "application/json"
        except OSError:
            return None
    parts = tail.split("/")
    seed_dir = out_dir
    if len(parts) == 4 and parts[0] == "seed" and parts[1].isdigit():
        seed_dir = os.path.join(out_dir, f"seed-{int(parts[1])}")
        kind, filename = parts[2], parts[3]
    elif len(parts) == 1:
        kind, filename = "", parts[0]
    else:
        return None
    if (kind == "frame" or filename.startswith("frame-")) and filename.endswith(".png"):
        prefix = "frame-"
    elif (kind == "collision" or filename.startswith("collision-")) and filename.endswith(".png"):
        prefix = "collision-"
    else:
        return None
    try:
        index = int(filename[len(prefix):-4] if filename.startswith(prefix) else filename[:-4])
    except ValueError:
        return None
    try:
        with open(os.path.join(seed_dir, f"{prefix}{index:04d}.png"), "rb") as fh:
            return fh.read(), "image/png"
    except OSError:
        return None


def run_version(rid):
    """Cheap change stamp: metrics size/mtime plus status step."""

    d = os.path.join(HISTORY, rid)
    try:
        info = os.stat(os.path.join(d, "metrics.jsonl"))
        stamp = f"{info.st_size}:{info.st_mtime_ns}"
    except OSError:
        stamp = "0:0"
    st = load_json(os.path.join(d, "status.json")) or {}
    return f"{stamp}:{st.get('step', 0)}:{st.get('state', '')}"


def frags_page(rid):
    d = os.path.join(HISTORY, rid)
    if not os.path.isdir(d):
        return None
    st = load_json(os.path.join(d, "status.json")) or {}
    st["_dir"] = d
    manifest = load_json(os.path.join(d, "manifest.json")) or {}
    cfg = load_json(os.path.join(d, "config.json")) or {}
    rep = load_json(os.path.join(d, "report.json")) or {}
    ev = load_json(os.path.join(d, "evaluation.json")) or rep.get("evaluation") or {}
    rows = read_metrics(d)
    latest = rows[-1] if rows else {}
    step = latest.get("step", st.get("step", 0)) or 0
    elapsed = step / max(float(latest.get("throughput", 1) or 1), 1e-9)
    reward = [float(r.get("reward", 0.0) or 0.0) for r in rows]
    entropy = [float(r.get("policy_entropy", r.get("entropy", 0.0)) or 0.0) for r in rows]
    length = [float(r.get("episode_length", r.get("survival_frames", 0.0) / 60.0) or 0.0) for r in rows]
    kills = [float(r.get("enemies_killed", 0.0) or 0.0) for r in rows]
    for r in rows:
        r["entropy"] = float(r.get("policy_entropy", r.get("entropy", 0.0)) or 0.0)
    state = st.get("state", "?")
    ev_html = (f"Eval: mean reward {ev.get('mean_reward', '?')}, mean survival "
               f"{ev.get('mean_survival_frames', '?')}" if ev else "")
    base = f"/run/{html.escape(rid)}/p"
    ctx = {
        "rid": rid, "st": st, "manifest": manifest, "cfg": cfg, "rows": rows,
        "latest": latest, "elapsed": elapsed, "reward": reward, "entropy": entropy,
        "length": length, "kills": kills, "state": state, "ev_html": ev_html,
        "base": base, "rep": rep,
    }
    panels = board_panels(ctx)
    panels.pop("replay", None)
    return json.dumps(
        {"v": run_version(rid), "header": board_header(ctx), "panels": panels}
    ).encode()


class Handler(BaseHTTPRequestHandler):
    server_version = "DodgeWebDashboard/1.0"

    def log_message(self, *args):
        pass

    def handle_ws(self, rid):
        import base64
        import hashlib
        import time

        key = self.headers.get("Sec-WebSocket-Key", "")
        if not key:
            self.send_error(400, "websocket upgrade required")
            return
        digest = hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
        self.send_response(101)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", base64.b64encode(digest).decode())
        self.end_headers()
        self.connection.settimeout(60)
        try:
            while True:
                payload = json.dumps({"v": run_version(rid)}).encode()
                frame = bytes([0x81, len(payload)]) + payload
                self.wfile.write(frame)
                self.wfile.flush()
                time.sleep(2)
        except (OSError, ValueError):
            pass

    def _send_bytes(self, data, content_type):
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = unquote(self.path.split("?", 1)[0])
        if path.startswith("/ws/"):
            self.handle_ws(path[len("/ws/"):].strip("/"))
            return
        if path.startswith("/run/") and path.endswith("/frags"):
            rid = path[len("/run/"): -len("/frags")].strip("/")
            data = frags_page(rid)
            if data is None:
                self.send_error(404, "unknown run")
                return
            self._send_bytes(data, "application/json")
            return
        if "/replay/" in path and path.startswith("/run/"):
            body = serve_replay_file(path)
            if body is None:
                self.send_error(404, "no replay frame")
                return
            data, content_type = body
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=31536000")
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/":
            page = catalog_page()
        elif "/p/" in path and path.startswith("/run/"):
            rest = path[len("/run/"):].strip("/").split("/p/", 1)
            page = panel_page(rest[0], rest[1] if len(rest) > 1 else "")
            if page is None:
                self.send_error(404, "unknown run or panel")
                return
        elif path.startswith("/run/"):
            page = run_page(path[len("/run/"):].strip("/"))
            if page is None:
                self.send_error(404, "unknown run")
                return
        else:
            self.send_error(404, "not found")
            return
        data = page.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        self.send_error(403, "read-only")

    do_PUT = do_POST
    do_DELETE = do_POST


def main():
    host = sys.argv[sys.argv.index("--host") + 1] if "--host" in sys.argv else "127.0.0.1"
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8875
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"serving {HISTORY} on http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
