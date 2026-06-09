"""
Multi-Agent System Live Visualizer
Run:  uv run python viz_server.py
Open: http://localhost:8080
"""

import asyncio
import json
import os
import sys
import time
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv()

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from sse_starlette.sse import EventSourceResponse

from common.llm import get_llm

app = FastAPI()
_queue: asyncio.Queue | None = None


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def search_tax_law(query: str) -> str:
    """Search tax law knowledge base for relevant statutes and penalties."""
    kb = [
        (["tax", "evasion", "fraud", "irs", "penalty"],
         "Tax evasion (26 U.S.C. §7201): felony, up to $250K fine + 5 years prison. "
         "Civil fraud: 75% of underpayment (IRC §6663)."),
        (["offshore", "overseas", "fbar", "fatca"],
         "FBAR: up to $100K or 50% of account balance per violation. "
         "FATCA non-compliance: 30% withholding on US-source payments."),
    ]
    q = query.lower()
    results = [t for kws, t in kb if any(k in q for k in kws)]
    return "\n".join(results) if results else "No specific tax law matches found."


@tool
def search_compliance_law(query: str) -> str:
    """Search regulatory compliance knowledge base."""
    kb = [
        (["data", "privacy", "gdpr", "ccpa", "user"],
         "CCPA: $7,500/intentional violation. GDPR: 4% global revenue or €20M. FTC Act §5."),
        (["sox", "sarbanes", "sec", "financial"],
         "SOX §906: false certification — $5M fine, 20 years. §802 record destruction: 20 years."),
    ]
    q = query.lower()
    results = [t for kws, t in kb if any(k in q for k in kws)]
    return "\n".join(results) if results else "No specific compliance matches found."


@tool
def search_privacy_law(query: str) -> str:
    """Search data privacy and GDPR knowledge base."""
    kb = [
        (["gdpr", "data", "privacy", "consent", "personal"],
         "GDPR: up to €20M or 4% global annual turnover. Data breach: 72-hour notification."),
        (["ccpa", "california", "consumer", "breach"],
         "CCPA/CPRA: $7,500/intentional violation. Private right: $100-$750/consumer."),
    ]
    q = query.lower()
    results = [t for kws, t in kb if any(k in q for k in kws)]
    return "\n".join(results) if results else "No specific privacy law matches found."


# ── State ─────────────────────────────────────────────────────────────────────

def _last_wins(a: str, b: str) -> str:
    return b if b else a


class LegalState(TypedDict):
    question: str
    law_analysis: str
    needs_tax: bool
    needs_compliance: bool
    needs_privacy: bool
    tax_result: Annotated[str, _last_wins]
    compliance_result: Annotated[str, _last_wins]
    privacy_result: Annotated[str, _last_wins]
    final_answer: str


# ── Event helpers ─────────────────────────────────────────────────────────────

async def emit(evt_type: str, **data) -> None:
    if _queue:
        await _queue.put({"type": evt_type, **data})


def instrument(fn, name: str):
    """Wrap a node to emit node_start (with timestamp) and node_end (with elapsed)."""
    async def wrapper(state):
        t0 = time.perf_counter()
        await emit("node_start", node=name)
        result = await fn(state)
        elapsed = round(time.perf_counter() - t0, 1)
        await emit("node_end", node=name, elapsed=elapsed)
        return result
    return wrapper


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def _analyze_law(state: LegalState) -> dict:
    result = await get_llm().ainvoke([
        SystemMessage(content=(
            "You are a senior litigation attorney. Analyse the legal aspects. "
            "IMPORTANT: Keep under 80 words. Use bullet points only."
        )),
        HumanMessage(content=state["question"]),
    ])
    return {"law_analysis": result.content}


async def _check_routing(state: LegalState) -> dict:
    q = state["question"].lower()
    needs_tax = any(kw in q for kw in [
        "tax", "irs", "evasion", "revenue", "penalty", "income", "fbar", "fatca", "offshore",
    ])
    needs_compliance = any(kw in q for kw in [
        "compliance", "sec", "sox", "regulation", "fcpa", "aml", "bsa",
    ])
    needs_privacy = any(kw in q for kw in [
        "data", "privacy", "gdpr", "ccpa", "breach", "consent", "personal",
    ])
    if not needs_tax and not needs_compliance and not needs_privacy:
        needs_tax = True
    await emit("routing",
               needs_tax=needs_tax,
               needs_compliance=needs_compliance,
               needs_privacy=needs_privacy)
    return {"needs_tax": needs_tax, "needs_compliance": needs_compliance, "needs_privacy": needs_privacy}


async def _call_tax(state: LegalState) -> dict:
    from langgraph.prebuilt import create_react_agent
    agent = create_react_agent(
        model=get_llm(), tools=[search_tax_law],
        prompt="Specialist tax attorney. Use search_tax_law. Max 80 words, bullet points.",
    )
    r = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})
    return {"tax_result": r["messages"][-1].content}


async def _call_compliance(state: LegalState) -> dict:
    from langgraph.prebuilt import create_react_agent
    agent = create_react_agent(
        model=get_llm(), tools=[search_compliance_law],
        prompt="Regulatory compliance officer. Use search_compliance_law. Max 80 words, bullet points.",
    )
    r = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})
    return {"compliance_result": r["messages"][-1].content}


async def _call_privacy(state: LegalState) -> dict:
    from langgraph.prebuilt import create_react_agent
    agent = create_react_agent(
        model=get_llm(), tools=[search_privacy_law],
        prompt="Data protection attorney. Use search_privacy_law. Max 80 words, bullet points.",
    )
    r = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})
    return {"privacy_result": r["messages"][-1].content}


async def _aggregate(state: LegalState) -> dict:
    sections = [s for s in [
        f"Legal:\n{state['law_analysis']}"            if state.get("law_analysis")       else None,
        f"Tax:\n{state['tax_result']}"                if state.get("tax_result")         else None,
        f"Compliance:\n{state['compliance_result']}"  if state.get("compliance_result") else None,
        f"Privacy:\n{state['privacy_result']}"        if state.get("privacy_result")     else None,
    ] if s]
    result = await get_llm().ainvoke([
        SystemMessage(content="Synthesise these into a clear legal summary. Max 200 words."),
        HumanMessage(content="\n\n".join(sections)),
    ])
    return {"final_answer": result.content}


def _route(state: LegalState) -> list:
    sends = []
    if state.get("needs_tax"):        sends.append(Send("call_tax",        state))
    if state.get("needs_compliance"): sends.append(Send("call_compliance", state))
    if state.get("needs_privacy"):    sends.append(Send("call_privacy",    state))
    return sends or [Send("aggregate", state)]


def create_graph():
    g = StateGraph(LegalState)
    g.add_node("analyze_law",     instrument(_analyze_law,     "analyze_law"))
    g.add_node("check_routing",   instrument(_check_routing,   "check_routing"))
    g.add_node("call_tax",        instrument(_call_tax,        "call_tax"))
    g.add_node("call_compliance", instrument(_call_compliance, "call_compliance"))
    g.add_node("call_privacy",    instrument(_call_privacy,    "call_privacy"))
    g.add_node("aggregate",       instrument(_aggregate,       "aggregate"))
    g.set_entry_point("analyze_law")
    g.add_edge("analyze_law", "check_routing")
    g.add_conditional_edges(
        "check_routing", _route,
        ["call_tax", "call_compliance", "call_privacy", "aggregate"],
    )
    g.add_edge("call_tax",        "aggregate")
    g.add_edge("call_compliance", "aggregate")
    g.add_edge("call_privacy",    "aggregate")
    g.add_edge("aggregate", END)
    return g.compile()


# ── API ───────────────────────────────────────────────────────────────────────

@app.post("/run")
async def run_endpoint(request: Request):
    global _queue
    body = await request.json()
    question = body.get("question", "")

    queue: asyncio.Queue = asyncio.Queue()
    _queue = queue

    async def _run() -> None:
        try:
            graph = create_graph()
            result = await graph.ainvoke({
                "question": question,
                "law_analysis": "",
                "needs_tax": False, "needs_compliance": False, "needs_privacy": False,
                "tax_result": "", "compliance_result": "", "privacy_result": "",
                "final_answer": "",
            })
            await emit("done", answer=result["final_answer"])
        except Exception as exc:
            await emit("error", message=str(exc))

    asyncio.create_task(_run())

    async def _gen():
        while True:
            try:
                # timeout=1s so we can send heartbeat pings to keep SSE alive
                ev = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield {"data": json.dumps(ev)}
                if ev.get("type") in ("done", "error"):
                    break
            except asyncio.TimeoutError:
                # heartbeat — keeps browser connection open and flushes buffers
                yield {"data": json.dumps({"type": "ping"})}

    return EventSourceResponse(_gen())


@app.get("/")
def index():
    return HTMLResponse(HTML)


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Legal Multi-Agent — Live View</title>
<style>
:root {
  --bg:      #08090f;
  --surface: #0f1118;
  --card:    #13161f;
  --border:  #1e2235;
  --text:    #dde3ef;
  --muted:   #4a5568;
  --active:  #f59e0b;
  --done:    #10b981;
  --skip:    #0f1118;
}
*,*::before,*::after { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; padding: 24px 32px; }

/* ── Header ── */
.hdr { margin-bottom: 20px; }
.hdr h1 { font-size: 1.2rem; font-weight: 700; display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
.hdr h1 em { font-size: 0.72rem; font-weight: 400; font-style: normal; color: var(--muted); border: 1px solid var(--border); border-radius: 99px; padding: 2px 9px; }
.irow { display: flex; gap: 10px; max-width: 880px; }
.irow input { flex: 1; padding: 9px 13px; background: var(--card); border: 1px solid var(--border); border-radius: 8px; color: var(--text); font-size: 0.88rem; }
.irow input:focus { outline: none; border-color: var(--active); }
.irow button { padding: 9px 20px; background: var(--active); color: #000; border: none; border-radius: 8px; font-weight: 700; font-size: 0.88rem; cursor: pointer; white-space: nowrap; }
.irow button:disabled { opacity: .35; cursor: not-allowed; }

/* ── Architecture badge ── */
.arch-bar {
  display: flex; align-items: center; flex-wrap: wrap; gap: 6px 16px;
  max-width: 880px;
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 10px 14px; margin: 12px 0 20px; font-size: 0.78rem;
}
.arch-chip {
  display: flex; align-items: center; gap: 5px;
}
.arch-chip .k { color: var(--muted); }
.arch-chip .v { font-weight: 600; }
.arch-chip .v.sw  { color: #a78bfa; }   /* supervisor-workers = purple */
.arch-chip .v.a2a { color: #38bdf8; }   /* a2a = blue */
.arch-sep { color: var(--border); font-size: 1.1rem; }

/* ── Graph wrapper ── */
#gwrap {
  position: relative; display: flex; flex-direction: column; align-items: center;
  padding: 8px 0 16px; margin-bottom: 8px;
}
#svg-ov { position: absolute; top: 0; left: 0; pointer-events: none; overflow: visible; }
.gap { height: 44px; }
.prow { display: flex; gap: 18px; align-items: flex-start; justify-content: center; }

/* ── Node card ── */
.node {
  position: relative; display: flex; flex-direction: column; align-items: center; gap: 4px;
  padding: 13px 18px 10px; min-width: 136px;
  background: var(--card); border: 1.5px solid var(--border); border-radius: 14px;
  text-align: center; transition: border-color .2s, box-shadow .2s, background .2s;
}
.node .ico  { font-size: 1.7rem; line-height: 1; }
.node .lbl  { font-size: 0.79rem; font-weight: 600; margin-top: 3px; }
.node .sub  { font-size: 0.64rem; color: var(--muted); font-family: monospace; }

/* timer strip */
.node .tmr {
  margin-top: 6px; padding: 3px 8px;
  background: rgba(255,255,255,.04); border-radius: 6px;
  font-size: 0.7rem; font-family: monospace; color: var(--muted);
  min-width: 60px; text-align: center;
  transition: color .2s;
}

/* status dot */
.node .dot {
  position: absolute; top: 7px; right: 9px;
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--muted); opacity: .3;
  transition: background .2s, opacity .2s;
}

/* ── active ── */
.node.active {
  border-color: var(--active); background: #1a1200;
  animation: gpulse 1.3s ease-in-out infinite;
}
.node.active .lbl  { color: var(--active); }
.node.active .dot  { background: var(--active); opacity: 1; }
.node.active .tmr  { color: var(--active); background: rgba(245,158,11,.1); }

/* ── done ── */
.node.done { border-color: var(--done); background: #071310; }
.node.done .lbl { color: var(--done); }
.node.done .dot { background: var(--done); opacity: 1; }
.node.done .tmr { color: var(--done); background: rgba(16,185,129,.08); }

/* ── skipped ── */
.node.skip { opacity: .2; }

@keyframes gpulse {
  0%,100% { box-shadow: 0 0 14px 2px rgba(245,158,11,.28), 0 0 32px 4px rgba(245,158,11,.09); }
  50%      { box-shadow: 0 0 26px 6px rgba(245,158,11,.55), 0 0 54px 10px rgba(245,158,11,.18); }
}

/* ── Status bar ── */
.sbar { max-width: 880px; display: flex; align-items: center; gap: 8px; margin: 14px 0 10px; }
#sdot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); flex-shrink: 0; }
#sdot.run  { background: var(--active); animation: blink 1s infinite; }
#sdot.done { background: var(--done); }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.25} }
#stxt { font-size: 0.78rem; color: var(--muted); }

/* ── Response ── */
.rbox { max-width: 880px; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
.rbox .rtitle { font-size: 0.7rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin-bottom: 10px; }
#ans { font-size: 0.87rem; line-height: 1.65; white-space: pre-wrap; color: var(--text); min-height: 36px; }
</style>
</head>
<body>

<!-- Header -->
<div class="hdr">
  <h1>⚖️ Legal Multi-Agent System <em>Live Visualizer</em></h1>
  <div class="irow">
    <input id="q" type="text"
      value="A tech startup shared user data without consent and avoided taxes on overseas revenue. What are the legal consequences?" />
    <button id="btn" onclick="run()">▶ Run</button>
  </div>
</div>

<!-- Architecture info bar -->
<div class="arch-bar">
  <div class="arch-chip">
    <span class="k">🏗 Pattern</span>
    <span class="v sw">Supervisor-Workers</span>
  </div>
  <span class="arch-sep">|</span>
  <div class="arch-chip">
    <span class="k">🔗 Transport</span>
    <span class="v">In-Process (direct calls)</span>
  </div>
  <span class="arch-sep">|</span>
  <div class="arch-chip">
    <span class="k">⚡ Parallelism</span>
    <span class="v">LangGraph Send API</span>
  </div>
  <span class="arch-sep">|</span>
  <div class="arch-chip">
    <span class="k">🌐 Stage 5 uses</span>
    <span class="v a2a">A2A Protocol (HTTP)</span>
  </div>
</div>

<!-- Graph -->
<div id="gwrap">
  <svg id="svg-ov"></svg>

  <div class="node" id="n-analyze_law">
    <div class="dot"></div>
    <div class="ico">⚖️</div>
    <div class="lbl">Lead Attorney</div>
    <div class="sub">analyze_law</div>
    <div class="tmr" id="tm-analyze_law">—</div>
  </div>

  <div class="gap"></div>

  <div class="node" id="n-check_routing">
    <div class="dot"></div>
    <div class="ico">🔀</div>
    <div class="lbl">Router</div>
    <div class="sub">check_routing</div>
    <div class="tmr" id="tm-check_routing">—</div>
  </div>

  <div class="gap"></div>

  <div class="prow">
    <div class="node" id="n-call_tax">
      <div class="dot"></div>
      <div class="ico">💰</div>
      <div class="lbl">Tax Agent</div>
      <div class="sub">call_tax</div>
      <div class="tmr" id="tm-call_tax">—</div>
    </div>
    <div class="node" id="n-call_compliance">
      <div class="dot"></div>
      <div class="ico">✅</div>
      <div class="lbl">Compliance</div>
      <div class="sub">call_compliance</div>
      <div class="tmr" id="tm-call_compliance">—</div>
    </div>
    <div class="node" id="n-call_privacy">
      <div class="dot"></div>
      <div class="ico">🔒</div>
      <div class="lbl">Privacy</div>
      <div class="sub">call_privacy</div>
      <div class="tmr" id="tm-call_privacy">—</div>
    </div>
  </div>

  <div class="gap"></div>

  <div class="node" id="n-aggregate">
    <div class="dot"></div>
    <div class="ico">📋</div>
    <div class="lbl">Aggregator</div>
    <div class="sub">aggregate</div>
    <div class="tmr" id="tm-aggregate">—</div>
  </div>
</div>

<!-- Status -->
<div class="sbar">
  <div id="sdot"></div>
  <div id="stxt">Ready — enter a question and click ▶ Run</div>
</div>

<!-- Response -->
<div class="rbox">
  <div class="rtitle">📨 Response</div>
  <pre id="ans">…</pre>
</div>

<script>
const ALL = ['analyze_law','check_routing','call_tax','call_compliance','call_privacy','aggregate'];
const EDGES = [
  ['analyze_law','check_routing'],
  ['check_routing','call_tax'],
  ['check_routing','call_compliance'],
  ['check_routing','call_privacy'],
  ['call_tax','aggregate'],
  ['call_compliance','aggregate'],
  ['call_privacy','aggregate'],
];

const st = {};          // node id → 'active' | 'done' | 'skip' | ''
const _t0 = {};         // node id → Date.now() when started
let _ticker = null;     // setInterval handle for live timers

// ── Node state ────────────────────────────────────────────────

function nEl(id) { return document.getElementById('n-' + id); }
function tmEl(id) { return document.getElementById('tm-' + id); }

function setNode(id, state) {
  st[id] = state;
  const el = nEl(id);
  if (!el) return;
  el.classList.remove('active','done','skip');
  if (state) el.classList.add(state);
  redrawEdges();
}

function setStatus(msg, mode) {
  document.getElementById('stxt').textContent = msg;
  document.getElementById('sdot').className = mode || '';
}

// ── Timers ────────────────────────────────────────────────────

function startTimer(id) {
  _t0[id] = Date.now();
  if (!_ticker) _ticker = setInterval(tickAll, 100);
}

function tickAll() {
  const now = Date.now();
  let any = false;
  for (const [id, t0] of Object.entries(_t0)) {
    const el = tmEl(id);
    if (el) el.textContent = ((now - t0) / 1000).toFixed(1) + 's';
    any = true;
  }
  if (!any) { clearInterval(_ticker); _ticker = null; }
}

function stopTimer(id, elapsed) {
  delete _t0[id];
  const el = tmEl(id);
  if (el) el.textContent = elapsed.toFixed(1) + 's ✓';
  if (Object.keys(_t0).length === 0 && _ticker) {
    clearInterval(_ticker); _ticker = null;
  }
}

function resetTimers() {
  for (const id of Object.keys(_t0)) delete _t0[id];
  if (_ticker) { clearInterval(_ticker); _ticker = null; }
  ALL.forEach(id => { const el = tmEl(id); if (el) el.textContent = '—'; });
}

// ── SVG edges ─────────────────────────────────────────────────

function mid(id) {
  const el = nEl(id);
  const wrap = document.getElementById('gwrap');
  if (!el) return null;
  const r = el.getBoundingClientRect(), w = wrap.getBoundingClientRect();
  return { cx: r.left - w.left + r.width/2, top: r.top - w.top, bot: r.top - w.top + r.height };
}

function edgeColor(from, to) {
  const fs = st[from]||'', ts = st[to]||'';
  if (fs==='skip' || ts==='skip') return 'transparent';
  if (fs==='active') return '#f59e0b';
  if (fs==='done' && (ts==='done'||ts==='active')) return '#10b981';
  if (fs==='done') return '#1d3b2e';
  return '#1e2235';
}

function redrawEdges() {
  const svg = document.getElementById('svg-ov');
  const wrap = document.getElementById('gwrap');
  svg.setAttribute('width', wrap.offsetWidth);
  svg.setAttribute('height', wrap.offsetHeight);
  svg.innerHTML = '';
  const ns = 'http://www.w3.org/2000/svg';
  for (const [from, to] of EDGES) {
    const f = mid(from), t = mid(to);
    if (!f || !t) continue;
    const c = edgeColor(from, to);
    if (c === 'transparent') continue;
    const my = (f.bot + t.top) / 2;
    const path = document.createElementNS(ns, 'path');
    path.setAttribute('d', `M${f.cx},${f.bot} C${f.cx},${my} ${t.cx},${my} ${t.cx},${t.top}`);
    path.setAttribute('stroke', c);
    path.setAttribute('stroke-width', '2');
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke-linecap', 'round');
    svg.appendChild(path);
    // arrowhead
    const aw=5, ah=8;
    const arr = document.createElementNS(ns, 'polygon');
    arr.setAttribute('points', `${t.cx},${t.top} ${t.cx-aw},${t.top-ah} ${t.cx+aw},${t.top-ah}`);
    arr.setAttribute('fill', c);
    svg.appendChild(arr);
  }
}

window.addEventListener('load', redrawEdges);
window.addEventListener('resize', redrawEdges);

// ── Events ────────────────────────────────────────────────────

function handleEvent(ev) {
  if (ev.type === 'ping') return;   // keepalive — ignore

  if (ev.type === 'node_start') {
    setNode(ev.node, 'active');
    startTimer(ev.node);
    setStatus(ev.node + ' đang chạy…', 'run');

  } else if (ev.type === 'node_end') {
    setNode(ev.node, 'done');
    stopTimer(ev.node, ev.elapsed ?? 0);

  } else if (ev.type === 'routing') {
    if (!ev.needs_tax)        setNode('call_tax',        'skip');
    if (!ev.needs_compliance) setNode('call_compliance', 'skip');
    if (!ev.needs_privacy)    setNode('call_privacy',    'skip');

  } else if (ev.type === 'done') {
    document.getElementById('ans').textContent = ev.answer;
    setStatus('Hoàn thành ✓', 'done');
    redrawEdges();

  } else if (ev.type === 'error') {
    document.getElementById('ans').textContent = '❌ ' + ev.message;
    setStatus('Lỗi: ' + ev.message, '');
  }
}

// ── Run ───────────────────────────────────────────────────────

function reset() {
  ALL.forEach(n => setNode(n, ''));
  resetTimers();
  document.getElementById('ans').textContent = '';
  setStatus('Đang kết nối…', 'run');
}

async function run() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  const btn = document.getElementById('btn');
  btn.disabled = true;
  reset();

  try {
    const resp = await fetch('/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q }),
    });

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try { handleEvent(JSON.parse(line.slice(6))); } catch {}
        }
      }
    }
  } catch (err) {
    setStatus('Lỗi kết nối: ' + err.message, '');
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("Open http://localhost:8080 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
