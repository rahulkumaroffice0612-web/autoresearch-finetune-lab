"""Record the web/ dashboard LIVE with Playwright: real page, synthetic cursor,
smooth scrolling paced to the six narration beats, with the overnight-loop
animation playing for real as it scrolls into view.

Produces a silent webm (video only); the caller muxes narration.mp3 and appends
the live playground clip. Pacing matches artifacts/demo_audio/scene_*.mp3.

Usage:
  .venv/bin/python src/record_dashboard.py --port 8808 --out artifacts/demo_video/_dash.webm
"""
from __future__ import annotations
import argparse, time, math
from pathlib import Path
from playwright.sync_api import sync_playwright

# Narration scene durations (s) — must match artifacts/demo_audio/scene_*.mp3.
SCENES = [9.96, 11.93, 18.20, 12.46, 10.53]  # scene 6 (playground) is the appended live clip

CURSOR_JS = r"""
(() => {
  const c = document.createElement('div');
  c.id='__cur';
  c.style.cssText='position:fixed;z-index:999999;width:20px;height:20px;margin:-10px 0 0 -10px;'
    +'border-radius:50%;background:rgba(45,212,191,.30);border:2px solid #2dd4bf;'
    +'box-shadow:0 0 10px rgba(45,212,191,.7);pointer-events:none;left:960px;top:150px;';
  document.body.appendChild(c);
  window.__moveCur=(x,y)=>{c.style.left=x+'px';c.style.top=y+'px';};
})();
"""


def ease(t):  # smoothstep
    return t * t * (3 - 2 * t)


def scroll_to(page, y, duration, fps=30, cursor_y=150):
    """Smoothly scroll window to absolute y over `duration` seconds."""
    start = page.evaluate("() => window.scrollY")
    frames = max(1, int(duration * fps))
    for i in range(1, frames + 1):
        f = ease(i / frames)
        ny = start + (y - start) * f
        page.evaluate("(y)=>window.scrollTo(0,y)", ny)
        # drift the cursor gently so the frame feels alive
        cx = 760 + 120 * math.sin(i / frames * math.pi)
        page.evaluate("([x,y])=>window.__moveCur(x,y)", [cx, cursor_y])
        time.sleep(1 / fps)


def hold(page, duration, fps=30):
    for _ in range(int(duration * fps)):
        time.sleep(1 / fps)


def run(port: int, out: Path):
    url = f"http://127.0.0.1:{port}/"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / "_pw_dash"
    tmp.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 720},
            record_video_dir=str(tmp),
            record_video_size={"width": 1280, "height": 720},
            device_scale_factor=1,
        )
        page = ctx.new_page()
        page.goto(url, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(500)
        page.evaluate(CURSOR_JS)

        # measure section tops so scroll targets land on real content
        tops = page.evaluate(
            """() => {
              const q = s => { const el=document.querySelector(s); return el ? Math.max(0, el.getBoundingClientRect().top + window.scrollY - 40) : 0; };
              return {
                hero: 0,
                loop: q('.panel'),                 // first panel = overnight loop
                generalize: q('#generalize'),
                charts: q('#generalize ~ .panel'), // charts panel after generalize
                table: q('#resultsTable') ? q('#resultsTable') - 80 : 0,
                guard: q('.guardrails'),
                max: document.body.scrollHeight - window.innerHeight
              };
            }"""
        )
        # Kick the loop animation (app.js auto-plays armA on load; replay to be safe when it scrolls in)
        page.wait_for_timeout(300)

        # Scene 1 (hero + stats): start at top, tiny drift
        hold(page, 1.2)
        scroll_to(page, min(tops["loop"] * 0.35, tops["max"]), SCENES[0] - 1.2, cursor_y=150)

        # Scene 2 (overnight loop appears): scroll to the loop panel; animation reveals runs
        page.evaluate("() => { const b=document.getElementById('replayBtn'); if(b) b.click(); }")
        scroll_to(page, min(tops["loop"], tops["max"]), SCENES[1], cursor_y=200)

        # Scene 3 (keep/revert): dwell on the loop runs (KEEP/REVERT), only part-way down
        scroll_to(page, min(tops["loop"] + (tops["generalize"] - tops["loop"]) * 0.45, tops["max"]), SCENES[2], cursor_y=260)

        # Scene 4 (report / generalization numbers): land squarely on the invoice case card
        scroll_to(page, min(tops["generalize"] + 120, tops["max"]), SCENES[3], cursor_y=240)

        # Scene 5 (charts + table): scroll through charts and the results table
        scroll_to(page, min(tops["table"], tops["max"]), SCENES[4] * 0.6, cursor_y=240)
        scroll_to(page, min(tops["guard"], tops["max"]), SCENES[4] * 0.4, cursor_y=240)

        ctx.close()
        browser.close()

    vids = sorted(tmp.glob("*.webm"), key=lambda f: f.stat().st_mtime)
    if vids:
        vids[-1].replace(out)
        print(f"[dash] saved {out} ({out.stat().st_size} bytes)")
    else:
        print("[dash] WARN: no video produced")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8808)
    ap.add_argument("--out", type=Path, default=Path("artifacts/demo_video/_dash.webm"))
    args = ap.parse_args()
    run(args.port, args.out)
