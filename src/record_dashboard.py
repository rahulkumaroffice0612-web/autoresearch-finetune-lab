"""Record the web/ dashboard LIVE with Playwright, ONE VIDEO PER SCENE so each
scene can be retimed to its exact narration duration (frame-accurate sync).

Playwright's webm runs at a variable framerate, so a single global retime drifts
the later scenes. Recording each scene in its own browser context yields one webm
per scene; the caller retimes each to SCENES[i] and concatenates. The page scroll
state is restored between scenes so motion stays continuous.

Usage:
  .venv/bin/python src/record_dashboard.py --port 8808 --outdir artifacts/demo_video/_scenes
"""
from __future__ import annotations
import argparse, time, math, json
from pathlib import Path
from playwright.sync_api import sync_playwright

# Narration scene durations (s) — must match artifacts/demo_audio/scene_*.mp3.
SCENES = [9.96, 11.93, 18.20, 12.46, 10.53]  # scene 6 (playground) is the appended live clip

CURSOR_JS = r"""
(() => {
  if (document.getElementById('__cur')) return;
  const c = document.createElement('div');
  c.id='__cur';
  c.style.cssText='position:fixed;z-index:999999;width:20px;height:20px;margin:-10px 0 0 -10px;'
    +'border-radius:50%;background:rgba(45,212,191,.30);border:2px solid #2dd4bf;'
    +'box-shadow:0 0 10px rgba(45,212,191,.7);pointer-events:none;left:760px;top:200px;';
  document.body.appendChild(c);
  window.__moveCur=(x,y)=>{c.style.left=x+'px';c.style.top=y+'px';};
})();
"""


def ease(t):
    return t * t * (3 - 2 * t)


def tops_of(page):
    return page.evaluate(
        """() => {
          const q = s => { const el=document.querySelector(s); return el ? Math.max(0, el.getBoundingClientRect().top + window.scrollY - 40) : 0; };
          return {
            loop: q('.panel'),
            generalize: q('#generalize'),
            table: (document.querySelector('#resultsTable') ? q('#resultsTable') - 80 : 0),
            guard: q('.guardrails'),
            max: document.body.scrollHeight - window.innerHeight
          };
        }"""
    )


def animate(page, y_from, y_to, duration, cursor_y, fps=30):
    """Scroll from y_from to y_to over duration seconds, drifting the cursor."""
    page.evaluate("(y)=>window.scrollTo(0,y)", y_from)
    frames = max(1, int(duration * fps))
    for i in range(1, frames + 1):
        f = ease(i / frames)
        ny = y_from + (y_to - y_from) * f
        page.evaluate("(y)=>window.scrollTo(0,y)", ny)
        cx = 760 + 110 * math.sin(i / frames * math.pi)
        page.evaluate("([x,y])=>window.__moveCur(x,y)", [cx, cursor_y])
        time.sleep(1 / fps)


def record_scene(browser, url, tmpdir, idx, plan_fn):
    """Record one scene into its own context => one webm. plan_fn(page, tops) does the motion."""
    ctx = browser.new_context(
        viewport={"width": 1280, "height": 720},
        record_video_dir=str(tmpdir),
        record_video_size={"width": 1280, "height": 720},
        device_scale_factor=1,
    )
    page = ctx.new_page()
    page.goto(url, wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(400)
    page.evaluate(CURSOR_JS)
    tops = tops_of(page)
    plan_fn(page, tops)
    vid = page.video
    ctx.close()  # finalizes the recording
    dst = tmpdir / f"scene{idx}.webm"
    vid.save_as(str(dst))  # blocks until the webm is fully written
    print(f"[dash] scene {idx} -> {dst.name} ({dst.stat().st_size} bytes)")
    return dst


def run(port: int, outdir: Path):
    url = f"http://127.0.0.1:{port}/"
    outdir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        def plan1(page, tops):  # hero + stats, gentle downward drift
            page.wait_for_timeout(300)
            animate(page, 0, int(tops["loop"] * 0.30), SCENES[0], cursor_y=160)

        def plan2(page, tops):  # overnight loop appears, runs animate in
            page.evaluate("() => { const b=document.getElementById('replayBtn'); if(b) b.click(); }")
            animate(page, int(tops["loop"] * 0.30), int(tops["loop"]), SCENES[1], cursor_y=200)

        def plan3(page, tops):  # keep/revert: scroll through the five runs
            page.evaluate("() => { const b=document.getElementById('replayBtn'); if(b) b.click(); }")
            y0 = int(tops["loop"])
            y1 = int(tops["loop"] + (tops["generalize"] - tops["loop"]) * 0.62)
            animate(page, y0, y1, SCENES[2], cursor_y=260)

        def plan4(page, tops):  # report / generalization card
            y0 = int(tops["loop"] + (tops["generalize"] - tops["loop"]) * 0.62)
            y1 = int(min(tops["generalize"] + 150, tops["max"]))
            animate(page, y0, y1, SCENES[3], cursor_y=240)

        def plan5(page, tops):  # charts + results table
            y0 = int(min(tops["generalize"] + 150, tops["max"]))
            y1 = int(min(tops["guard"], tops["max"]))
            animate(page, y0, y1, SCENES[4], cursor_y=240)

        plans = [plan1, plan2, plan3, plan4, plan5]
        for i, fn in enumerate(plans, start=1):
            record_scene(browser, url, outdir, i, fn)

        browser.close()
    print(f"[dash] all scenes saved in {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8808)
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/demo_video/_scenes"))
    args = ap.parse_args()
    run(args.port, args.outdir)
