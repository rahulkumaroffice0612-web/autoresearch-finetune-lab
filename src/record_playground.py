"""Record a 'live' playground interaction with Playwright: a synthetic cursor
moves across the real Gradio UI, picks presets, clicks Decide, and waits for the
REAL model output — then Playwright saves it as a video. No screen capture; this
drives the actual running app at http://127.0.0.1:<port>.

Usage:
  .venv/bin/python src/record_playground.py --port 7870 --out artifacts/demo_video/playground_live.webm
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
from playwright.sync_api import sync_playwright

# A small JS cursor we inject so the recording shows a moving pointer + click ripple.
CURSOR_JS = r"""
(() => {
  const c = document.createElement('div');
  c.id = '__cur';
  c.style.cssText = 'position:fixed;z-index:999999;width:22px;height:22px;margin:-11px 0 0 -11px;'
    + 'border-radius:50%;background:rgba(45,212,191,.35);border:2px solid #2dd4bf;'
    + 'box-shadow:0 0 10px rgba(45,212,191,.8);pointer-events:none;transition:transform .05s;'
    + 'left:640px;top:360px;';
  document.body.appendChild(c);
  window.__moveCur = (x,y) => { c.style.left = x+'px'; c.style.top = y+'px'; };
  window.__clickCur = () => {
    const r = document.createElement('div');
    r.style.cssText = 'position:fixed;z-index:999998;width:10px;height:10px;margin:-5px 0 0 -5px;'
      + 'border-radius:50%;border:2px solid #2dd4bf;pointer-events:none;'
      + 'left:'+c.style.left+';top:'+c.style.top+';opacity:1;transition:all .45s ease-out;';
    document.body.appendChild(r);
    requestAnimationFrame(()=>{ r.style.width='56px'; r.style.height='56px'; r.style.margin='-28px 0 0 -28px'; r.style.opacity='0'; });
    setTimeout(()=>r.remove(), 500);
  };
})();
"""


def glide(page, x, y, steps=28, pause=0.012):
    """Move the synthetic cursor smoothly to (x,y)."""
    cur = page.evaluate("() => { const c=document.getElementById('__cur'); return [parseFloat(c.style.left), parseFloat(c.style.top)]; }")
    x0, y0 = cur
    for i in range(1, steps + 1):
        nx = x0 + (x - x0) * i / steps
        ny = y0 + (y - y0) * i / steps
        page.evaluate("([x,y])=>window.__moveCur(x,y)", [nx, ny])
        page.mouse.move(nx, ny)
        time.sleep(pause)


def click_at(page, locator):
    box = locator.bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    glide(page, cx, cy)
    page.evaluate("()=>window.__clickCur()")
    time.sleep(0.15)
    locator.click()
    time.sleep(0.3)


def run(port: int, out: Path):
    url = f"http://127.0.0.1:{port}/"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmpdir = out.parent / "_pw_video"
    tmpdir.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 900},
            record_video_dir=str(tmpdir),
            record_video_size={"width": 1280, "height": 900},
            device_scale_factor=1,
        )
        page = ctx.new_page()
        page.goto(url, wait_until="networkidle", timeout=60000)
        # keep everything in one viewport: nudge Gradio's layout tighter
        page.add_style_tag(content="footer{display:none!important} .gradio-container{padding-top:8px!important}")
        page.wait_for_timeout(800)
        page.evaluate(CURSOR_JS)
        page.wait_for_timeout(600)

        # 1) open the preset dropdown and choose the clean match
        preset = page.get_by_label("Load an example")
        click_at(page, preset)
        page.wait_for_timeout(400)
        clean = page.get_by_role("option", name="Clean match (should approve)")
        click_at(page, clean)
        page.wait_for_timeout(700)

        # 2) press Decide and wait for the real model output
        decide = page.get_by_role("button", name="Decide")
        click_at(page, decide)
        # wait until the headline shows a Decision (real inference completes)
        page.wait_for_function(
            "() => document.body.innerText.includes('Decision:')", timeout=60000
        )
        page.evaluate("() => window.scrollTo({top:0, behavior:'smooth'})")
        page.wait_for_timeout(600)
        glide(page, 760, 200)  # move cursor up near the decision headline
        page.wait_for_timeout(2800)  # let viewers read the approve result

        # 3) switch to the under-delivered discrepancy case
        click_at(page, preset)
        page.wait_for_timeout(400)
        under = page.get_by_role("option", name="Under-delivered -50% (discrepancy)")
        click_at(page, under)
        page.wait_for_timeout(600)
        click_at(page, decide)
        page.wait_for_function(
            "() => document.body.innerText.includes('manual review') || document.body.innerText.includes('Decision:')",
            timeout=60000,
        )
        page.evaluate("() => window.scrollTo({top:0, behavior:'smooth'})")
        page.wait_for_timeout(600)
        glide(page, 760, 200)  # point at the manual-review decision
        page.wait_for_timeout(3000)

        ctx.close()  # finalizes the video
        browser.close()

    # move/rename the produced webm to the requested path
    vids = sorted(tmpdir.glob("*.webm"), key=lambda f: f.stat().st_mtime)
    if vids:
        vids[-1].replace(out)
        print(f"[record] saved {out} ({out.stat().st_size} bytes)")
    else:
        print("[record] WARN: no video produced")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7870)
    ap.add_argument("--out", type=Path, default=Path("artifacts/demo_video/playground_live.webm"))
    args = ap.parse_args()
    run(args.port, args.out)
