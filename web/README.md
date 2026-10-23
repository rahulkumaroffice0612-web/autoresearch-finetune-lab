# LazyDecider demo UI

A self-contained static site that tells the LazyDecider story: the overnight Gemma→ASHA→Tinker loop
(replayed from the real logs), the learning-curve + results charts, the sealed-test table, the Daniel
narration, and links. **No backend, no build step.**

## Files

```
web/
  index.html     structure
  style.css      dark "night dashboard" theme
  app.js         renders stats / replay / table / links from window.LAZY
  data.js        REAL numbers, baked by src/build_demo_data.py (do not hand-edit)
  assets/        charts + cover + narration.mp3 (copied by the builder)
```

## Refresh the data

After any experiment change, regenerate `data.js` and re-copy assets in one command:

```bash
.venv/bin/python src/build_demo_data.py
```

## Run locally

```bash
cd web && python3 -m http.server 8777
# open http://127.0.0.1:8777
```

## Host it

**Option 1 — Render (free static site).** The repo root has `render.yaml`. On Render: *New → Blueprint
→ pick this repo*. It serves `lazylab/web` with no build step. Done in seconds.

**Option 2 — your free `.tech` domain (or any static host).** The `web/` folder is fully
self-contained — just upload its contents (`index.html`, `style.css`, `app.js`, `data.js`, `assets/`)
to the host's web root. Works on Netlify drop, GitHub Pages, Cloudflare Pages, S3, or a plain `.tech`
domain's `public_html`.
