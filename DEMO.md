# Demo — recording script & storyboard

A **~90–120 second** screen capture that shows LazyDecider's overnight loop, the morning report, and a
**live playground run**. The loop/report sections replay the **real** experiment logs (honest,
reproducible); the playground runs the **real model** on a custom invoice.

## What to capture (target ~90–120s)

| # | Screen | Say / show | ~time |
|---|---|---|---|
| 1 | Terminal, repo root | "LazyDecider: an open-weight agent fine-tunes a model while I sleep." Show `ls` of `src/`. | 0:00–0:08 |
| 2 | Run the replay | `./.venv/bin/python src/demo.py --arm A` — Gemma proposes run 1 (head-only baseline) with its one-line reason. | 0:08–0:25 |
| 3 | Continue | Run 2 unfreezes the backbone (LoRA) → eval loss falls 1.04 → 0.84 → **KEEP (new best)**. Runs 3–5 chase calibration and get **REVERTED**. | 0:25–0:48 |
| 4 | Morning report | The loop prints the winner: acc **0.770**, KL **0.094**; Tinker confirm acc 0.601 vs 0.483 zero-shot. | 0:48–0:58 |
| 5 | Demo dashboard / charts | Show the `web/` dashboard: learning curves, results table, the per-domain generalization (invoice 0.840). | 0:58–1:12 |
| 6 | **Playground (the payoff)** | `bash src/run_playground.sh` → open `http://127.0.0.1:7860`. Load the "Clean match" preset → **Decide** → `approve · 83%`. Then "Under-delivered −50%" → **manual review**. "The real model, on your own invoice." | 1:12–1:45 |
| 7 | Browser (optional) | Flash the GitHub repo + the Hugging Face model page. | 1:45–2:00 |

> Tip: start the playground **before** recording (so the model is already preloaded and the first
> **Decide** is instant), then begin the capture.

## Exact commands to run on camera

```bash
cd lazylab

# 0. (before recording) start the playground so the model is preloaded
bash src/run_playground.sh          # leave running; open http://127.0.0.1:7860

# 1. the overnight loop, replayed from the real logs (readable pace)
./.venv/bin/python src/demo.py --arm A

# 2. show the morning report, charts, and the web dashboard
open REPORT.md
open artifacts/results_comparison.png artifacts/gemma_learning_curves.png
(cd web && python3 -m http.server 8777) &   # http://127.0.0.1:8777

# 3. the payoff: in the playground, load a preset -> Decide -> show the decision
```

If you want to re-record quickly without the typing delay, add `--fast`.

## Live variant (optional, if the local server is up)

To capture a genuinely live Gemma proposal instead of a replay, start the local server first, then run
a short real search (writes new `source: gemma` rows):

```bash
# in LM Studio / Bionic: start the server on :1235 and load gemma-4-e4b-it-mlx
bash src/run_gemma_search.sh 5      # 5-minute live search
```

Note: the live search actually trains, so it's slower and noisier on camera — the replay in `demo.py`
is the cleaner story for a short video. Both draw from the same logs and the same scorer.

## Recording tips

- Use a dark terminal theme with a large font (18pt+) — the demo uses colour (yellow = Gemma's
  proposal, green = KEEP, red = REVERT) that reads well on dark backgrounds.
- Record at 1280×720 or 1920×1080; keep the terminal roughly 100 columns wide so lines don't wrap.
- `asciinema rec demo.cast` is a good lightweight option if you'd rather embed a terminal cast than
  an MP4; otherwise QuickTime screen recording → export to the DEV post.

## Voiceover (ElevenLabs)

### Option A — free, offline (macOS `say`, no account) ✅ used for the committed audio

```bash
.venv/bin/python src/make_voiceover_local.py                 # all scenes -> artifacts/demo_audio/
.venv/bin/python src/make_voiceover_local.py --voice Daniel  # British male voice
.venv/bin/python src/make_voiceover_local.py --scene 3       # re-do one scene
```

Needs only the built-in macOS `say` and `ffmpeg` (`brew install ffmpeg`). Produces `scene_N.mp3`
for each scene plus a concatenated `narration.mp3` (~72s, voice = Daniel, British male). This is the
path used for the audio checked into the repo — no account, no cost. For a more natural voice,
install a "Premium/Enhanced" one in *System Settings → Accessibility → Spoken Content → Manage
Voices* and pass it with `--voice`.

### Option B — ElevenLabs SDK (higher quality, needs a paid/credited account)

```bash
npm install                                      # one-time: installs @elevenlabs/elevenlabs-js
node src/make_voiceover.mjs                       # all scenes -> artifacts/demo_audio/
node src/make_voiceover.mjs --scene 3             # re-do one scene
node src/make_voiceover.mjs --voice <id>          # use a voice you own
```

Reads the key from `../.env` (`ELEVENLABS_API_KEY`, never committed). Note: a *free* ElevenLabs tier
can't synthesize library voices via the API (HTTP 402) — add API credits or use an owned voice. Until
then, Option A is the working path.

## Interactive playground (`src/playground.py`)

Run the **real model** on a custom invoice — a local Gradio app with form inputs and preset examples.
On Apple Silicon it loads the MLX build in `artifacts/best_model` (load ~11s, predict ~0.5s).

```bash
uv pip install --python .venv/bin/python gradio
.venv/bin/python src/playground.py            # http://127.0.0.1:7860
.venv/bin/python src/playground.py --share     # also a temporary public gradio.live link
```

Returns calibrated probabilities for all five accounts-payable decisions (disposition, matches_order,
discrepancy_severity, duplicate, urgency). The `--share` link is handy for recording the demo without
screen-capturing your own machine. Note: this is the Apple-Silicon MLX build; for x86/Linux hosting
you'd first convert to portable transformers weights (the unfinished N12 step).

## Auto-generated demo video (no screen recording needed)

A committed **~72s narrated MP4** at `artifacts/demo_video/lazydecider_demo.mp4` (1280×720, H.264+AAC)
is assembled entirely from **real artifacts** — the `web/` dashboard (overnight loop, keep/revert,
morning report, charts) for scenes 1–5, and the **live playground's real inference output** for the
scene-6 payoff (`approve · 83%` → the under/over cases). It is muxed to the committed
`artifacts/demo_audio/narration.mp3`, so the visuals track the six narration beats.

How it's produced (headless Chrome screenshots → ffmpeg, all local, no screen capture):

1. Launch the playground, wait for `model ready`, capture a headless screenshot of the live UI and
   drive `/predict` via `gradio_client` for the three invoice presets (the real distributions).
2. Render the `web/` dashboard (served locally) to a tall PNG and crop one 1280×720 window per scene.
3. Render the playground result as a themed card per preset from the captured `results.json`.
4. For each scene, build a still clip held for that scene's narration duration (with in/out fades),
   `concat` them, then mux `narration.mp3`.

`poster.png` beside the MP4 is a title-frame thumbnail for embedding. To re-record live instead, the
QuickTime storyboard above still applies.

## Live playground recording (real cursor, real inference)

`artifacts/demo_video/playground_live.mp4` (~21s) is a genuine recording of the **live Gradio
playground** being driven end to end — a synthetic cursor glides to the preset dropdown, loads the
"Clean match" invoice, presses **Decide**, and the real 0.8B model returns `approve · 83%`; then it
loads the "Under-delivered −50%" case and the model returns `manual review · 44%`. Nothing is faked:
the form values and the probability table are the model's actual output. This clip is also appended to
the end of the main `lazydecider_demo.mp4` after a short "Try it yourself" card.

It's produced headlessly with Playwright (no screen capture needed):

```bash
uv pip install --python .venv/bin/python playwright==1.63.0
.venv/bin/python -m playwright install chromium
# start the playground, wait for 'model ready', then:
.venv/bin/python src/record_playground.py --port 7870 --out artifacts/demo_video/playground_live.webm
```

`src/record_playground.py` launches Chromium with Playwright's video recorder, injects a visible
cursor + click ripple, and clicks the real controls, waiting for the real model response between steps.

## Where it goes
Embed the final video (`artifacts/demo_video/lazydecider_demo.mp4`) or an asciinema cast in `POST.md`
at the **Demo** line, alongside the repo and Hugging Face links.
