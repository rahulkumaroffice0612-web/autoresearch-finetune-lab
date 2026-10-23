# Demo — recording script & storyboard

A ~90-second screen capture that shows LazyDecider's overnight loop and the morning report.
Everything here replays the **real** experiment logs, so the demo is honest and reproducible — no
staged numbers, no live-server dependency during recording.

## What to capture (target ~60–90s)

| # | Screen | Say / show | ~time |
|---|---|---|---|
| 1 | Terminal, repo root | "LazyDecider: an open-weight agent fine-tunes a model while I sleep." Show `ls` of `src/`. | 0:00–0:08 |
| 2 | Run the replay | `./.venv/bin/python src/demo.py --arm A` — Gemma proposes run 1 (head-only baseline) with its one-line reason. | 0:08–0:25 |
| 3 | Continue | Run 2 unfreezes the backbone (LoRA) → eval loss falls 1.04 → 0.84 → **KEEP (new best)**. Runs 3–5 chase calibration and get **REVERTED**. | 0:25–0:50 |
| 4 | Morning report | The loop prints the winner: acc **0.770**, KL **0.094**; Tinker confirm acc 0.601 vs 0.483 zero-shot. | 0:50–1:00 |
| 5 | Open `REPORT.md` + charts | Show `artifacts/results_comparison.png` and `artifacts/gemma_learning_curves.png`. | 1:00–1:15 |
| 6 | Browser (optional) | Flash the GitHub repo + the Hugging Face model page. | 1:15–1:30 |

## Exact commands to run on camera

```bash
cd lazylab

# 1. the overnight loop, replayed from the real logs (readable pace)
./.venv/bin/python src/demo.py --arm A

# 2. (optional) show the morning report and charts
open REPORT.md
open artifacts/results_comparison.png artifacts/gemma_learning_curves.png
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

The narration first draft is in `artifacts/demo_voiceover.md` — six blocks, one per scene above,
~196 spoken words (~78s, fits the 90s target). Generate the audio with:

```bash
.venv/bin/python src/make_voiceover.py                 # all scenes -> artifacts/demo_audio/
.venv/bin/python src/make_voiceover.py --scene 3       # re-do one scene
.venv/bin/python src/make_voiceover.py --voice <id>    # use a voice you own
```

This writes `artifacts/demo_audio/scene_N.mp3` (drop each under its matching screen in the editor)
plus a concatenated `narration.mp3`. The key is read from `../.env` (`ELEVENLABS_API_KEY`, never
committed).

> **Heads-up on the current key:** it's a *free* ElevenLabs tier, which can't synthesize the premade
> library voices via the API (HTTP 402). To produce audio, either add $5 of API credits, or create a
> personal/cloned voice in the ElevenLabs app and pass it with `--voice <id>`. The script prints this
> exact guidance if it hits the limit.

## Where it goes

Embed the final video (or asciinema cast) in `POST.md` at the **Demo** line, alongside the repo and
Hugging Face links.
