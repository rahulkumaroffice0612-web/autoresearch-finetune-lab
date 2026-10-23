# Demo voiceover — first draft

Narration for the ~90-second LazyDecider demo. Timed to the six beats in `DEMO.md`.
Spoken-length target ~150–165 words (≈22s of slack for pauses/typing). Voice: calm, a little playful.

Generate the audio with `src/make_voiceover.py` (ElevenLabs). Each `## Scene N` below maps to one
`.mp3` in `artifacts/demo_audio/` that you drop under the matching screen in your editor.

---

## Scene 1 — title (0:00–0:08)

> This is LazyDecider. The idea is simple: I go to sleep, and an open-weight agent fine-tunes a model
> for me. No cloud bill — it all runs on my laptop.

## Scene 2 — Gemma proposes (0:08–0:25)

> Here's the overnight loop, replayed from the real logs. Gemma — running locally — reads the history
> and proposes one training config, with a one-line reason. Run one is a safe head-only baseline.

## Scene 3 — keep / revert (0:25–0:50)

> Now watch run two. Gemma decides to unfreeze the backbone with LoRA, and the validation loss drops
> from one-oh-four to zero-eight-four — the biggest jump of the night, so we keep it. The next few runs
> chase calibration, don't beat it, and get reverted. That keep-or-revert discipline is the whole trick.

## Scene 4 — morning report (0:50–1:00)

> By morning, the winner is scored on a sealed test set: seventy-seven percent accuracy, and a KL from
> gold of just zero-point-oh-nine-four — from an 0.8B model, on an 18-gig Mac.

## Scene 5 — charts (1:00–1:15)

> Tinker then confirms the recipe on a bigger model in the cloud, under a hard dollar cap — a clear lift
> over the zero-shot baseline. Different model, same ruler, no leakage.

## Scene 6 — close (1:15–1:30)

> Open agent, open model, sealed test, auditable guardrails. Slow and steady — while you sleep.
> Links are in the description.

---

### Pronunciation notes for TTS
- "LoRA" → say "LOR-ah".
- "KL" → say "K-L" (letters).
- "0.8B" → the script spells it "an 0.8B"; ElevenLabs reads it fine as "an oh-point-eight-B".
- "zero-point-oh-nine-four" is written out so the number lands clearly.
