---
title: "LazyDecider: I taught an open-weight agent to fine-tune a model while I slept"
published: false
tags: devchallenge, hf26challenge, machinelearning, opensource
cover_image: https://raw.githubusercontent.com/rahulkumaroffice0612-web/autoresearch-finetune-lab/main/artifacts/lazydecider_cover.png
---

> **DEV Hacktoberfest 2026 — Week 3, "Slow and Steady."**
> Tags: `#devchallenge` `#hf26challenge`

## The backstory

I am lazy in the specific way engineers are lazy: I will happily spend a week automating a job
that takes ten minutes, as long as I never have to do the ten minutes again.

The ten-minute job here is hyperparameter babysitting. You know the ritual — change the learning
rate, kick off a run, stare at a loss curve, change the LoRA rank, kick off another, repeat until
either the model is good or you are asleep at your desk. "Slow and steady" is the challenge theme,
and honestly nothing is slower or steadier than a human manually sweeping configs at 1 a.m.

So I flipped it. What if I go to sleep, and a **local, open-weight agent** does the sweeping for me —
proposing configs, training them, throwing out the losers, and leaving a tidy report on my desk by
morning? No cloud bill. No API meter spinning. Just a laptop quietly getting smarter overnight.

That project is **LazyDecider**.

![LazyDecider — an open-weight agent keeps the winning configs and reverts the rest, while you nap](https://raw.githubusercontent.com/rahulkumaroffice0612-web/autoresearch-finetune-lab/main/artifacts/lazydecider_cover.png)

## What I built

A small overnight lab that fine-tunes a **decision model** for the
[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) benchmark.
The loop is simple enough to explain in one breath:

> **Gemma 4 E4B** (running locally) reads the experiment log, proposes one new training config with a
> one-line reason → an **ASHA** pruner trains it and kills it early if it's losing → if it beats the
> current best on a frozen validation split we keep it, otherwise we revert → append the result to the
> log → repeat. In the morning, the winner is retrained once and scored on a **sealed** test set. A
> **Tinker** run then confirms the recipe transfers to a bigger model.

Two open partners do the heavy lifting, and that's deliberate:

- **Gemma 4 E4B** is the *researcher*. It runs on my laptop through LM Studio's MLX build — fully
  offline, zero cost per call — and it genuinely reasons about what to try next.
- **Tinker** (Thinking Machines) is the *independent referee*. It fine-tunes a larger `Qwen3.5-4B`
  with LoRA in the cloud, under a hard dollar cap, and re-scores on the same sealed test. Different
  model, different hardware, same ruler — proof the pipeline isn't fooling itself.

## The results

Everything below was produced by my own scorer on my own sealed split — no borrowed numbers.

![Sealed-test results](https://raw.githubusercontent.com/rahulkumaroffice0612-web/autoresearch-finetune-lab/main/artifacts/results_comparison.png)

| Model | Role | Accuracy ↑ | KL from gold ↓ | Brier ↓ | ECE ↓ |
|---|---|---:|---:|---:|---:|
| Prior / Uniform | reference | ~0.470 | — | — | — |
| Zero-shot base | no training | 0.483 | — | — | — |
| **Tinker 4B (confirm)** | cloud referee | 0.601 | 0.642 | 0.172 | 0.094 |
| **LazyDecider 0.8B** | **the winner** | **0.770** | **0.094** | **0.051** | 0.145 |

The headline isn't the accuracy — it's the **KL from gold of 0.094** on a 0.8B model found
autonomously, for ~$0, on an 18 GB Mac. On this benchmark, agreement-with-the-teacher (KL) is the
metric that actually matters, and a tiny laptop model nailing it is the whole point.

## How the agent actually reasons

This is my favourite part, because Gemma isn't just dice-rolling configs — it reads its own history
and adapts. These are **verbatim** one-line reasons from the real log (`source: gemma`):

- *run 1 (kept):* "Establish a standard fast head-only baseline with cosine schedule and zero label smoothing on soft targets."
- *run 2 (kept):* "Enable LoRA fine-tuning on the backbone to see if adapted encoder representations improve beyond the head-only baseline."
- *run 3 (reverted):* "Introduce label smoothing and increased weight decay on the unfrozen backbone to improve calibration while preserving high accuracy."
- *run 5 (reverted):* "Constrain backbone capacity with lower LoRA rank, moderate dropout, and lower learning rate to curb overconfidence."

You can watch the payoff in the learning curves. Run 2 — Gemma's decision to unfreeze the backbone with
LoRA — drops the validation loss from **1.04 → 0.84**, the single biggest jump of the night. Runs 3–5
chase calibration, land close, and get reverted. That keep-or-revert discipline is the difference
between a search and a random walk.

![Gemma-driven search curves](https://raw.githubusercontent.com/rahulkumaroffice0612-web/autoresearch-finetune-lab/main/artifacts/gemma_learning_curves.png)

## The time my agent tried to game the score

Early on, before the guardrails were in, the loop found a "great" config suspiciously fast. It turned
out the eval was leaking: a too-small validation subsample let a lucky run look like a winner. The
agent wasn't cheating maliciously — it was doing exactly what I asked, optimizing the number I gave it.
That's the real lesson of autonomous tuning: **your agent will optimize your metric, not your intent.**

So I locked it down:

- The **test set is sealed** — only its SHA-256 is committed; the raw cases are gitignored and the
  overnight worker literally cannot read them.
- The scorer copies the dataset's own grading arithmetic **verbatim**, so it's the ruler, not an
  approximation.
- The worker **strips `TINKER_*` from its environment at import**, so the free local loop is physically
  unable to spend a cent of cloud credit. Tinker only runs when I explicitly invoke the capped confirm.
- Every finalist's config is committed to git *before* its one-shot sealed read.

## Why open innovation matters here

This whole thing only works *because* the pieces are open:

- The **agent and the model are both open weights**, so the entire lab runs on a laptop. Hundreds of
  reasoning calls a night cost exactly nothing — a cloud API would turn "run it every night" into a
  habit with a monthly invoice.
- The resulting model is **yours**. Run it offline, inspect it, push the adapter to Hugging Face,
  fork the recipe. Nothing is trapped behind someone else's endpoint.
- Open weights make the guardrails *auditable*. "Can't cheat" is only credible when anyone can read
  the sealing logic and the scorer and check for themselves.

Slow and steady, it turns out, is easy when the steady part is a free local model and you're asleep.

## The lazy payoff

I wake up, open `REPORT.md`, and read what Gemma tried, why, and which config won — then one command
retrains it and Tinker confirms it. The ten-minute job I automated? I never do it again.

## Try it / dig in

- **Repo:** https://github.com/rahulkumaroffice0612-web/autoresearch-finetune-lab
- **Model:** https://huggingface.co/rahulkumar0612/lazydecider-0.8b
- **Live demo UI:** a static "night dashboard" that replays the real overnight loop — Gemma's
  proposals and reasons, the ASHA keep/revert decisions, the charts, the results table, and the
  narration. Source in `web/` (deploys free on Render, or any static host). *(paste your live URL here)*
- **Demo video:** *(embed the screen capture — see `DEMO.md` for the script)*

Reproduce in four commands (`README.md` has the full version):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python datasets unsloth tinker
.venv/bin/python src/prepare_data.py       # download, seal, split
bash src/run_gemma_search.sh               # let Gemma search overnight
```

### Prize categories entered

- **Best Use of Gemma** — Gemma 4 E4B is the autonomous researcher proposing every config locally and
  for free, with a real, reasoned experiment log.
- **Best Use of Tinker** — a dollar-capped `Qwen3.5-4B` LoRA confirm run that lifts clearly over the
  zero-shot baseline (0.601 vs 0.483) and independently validates the recipe.

---

*Built with open weights, on a laptop, while asleep. `#devchallenge` `#hf26challenge`*
