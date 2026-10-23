# Submission readiness — DEV Hacktoberfest 2026, Week 3 "Slow and Steady"

Single source of truth for what the submission needs, what's done, and what's left.
Last updated: 2026-10-23 ~22:30 IST.

## Deadline

- **Window:** Oct 19 – Oct 25 2026, 11:59 PM PDT. Repo must be created inside the window.
- **Repo created:** Oct 23 2026 (first commit `feat: Day-1 scaffold`) — inside the window.
- **Time left:** ~2 days. Remaining work is writing + publishing, not engineering.
- Any commit after the deadline must be noted in the README.

## The one-line pitch

You sleep with a dataset and one sentence. A local open-weight agent proposes fine-tuning
configs; an ASHA loop trains + prunes ~200 runs for ~$0 on an 18 GB Mac; the best model is
retrained once and scored on a sealed test set. No hand-tuning.

## What the submission IS (judged in this order)

1. **DEV post** — writing quality weighs most, then theme relevance, creativity, technical
   execution, partner use. Tie-break: positive reactions on the post.
2. Tags `#devchallenge` `#hf26challenge`.
3. **Embedded public repo** + **demo link or video**.
4. A "why open innovation matters" section + the prize categories entered.

---

## Readiness checklist

### A. Technical core — DONE
- [x] Autonomous fine-tune lab (`lazylab/`): agent → config → train (MLX/Unsloth) → ASHA prune → keep/revert → log.
- [x] Sealed test set (hashed, scored once). SHA-256 `66342f30…40b8ac`.
- [x] Official scorer (accuracy / KL / Brier / ECE) matching `bench/score_with_jobs.py`.
- [x] Winning config finalized + retrained standalone → `artifacts/best_model/`.
- [x] **Verified sealed-test result:** accuracy **0.770**, KL **0.0938**, Brier **0.0510**, ECE **0.1454** (2,000 decisions).
- [x] Clean git history, conventional commits.

### B. Hugging Face model — DONE
- [x] Repo created: `rahulkumar0612/lazydecider-0.8b`.
- [x] Model card `README.md` (specialist framing, comparison table, recipe, limits).
- [x] `.eval_results/rahulkumar0612__lazydecider-0.8b.yaml` (leaderboard, specialist notes).
- [x] All 20 files uploaded (1.83 GB total; 1.75 GB weights via plain backend — Xet stalled, disabled it).
- [x] Hero image `lazydecider.png` + `thumbnail` added to the model card.
- [x] **PR #1 MERGED** — eval YAML now on `main`: https://huggingface.co/rahulkumar0612/lazydecider-0.8b
- [x] Fixed eval YAML schema: HF's newer `.eval_results` format wants `value`/`date`/`source`/`notes`
      at the **list-item level** (only `id`/`task_id` nested under `dataset:`). The dataset's own
      `submit_results.py` nests everything under `dataset:` and is out of date with HF's validator.
- [x] `task_id`s match the dataset's `eval.yaml` (accuracy / kl_from_gold / brier / ece, config `all`, split `test`).
- [ ] **Leaderboard visibility** — waiting on HF's periodic crawler to index the `dataset.id` back-reference
      (not a PR on the dataset repo). Optional backup: Path A = manual PR editing the dataset card's Table 2.

### C. Public repo — BLOCKER (local only, no remote)
- [ ] **Create a public remote** (GitHub or Cursor-hosted `origin`) and push `lazylab/`.
- [ ] Commit the currently-untracked artifacts (`best_model/`, `experiments_armA.jsonl`, eval YAML).
- [ ] Confirm repo is public and embeddable in the DEV post.

### D. Morning report + visuals — NOT STARTED
- [ ] Results table: baseline (Unsloth 0.73) vs our explore winner (0.77) on the sealed test; + KL/Brier/ECE.
- [ ] Progress chart: validation score / loss over the run (per-run `train_curve` is already logged).
- [ ] Agent-reasoning excerpts: a few one-line "why" entries from `experiments_armA.jsonl`.
- [ ] Final comparison vs leaderboard neighbours (calibration angle — see §18.5 of the design doc).

### E. DEV post — NOT STARTED (the thing judged most)
- [ ] Draft following the outline: backstory → what I built → demo → code → how it works
      (loop + guardrails) → "my agent tried to game the score" → why open matters → lazy payoff → prizes.
- [ ] Embed repo + HF model + demo video.
- [ ] Tags `#devchallenge` `#hf26challenge`.

### F. Demo — NOT STARTED
- [ ] Short screen capture: a run proposing a config + the morning report/table.

---

## Prize categories targeted

| Category | Prize | Requirement | Status / gap |
|---|---|---|---|
| **Best Use of Gemma** | $200 | Gemma as the local agent proposing experiments | ✅ **DONE (log filling)** — Gemma 4 E4B (MLX, Bionic, local port 1235) is autonomously proposing fine-tune configs with one-line reasons, logged `source=gemma` to `artifacts/experiments_armB.jsonl`. Gemma reads the history and adapts (e.g. "transitioning to full tuning to counteract overfitting seen in previous runs"). Fully local/free; Tinker-spend guardrail fires each run. |
| **Best Use of Tinker** | $200 | Fine-tune for a task, show clear improvement over a baseline; confirm on Tinker | ✅ **DONE** — `Qwen/Qwen3.5-4B` LoRA fine-tuned via Tinker (60 steps, cross-entropy, dollar-capped). Sealed-test **accuracy 0.601** (150 cases, 750 decisions, all real) vs **0.483 zero-shot** = clear improvement. Result in `artifacts/tinker_confirm_result.json`; script `src/tinker_confirm.py`. ~$2 of $10 budget used. |
| Sentry Agent Tracing | $100 | Trace agent decisions (latency/tokens/cost) | Optional, not started. |
| Temporal | $100 | Durable loop that survives crashes/retries | Optional, not started. |

### Extra work to claim Gemma + Tinker
- **Gemma:** DECIDED — frame Gemma as the autonomous researcher that finds the configs, backed by a
  **real experiment log** (`source: gemma` + one-line reasons). Approach = Option 1 (reproduce/validate):
  keep the finalized 0.77 model; disclose that the submitted checkpoint's config was first found via a
  budget `gemini-3.8-flash` dev run and reproduced by Gemma. Running **Gemma 4 E4B locally in LM Studio**
  (the `lms get` CLI can't resolve downloads on this machine → downloading the GGUF from HF:
  `unsloth/gemma-4-E4B-it-GGUF` Q4_K_M, ~5 GB, into `~/.lmstudio/models/`). Launcher:
  `src/run_gemma_search.sh` (points the agent at LM Studio :1234, writes to arm B = Gemma-provenance log).
- **Tinker:** `TINKER_API_KEY` is in `.env`; SDK not installed yet. Write the confirm loop
  (`forward_backward` / `optim_step` / `sample` / `save_state`) on a 4B model, hard dollar cap in code,
  score the sealed test once, add a row to the results table. Verify promo credit first.

---

## Why open matters (post section — drafted talking points)

- Agent and tuned model are both open weights; whole lab runs on a laptop.
- Hundreds of agent calls/night are free locally; a cloud API makes the nightly habit cost money.
- The resulting small model is yours: run offline, push the adapter to Hugging Face.

## Honest risks / notes to disclose in the post

- Agent for the dev search was `gemini-3.8-flash` (budget); official Gemma run is a separate, declared step.
- Accuracy is mid-table by design; our axis is **calibration-per-dollar** (KL/ECE from an 18 GB laptop
  vs 27B–80B models). The ASHA accept metric (`select` loss) ≈ `kl_from_gold`, the metric the dataset
  author calls "the point."
- Gold = mean of 3 teacher samples, so scores measure agreement with the teacher, not correctness.

## Ordered next steps

1. Finish the HF weights upload (running) → open + merge the `.eval_results` PR.
2. Create public remote, commit artifacts, push.
3. Generate the morning report (table + chart + reasoning excerpts).
4. (Prizes) Short Gemma 4 E4B agent run; Tinker confirm run on 4B.
5. Record the demo.
6. Write + publish the DEV post, embed everything, add tags.
