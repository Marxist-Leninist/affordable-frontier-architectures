# P039-L2: matched GPU rematch of P039 (autoregression vs masked diffusion, attention vs a Mamba-attention hybrid)

22 November 2026 · one NVIDIA GeForce GTX 1660 SUPER (6 GB) · PyTorch 2.9.1+cu128 · site card:
[zqx.lat/agi-architectures/#p039-l2](https://zqx.lat/agi-architectures/#p039-l2)

P039 was a small GTX 1660 pilot comparing autoregressive (AR) and masked-diffusion byte models; its CPU follow-up P039-L1
(476,736 parameters, 3,200 updates) found that diffusion still produced only spaces. P039-L2 repeats the comparison on a GPU
at the scale of our P045 byte-model benchmark: about 3.29M parameters and 20,000 updates of 16 × 256 bytes per run, two
seeds, four matched arms, the same data for every arm.

**Verdict:** both AR models pass the quality gate and both masked-diffusion models fail it. The two AR models are close, and
which one ranks first depends on how cost is counted.

## Result

| # | model | combined index (attention AR = 100) | quality index | held-out bits/byte [95% CI] | non-space acc | exact 16/16 | space rate | degenerate | decode ms, GPU kernel | decode ms, wall-clock | train GPU min | GPU min to common dev target | dev bits gained / train min | train-step peak MiB | params | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | attention AR ★ | **100.0** | 100.0 | 1.823 [1.780, 1.873] | 15.0% | 0.10% | 20.8% | 0.3% | 6.50 | 27.4 | 14.2 | 12.8 | 0.0291 | 393 | 3,290,880 | pass |
| 2 | hybrid AR ★ | **97.5** | 102.2 | 1.784 [1.741, 1.832] | 15.4% | 0.20% | 20.9% | 0.3% | 6.18 | 29.7 | 17.2 | 11.2 | 0.0184 | 639 | 3,291,224 | pass |
| – | hybrid diffusion, 16 passes ★ | not ranked | 68.3 | 2.449 [2.414, 2.485] | 9.4% | 0.00% | 25.4% | 2.4% | 34.67 | 48.7 | 25.2 | not reached | 0.0271 | 918 | 3,291,224 | fail (bits, non-space) |
| – | attention diffusion, 16 passes ★ | not ranked | 64.9 | 2.543 [2.510, 2.577] | 8.8% | 0.00% | 24.7% | 2.7% | 19.16 | 31.4 | 16.3 | not reached | 0.0471 | 393 | 3,290,880 | fail (bits, non-space) |
| ref | byte 9-gram (Kneser-Ney) | – | – | 2.040 [1.996, 2.088] | 14.1% | 0.15% | 21.0% | 0.4% | – | – | – | – | – | – | – | reference |
| ref | attention AR, learned absolute positions (v1) | – | – | 1.745 [1.709, 1.785] | 15.0% | 0.12% | 21.1% | 0.4% | – | – | – | – | – | – | 3,356,416 | reference, not timed |
| ref | always a space | – | – | – | 0.0% | 0.00% | 100.0% | 100.0% | – | – | – | – | – | – | – | reference |

★ = decoder set before any network was scored on the test set (AR greedy; diffusion = 16 passes, each committing the most
confident masked byte). All 14 network × decoder rows (every diffusion sampler), the per-arm detail tables and the paired
differences are in [REPORT.md](REPORT.md). 95% intervals resample test documents, not training seeds.

- **Masked diffusion fails the quality gate.** Scored by its exact left-to-right likelihood (one byte revealed per step),
  it reaches 2.449 bits/byte (hybrid) and 2.543 (attention), worse than the byte 9-gram's
  2.040 (NELBO: 2.511 and
  2.595). Its best non-space accuracy, 9.7%
  (hybrid, decoder l2r16), is below the 9-gram's 14.1%.
  With 16 passes it no longer collapses to spaces as in P039-L1 (2–3% of
  answers degenerate), but it needs more GPU time than cached AR. With 1–4 passes (attention) or
  1–2 (hybrid) it needs less, but 20–94%
  of those answers repeat one byte, mostly spaces. On raw byte accuracy one-pass diffusion scores 21%,
  above AR's 19%, only because spaces are common; that is why the gate uses non-space accuracy.
- **Among the rotary-position models, the hybrid predicts best:** 1.784 against 1.823 bits/byte
  for attention (paired −0.039, 95% CI −0.048 to −0.030;
  per seed the hybrid leads by 0.052 and 0.026). Non-space accuracy is level
  (+0.0034, CI −0.0006 to
  +0.0073). It uses slightly less GPU kernel time per answer
  (6.18 against 6.50 ms) but is slower in eager wall-clock time (29.7
  against 27.4 ms). It reaches the common dev target sooner (11.2 against
  12.8 GPU minutes), but its full 20,000 updates take 21%
  more GPU time (17.2 against 14.2 minutes).
- **Rotary positions cost attention AR quality here** (next section). The hybrid was not trained with absolute positions,
  so these runs cannot tell whether it is the better design or only less hurt by rotary.
- **Ranking:** the preset combined index gives attention AR 100 and the hybrid 97.5. Counting GPU time to the
  common dev target instead of the full 20,000-update run gives the hybrid 105.8; counting wall-clock decode time
  instead of GPU kernel time gives it 94.4. On quality alone the hybrid leads rotary attention (102.2).
- **"Dev bits gained per train minute"** is highest for attention diffusion (0.0471) only
  because its dev curve starts far higher; it is not comparable between AR and diffusion. No diffusion run reached the common
  dev target (1.748 bits/byte) within 20,000 updates.

## Absolute vs rotary positions

`code/positions_check.py` scores the v1 attention AR runs (P045's learned absolute position table, otherwise the same model,
optimizer, schedule, data and seeds) on the same 4,768 cases with the same document bootstrap as
`summarize.py`. It first reproduces summary.json's v2 numbers exactly (`reproduces_summary_json_v2_bits`).

| model (same 4,768 cases) | positions | params | bits/byte [95% CI] | non-space acc | exact 16/16 |
|---|---|---|---|---|---|
| attention AR (v1) | learned absolute table | 3,356,416 | 1.745 [1.709, 1.785] | 15.0% | 0.12% |
| attention AR (v2) | rotary | 3,290,880 | 1.823 [1.780, 1.873] | 15.0% | 0.10% |
| hybrid AR (v2) | rotary | 3,291,224 | 1.784 [1.741, 1.832] | 15.4% | 0.20% |

- Absolute minus rotary, attention: −0.078 bits/byte (95% CI −0.092 to
  −0.064); non-space accuracy −0.0001 (CI
  −0.0043 to +0.0041).
- Absolute attention minus rotary hybrid: −0.038 bits/byte (CI −0.053 to
  −0.025); non-space accuracy −0.0034 (CI
  −0.0077 to +0.0008).
- On P045's own test protocol the ordering is the same: v1 attention 1.749 / 1.756
  bits/byte (seeds 17 / 43), v2 attention 1.825 / 1.829, v2 hybrid
  1.784 / 1.802.
- Why every matched arm uses rotary: with absolute positions, attention diffusion was at 4.52
  and 4.51 dev bits/byte (NELBO) at update 2,000 (a byte unigram scores 4.57 on
  dev), while the absolute-position AR runs were at 2.28 and 2.31.
  We stopped the two v1 diffusion runs at updates 2,625 and 2,604, still on that plateau, so it is
  not shown that they could never have left it. With rotary positions the same diffusion models were at
  3.24 (attention) and 3.07 (hybrid)
  at update 2,000 (seed 17). Only attention diffusion was tried with absolute positions; the hybrid was not.

## Setup

- **Arms** (`ARMS` in `code/p039l.py`): `attn_ar` = 4 attention layers, causal; `hybrid_ar` = Mamba, attention, Mamba,
  attention, causal; `attn_diff` and `hybrid_diff` = the same stacks, bidirectional, trained as masked diffusion (the
  hybrid's Mamba layers average a forward and a backward scan).
- **Model:** width 256, 4 layers, 8 heads, FFN 1024, context 256 bytes, byte vocabulary plus a mask token (input only; the
  output head has 256 byte classes), rotary positions. Mamba mixer: Mamba-1-style selective SSM (input projection, causal
  depthwise conv k=4, selective scan with state 16 and dt rank 16, SiLU gate, output projection), inner width 308 so the
  mixer has 263,340 parameters against attention's 263,168 (+0.07%). The scan is a fused CUDA kernel (`code/fused_scan.py`)
  that matches an fp64 reference to within 5.2e-07 including gradients (`results/v1/smoke.json`).
- **Training:** AdamW, lr 1e-3, 500 warm-up updates, cosine to 1e-4, weight decay 0.01 (none on A_log and D), gradient clip
  1.0, 20,000 updates × batch 16 × 256 bytes (about 6.9 passes over the training bytes), seeds 17 and 43, dev check
  every 2,000 updates, best dev checkpoint kept. Diffusion: absorbing-state masked diffusion with the 1/t-weighted ELBO loss.
  Every diffusion run had its best dev score at the final update, so all of them were still improving.
- **Pilots:** before the v2 runs, two ~1,070-update pilots of rotary attention diffusion (seed 17, scored on dev only)
  compared the 1/t ELBO loss with a per-token mean loss: 3.424 against 3.418 dev bits/byte at
  update 1,000, within 0.006 bits/byte on one seed. The ELBO, the proper likelihood bound, was kept. Note: `p039l.py`'s `--diff-loss` flag defaults to
  `mean`, so pass `--diff-loss elbo` to reproduce these runs.
- **Data:** the P045 byte corpus (11,873,191 training bytes; articles from WikiText-2, public-domain fiction from
  Project Gutenberg, dialogue from OpenAssistant), split by source entity with exact and near-duplicate exclusion between
  training and evaluation (`data/manifest.json`: "heuristic not proof against all paraphrase overlap"). Held out:
  31 article, 32 fiction and 240 dialogue test documents. File hashes are pinned in
  `PINS` in `code/p039l.py`. The corpus text is not redistributed here. Data credits: WikiText-2 (Merity et al., 2016; Wikipedia text, CC BY-SA 3.0), Project Gutenberg books (public domain in the US) and OpenAssistant Conversations (Köpf et al., 2023; Apache-2.0). The models were trained on this text; none of it is included here.
- **Cases:** a 240-byte prompt and the next 16 bytes at every 64th position after each test window's 256-byte context
  (4,768 cases: 496 article, 512 fiction,
  3,760 dialogue). Scores are averaged per document, then per domain with equal weight, so
  dialogue's share of the cases does not dominate (but article and fiction rest on only 31 and
  32 documents). 95% CIs come from 2,000 bootstrap replicates that resample documents within each domain;
  paired differences use the same replicates. Each case's score is the mean of the two seeds' scores (each model is scored
  on its own predictions), so the intervals do not include seed-to-seed variation.
- **Likelihood:** AR = exact next-byte NLL of the 16 answer bytes; diffusion = exact left-to-right NLL of the same network
  (one masked byte revealed at a time, left to right), with the NELBO also reported.
- **Decoders:** AR greedy with a KV/state cache. Diffusion `confN` = N passes that commit the most confident masked bytes
  first (N = 1, 2, 4, 8, 16); `l2r16` = 16 passes left to right.
- **Metrics:** non-space accuracy (correct non-space answer bytes over non-space answer bytes), exact 16/16, space rate,
  degenerate (one byte 12 or more times in 16).
- **Gate and index:** bits/byte below the byte Kneser-Ney n-gram (order 9, chosen on dev from orders 1–12), non-space
  accuracy above the n-gram's greedy answers, and fewer than half of the answers degenerate. The gate and the four 25%
  weights were added at the owner's request at 04:01 UTC on 22 Nov, 38 minutes after the first v2 runs started and before any
  network was scored on the test set; an earlier draft of the index also weighted training memory. The ★ decoders were
  already fixed in that draft. Combined index = weighted geometric mean of four ratios to attention AR (likelihood,
  non-space accuracy, decode time, training time; 25% each; inverted where lower is better), only for rows that pass the
  gate. Quality index = likelihood and non-space accuracy only.
- **Timing:** batch 1, three rounds in rotated arm order with no other GPU process (checked before and after each
  measurement). Decode = one 16-byte answer after a 240-byte prompt. "GPU kernel" = summed CUDA kernel time from
  torch.profiler (the index uses this); "wall-clock" = eager PyTorch time with synchronisation. Train GPU min = 20,000 × the
  median over three rounds of the mean kernel time per training step. All timings are in `results/v2/timing.json`
  (`dev_*` keys = kernel time, the rest wall-clock).
- **Dev scores and training curves:** exact NLL for AR, an 8-sample NELBO (an upper bound) for diffusion, every 2,000
  updates. Common dev target = the worst AR run's best dev score (1.748); GPU min to target is the
  mean over the two seeds. Dev bits gained per train minute = drop from the first dev check to the best one, per GPU minute
  between them. Train-step peak MiB = one training step in the timing pass; the full runs, with dev evaluation, peaked at
  388 (attention AR), 630 (hybrid AR), 784 (attention
  diffusion) and 905 MiB (hybrid diffusion).

## Limits

- One small scale (3.3M parameters, 4 layers, 256-byte context), one corpus, two seeds, one 20,000-update budget (every
  diffusion run was still improving at its last dev check), one GPU type and one eager implementation.
- One diffusion formulation (absorbing-state masked diffusion, 1/t ELBO, confidence-ordered or left-to-right unmasking of a
  16-byte answer). Other objectives, samplers, longer training and larger models are untested here.
- GPU kernel time leaves out Python and kernel-launch overhead. At batch 1 that overhead weighs most on cached AR's
  single-byte steps: in eager wall-clock one answer takes 27.4 ms (attention AR), 29.7 ms
  (hybrid AR), 31.4 ms (attention diffusion, 16 passes) and 48.7 ms (hybrid
  diffusion, 16 passes), against 6.50, 6.18, 19.16 and 34.67 ms
  of kernel time. Kernel time therefore favours AR.
- The hybrid's quality lead is over rotary attention only; attention with learned absolute positions beats both rotary AR
  models on likelihood. The hybrid was not run with absolute positions, and the v1 runs were not part of the timing pass.
- v2 attn_ar seed 17 died in its first launch with a CUDA out-of-memory error while other runs shared the GPU, was
  restarted from scratch, stopped at update 424 and resumed from that checkpoint with `p039l.v2b.py` (PROVENANCE.json).
  No other v2 run was interrupted. The code versions differ only in evaluation batching, out-of-memory retries and timing;
  the model, loss and data are the same.
- The intervals cover test documents only, not training seeds or other data.

## Files

- `REPORT.md`: the full table (all 14 network × decoder rows), detail tables, paired differences.
- `results/v2/`: `summary.json` (everything in REPORT.md), `eval_*.json` (per run), `timing.json` (raw timings, three
  rounds), `p045protocol_*.json`, `baselines.json` (Kneser-Ney orders 1–12 on dev and test), `targets_descriptive.json`,
  `cases.json` (case counts and document ids, which are SHA-256 hashes), `positions_check.json`.
- `results/v1/`: the absolute-position runs, the v1 baselines and the smoke tests.
- `runs/{v1,v2}/<arm>_<seed>/state.json`: training curves (loss every 100 updates, dev every 2,000), training wall
  time, peak memory, code and checkpoint hashes; `done.json`.
- `code/`: `p039l.py` (final; it also trained v2 attn_ar seed 43), `p039l.v2a.py` and `p039l.v2b.py` (trained the other v2
  runs, see PROVENANCE.json), `p039l.v1.py` (v1), `fused_scan.py`, `baselines.py`, `summarize.py`, `index_table_p039.py`, `run_pipeline.sh`, `positions_check.py`,
  `make_card.py` (builds the site card from these files), `make_bundle.py` (builds this folder). Comments that named the
  owner were reworded; the report text written by `summarize.py` and `index_table_p039.py` was corrected after review
  (REPORT.md regenerated with the published files, summary.json byte-identical); one comment in `run_pipeline.sh` no
  longer names the rental instance. No training, evaluation or scoring code changed (PROVENANCE.json `published_edits`).
  `results/v1/smoke_v1_logspace.json` and `smoke_v1b_fused_abs.json` come from intermediate code drafts that are not
  included.
- `data/`: the P045 `manifest.json` and `evaluation_manifest.json` (counts and hashes, no text).
- `PROVENANCE.json`: which code trained and resumed each run, checkpoint SHA-256, full-run peak memory.
- `CHECKPOINTS.md`: the ten best-dev checkpoints on Hugging Face (https://huggingface.co/ScAGI/affordable-frontier-architectures/tree/main/experiments/2026-11-22/p039-l2-gpu-diffusion-rematch).
- Not included: per-case predictions and answer bytes (they contain corpus text), the corpus itself, final-update
  checkpoints, the two dev-only pilots.

## Reproduce

```bash
cd code                                     # the scripts keep data/, runs*/, results*/ and private*/ next to themselves
ln -s /path/to/p045/data data               # the P045 corpus files with the hashes in PINS (p039l.py)
python3 p039l.py smoke                      # data hashes, fused scan vs fp64, parameter matching
for s in 17 43; do for a in attn_ar hybrid_ar attn_diff hybrid_diff; do
  python3 p039l.py train --arm $a --seed $s --out runs_v2 --pos rope --diff-loss elbo   # --diff-loss defaults to mean
done; done
python3 baselines.py results_v2 private_v2  # Kneser-Ney orders 1-12, order chosen on dev
python3 p039l.py evaluate --out runs_v2
python3 p039l.py p045 --out runs_v2 --only attn_ar_17,attn_ar_43,hybrid_ar_17,hybrid_ar_43
python3 p039l.py timing --out runs_v2 --rounds 3
python3 summarize.py _v2                    # writes results_v2/summary.json and REPORT.md
# v1 reference: train attn_ar with --pos abs --out runs, evaluate it with --out runs, then
python3 positions_check.py
```

`code/run_pipeline.sh` is the pipeline that produced these files (it runs the trainers concurrently on one GPU when memory
allows; timing is measured separately on a quiet GPU).
