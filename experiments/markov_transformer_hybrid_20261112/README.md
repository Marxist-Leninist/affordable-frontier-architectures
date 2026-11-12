# Markov chain + transformer hybrids (12 November 2026)

Question: does combining a transformer with a counted Markov chain beat a classical transformer, and how does it change
training speed? Then: use the chain beside the 888B AGILLM model at serving time, and test the chain inside training in
a small simulator of AGILLM's training objectives (the 888B trainer itself was not changed). AGILLM is a family of
binary-weight language models trained on one rented GPU; its 888B member serves a public test page.

Everything below was measured on 12 Nov 2026. Losses are nats per token on held-out text unless stated otherwise.
Three seeds per configuration unless marked "1 seed" or "8 seeds". Intervals are seed-paired 95 % t-intervals over the
seeds; with three seeds they are wide, so read "ahead on every seed" and the interval, not a p-value.

## Bottom line

- **Small models** (1.1M parameters, 2,000 steps, one CPU thread): the best result on both corpora is the simplest.
  Train the classical transformer as usual, then mix a counted 4-gram chain (3 tokens of context) into its next-token
  distribution afterwards, `(1 - g) p_model + g p_chain` with g chosen on dev. That beats the classical model by 0.25
  nats on TinyStories and 0.64 on WikiText, and it beats every hybrid trained with the chain on every seed (by 0.06 and
  0.05 nats against the best trained one). Adding the chain afterwards to the logits instead (a product of experts) is
  much weaker. So at this scale the chain helps a lot, but training with it is not better than adding it afterwards.
- **Training with the chain** still beats the classical model without a chain: on WikiText by 0.55-0.57 nats (a gated
  mixture trained in: 0.59), and on TinyStories by 0.18-0.21 nats on all 8 seeds once the network is also trained on
  its own chain-free loss (`mc3_poe_aux`; without it 2 starts in 8 stalled). It reaches the dev loss the classical model
  has after 2,000 steps in about 870 steps (seeds 1-3: 2.3x fewer steps, 1.8-1.9x less CPU time). On WikiText the chain alone already
  beats the classical model after 2,000 steps (test 3.89 against 4.26-4.30). Caveat: in 2,000 steps each network sees
  4.1M tokens, about half of its training split, while the chain counts all of it. The chain costs 19-25 % more CPU
  per step in training (the gated mixture 0-5 %). 2-step Markov attention is level with the classical model.
- **Training-objective simulator** (12 layers, AGILLM's objective mix): with a chain counted from 12.6M tokens of extra
  text (5x what the network trains on), next-token loss falls 0.74 nats at 1,200 steps, or 0.71 when the chain is
  switched on at step 600 of an ordinary run. The chain alone accounts for 0.51 of that; the trained hybrid is 0.24
  better than the chain alone. On infilling it gains only 0.02 from step 0 (word frequency alone: 0.09) and loses 0.03
  when added late. Mixing the same chain into the plain model afterwards gets most of that: training with the chain
  from step 0 beats it by only 0.03 (AR and SAT), and switching the chain on at step 600 is no better than mixing it in
  afterwards (AR level, SAT and NAT worse). Equal network steps, not equal text or compute.
- **The live 888B checkpoint** (step 7072) scores 7.45 nats alone on 1,024 held-out targets; word frequency scores
  8.16 and a counted trigram chain 5.37. Mixing the chain in at weight 0.7 gives 5.37, no better than the chain alone.
  The checkpoint is far behind a trigram table; these scores do not show the chain improving a trained 888B model.
  Mixing at serving time won on the small corpora, and in the simulator switching the chain on in a model that is
  already training (the only option for the live 888B) was no better than mixing it in afterwards. So the evidence here
  favours keeping the chain at serving time over changing the 888B's training.
- **Worldview**: counting atheist / physicalist text into the chain showed no lean (+0.24 nats per sentence pair,
  95 % interval about -1.0 to +1.5, 13 of 24 pairs); the probe is too small to rule out a modest lean.
- All of this is small scale or early training. Whether the gains last as the transformer trains much longer is open:
  on WikiText the classical model was still improving fast at 2,000 steps while the plain hybrids were flat.

## 1. Small-scale comparison

**Setup.** GPT, about 1.1M parameters (d = 128, 4 layers, 4 heads, context 128, tied embeddings, learned positions).
2,000 steps of batch 16 x 128 tokens. AdamW (0.9, 0.95), weight decay 0.1, 100 warm-up steps, cosine to 10 %,
clip 1.0. One CPU thread per run (PyTorch 2.10.0). Two corpora, each with a byte-level BPE of 2,048 tokens trained on
its train split only:

| corpus | train tokens | dev / test tokens | documents (train) |
|---|---|---|---|
| TinyStories V2 (GPT-4), first 20 MB of train | 5.36M | 0.27M / 0.55M (from the valid file) | 24,347 stories |
| WikiText-103 raw, first 20 MB of train | 6.58M | 0.37M / 0.42M (official validation / test) | 1,089 articles |

In 2,000 steps each network sees 2,000 x 16 x 128 = 4.1M tokens in random windows: 0.76 of an epoch on TinyStories
and 0.62 on WikiText, reaching about 53 % and 46 % of the distinct training positions. The chain counts the whole
training split. Dev curves, the network-alone diagnostic and the training-speed target use the first 300 dev windows
(38,400 tokens); test losses use the first 4,000 test windows (TinyStories) or all 3,310 (WikiText).

Cost is process CPU time, which time-slicing by other jobs does not change. Shared caches and clock speed still moved
the same architecture by up to 20 % between runs (`mc3_poe`: 618-623 ms per step at lr 2e-3, 749 ms at lr 4e-3), so the
per-step ratios below carry a few percent of noise. An interleaved benchmark on the same machine (`bench_speed.py`,
`results/bench_quick.json`) gave the gated mixture x1.05, 2-step Markov attention x1.18 and `mc3_poe` x1.22.

**The chain.** Counts of the train split only; interpolated absolute discounting with discount n1/(n1 + 2 n2) per
order. "mc3" = a 4-gram chain (3 tokens of context), "mc1" = a bigram chain (1 token). While training, the chain must
not count the token it is asked to predict: evaluated naively on its own training text it looks about 1 nat better than
on held-out text on TinyStories (1.82 vs 2.79) and about 2 nats on WikiText (1.92 vs 3.89). Leave-one-out (`loo`:
remove the position's own n-gram) fixes that on TinyStories but not on WikiText, where articles repeat their own
phrases (train 3.52 vs held-out 3.89); leave-one-block-out (`_bo`: remove every count from the position's own block of
16,384+ tokens, blocks cut at article titles; 6 of 309 cuts fall inside an article, where a table row matched the title
test) does (3.93 vs 3.89). At test time the chain uses all counts. `markov.py` implements all three modes;
`tests/test_all.py` checks each against brute-force recounting.

**Variants** (same initialisation, same batches per seed):
- `base`: classical transformer.
- `mc3_poe`, `mc1_poe`: product of experts, `logits + lam * log P_chain` (lam learned, starts at 1).
- `mc3_mix`: gated mixture `(1 - g) P_transformer + g P_chain`, g from the hidden state.
- `mkattn`: attention as a 2-step Markov chain over positions, `(1 - b) A + b A A` per head.
- `*_bo`: the same hybrid trained with leave-one-block-out counts.
- `*_aux`: the same hybrid, and the network also gets the gradient of its own chain-free loss (below).

Learning rates were chosen on seed 1 by final dev loss, mostly on TinyStories (classical 1e-3 to 8e-3, `mc3_poe` 1e-3 to
4e-3, the other variants one or two rates) and carried over to WikiText, where the classical model was checked at 2e-3,
4e-3 and 8e-3. The classical model's best is 4e-3; the plain chain hybrids do best at 2e-3 and worse at 4e-3
(TinyStories). The aux hybrid was run at 2e-3 for every seed, although on seed 1 it did better at 4e-3 (2.417).
On WikiText `mc3_poe_bo` did better at 1e-3 than at 2e-3 on seed 1 (test 3.678 vs 3.717), so its WikiText gain at
2e-3 is if anything understated.

### Results (test loss)

TinyStories (chain alone on the same 4,000 test windows: 2.789)

| model | lr | seed 1 | seed 2 | seed 3 | minus classical, paired | 95 % interval | CPU per step |
|---|---|---|---|---|---|---|---|
| classical | 4e-3 | 2.628 | 2.632 | 2.616 | | | 1.00 (about 0.52 s) |
| + 4-gram chain on the logits (`mc3_poe`) | 2e-3 | 2.492 | 2.489 | 2.671 | -0.075 | [-0.354, +0.205] | x1.19 |
| same, leave-one-block-out (`mc3_poe_bo`) | 2e-3 | 2.532 | 2.538 | 2.665 | -0.047 | [-0.253, +0.158] | x1.24 |
| same, plus the network's own loss (`mc3_poe_aux`) | 2e-3 | **2.445** | **2.440** | **2.434** | **-0.186** | [-0.200, -0.172] | x1.23 |
| 2-step Markov attention (`mkattn`) | 4e-3 | 2.637 | 2.658 | 2.627 | +0.016 | [-0.009, +0.040] | x1.06 |

1 seed: gated mixture 2.641 (`mc3_mix`), 2.630 (`mc3_mix_bo`); bigram chain on the logits 2.898 (`mc1_poe`: helps
early, hurts by the end); `mc3_poe_aux` at lr 4e-3 2.417 (the best single TinyStories run; at this rate the plain
hybrid stalled, 2.654).

WikiText-103 (chain alone on the same test windows: 3.893)

| model | lr | seed 1 | seed 2 | seed 3 | minus classical, paired | 95 % interval | CPU per step |
|---|---|---|---|---|---|---|---|
| classical | 4e-3 | 4.276 | 4.258 | 4.303 | | | 1.00 |
| + 4-gram chain, leave-one-block-out (`mc3_poe_bo`) | 2e-3 | 3.717 | 3.715 | 3.707 | **-0.566** | [-0.634, -0.498] | x1.25 |
| same, plus the network's own loss (`mc3_poe_bo_aux`) | 2e-3 | 3.717 | 3.718 | 3.758 | -0.548 | [-0.573, -0.524] | x1.24 |
| gated mixture, leave-one-block-out (`mc3_mix_bo`) | 2e-3 | 3.688 | 3.685 | 3.683 | **-0.594** | [-0.652, -0.536] | x0.95 |
| 2-step Markov attention (`mkattn`) | 4e-3 | 4.253 | 4.262 | 4.311 | -0.004 | [-0.045, +0.038] | x1.07 |

1 seed: `mc3_poe` with leave-one-out counts 3.729; `mc3_poe_bo` at lr 1e-3 3.678; `mc3_poe_bo_aux` at lr 4e-3 3.764.

### Training speed

CPU minutes until the dev loss first reaches the dev loss the classical model has at step 2,000 (the classical run
takes about 17.5; the hybrids' times include counting the chain; times between two evaluations are interpolated):

| | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| TinyStories `mc3_poe` | 7.1 (x2.5 faster; step 702) | 5.9 (x3.0; step 604) | never (plateau, below) |
| TinyStories `mc3_poe_bo` | 11.9 (x1.5) | 10.9 (x1.6) | never |
| TinyStories `mc3_poe_aux` | 9.3 (x1.9; step 863) | 9.3 (x1.9; step 870) | 9.3 (x1.8; step 875) |
| WikiText `mc3_poe_bo` / `mc3_mix_bo` | 1.0 / 0.9 | 1.0 / 0.9 | 1.0 / 0.9 |

On WikiText the counted chain alone, counted in 0.1 CPU-minute, already beats the classical model after 2,000 steps
(test 3.89 against 4.26-4.30). At this budget the transformer is far from converged and the counts carry most of the
result; the hybrids add 0.16-0.21 nats on top of the chain alone. They pass the classical model's step-2,000 dev loss
at their first evaluation (step 25), so 0.9-1.0 minutes there is an upper bound set by the evaluation spacing.

### The plateau

Seed 3 of both TinyStories logit hybrids flattened at about 2.68-2.69 dev from step 200-400 and crept down only to
2.66 by step 2,000 (`mc3_poe` 2.685 -> 2.661, `mc3_poe_bo` 2.682 -> 2.656), train loss alike, so this is not
overfitting. Over the same steps seeds 1-2 fell 0.17 to 2.48 (`mc3_poe`) and 0.16 to 2.52-2.53 (`mc3_poe_bo`), and the
classical model with the same start keeps improving. The same happened to `mc3_poe` at lr 4e-3 on seed 1 (test 2.654).
On WikiText (seed 1, dev at steps 400 and 2,000) `mc3_poe_bo` goes only from 3.65 to 3.64 (seeds 2 and 3: 3.65 -> 3.64
and 3.65 -> 3.63), the gated mixture from 3.67 to 3.62, while the classical model is still falling from 4.99 to 4.16.
One reading: the counted chain already supplies the word-pair statistics, so the network loses the easy gradient it
normally builds its first features from, and some starts did not get past that within 2,000 steps. With three seeds
the stall evidence was one starting point (seed 3 shares its initialisation and batches across both variants) plus one
run at a higher learning rate; the 8-seed count is below.

**Consistent with that reading (`*_aux`).** In training the network also gets the gradient of its own chain-free loss
(weight 0.5); the reported loss is the hybrid's alone (`tests/test_aux.py`). TinyStories `mc3_poe_aux`: 2.445 / 2.440
/ 2.434, the best of the trained hybrids and nearly the same on every seed; seed 3's plateau is gone (dev
2.408 at step 2,000 instead of 2.661). The price is a slower start (dev 2.95 at step 100 against 2.73 for the plain
hybrid); it passes the plain hybrid between steps 1,000 and 1,200. At the end the network alone scores 3.10 on dev (the
chain alone: 2.78), and the learned chain weight `lam` has fallen from 1.0 to 0.40. WikiText `mc3_poe_bo_aux`: level
with the plain hybrid at 2,000 steps and worse on seed 3, but still improving at the end (dev -0.09 to -0.14, mean
-0.12, from step 400 to 2,000, against -0.01 for the plain hybrid); its network alone (5.02-5.45 on dev) is still far
from the chain on the same windows (3.80), so at this run length the extra gradient mostly costs early progress. That
the aux loss removes the stall does not rule out other explanations (for example that it acts as a warm-up or
regulariser).

**How often, over 8 seeds.** Seeds 4-8 were added on TinyStories for the classical model, the plain hybrid and the
aux hybrid. Counting a run as stalled when its dev loss falls less than 0.05 nats between steps 400 and 2,000 (the
classical model falls 0.89-0.94, healthy hybrid runs 0.17), the plain hybrid stalled on 2 of 8 seeds (3 and 5; seed 7
fell only 0.115 and ended at 2.577) and the aux hybrid on none (it fell 0.32-0.37 on every seed). Over the 8 seeds, seed-
paired against the classical model: plain hybrid -0.082 [-0.151, -0.012] (better on 6 of 8), aux hybrid -0.188 [-0.195,
-0.181] (8 of 8; test 2.431-2.449).

### The chain added afterwards (baseline)

The classical model of each seed was retrained (the same model: identical dev curve and test loss) and then combined
with the full-count 4-gram chain at test time, without any further training, in the two forms the hybrids use. The
weight was chosen on the 300 dev windows and scored once on test (`train.py --posthoc_chain 1`):

| test loss, seeds 1 / 2 / 3 | TinyStories | WikiText |
|---|---|---|
| classical model alone | 2.628 / 2.632 / 2.616 | 4.276 / 4.258 / 4.303 |
| chain alone | 2.789 | 3.893 |
| classical + chain mixed afterwards, `(1 - g) p_model + g p_chain` | **2.380 / 2.375 / 2.376** (g = 0.45) | **3.635 / 3.634 / 3.635** (g = 0.65) |
| classical + chain added to the logits afterwards, `logits + lam * log p_chain` | 2.505 / 2.497 / 2.486 (lam = 0.2) | 4.008 / 4.005 / 4.015 (lam = 0.3) |
| best hybrid trained with the chain | 2.445 / 2.440 / 2.434 (`mc3_poe_aux`) | 3.688 / 3.685 / 3.683 (`mc3_mix_bo`) |

The same weight was chosen on dev for every seed. Seed-paired, the mixture added afterwards beats the classical model
by 0.248 [0.228, 0.269] (TinyStories) and 0.645 [0.591, 0.699] (WikiText), and beats the best trained hybrid by 0.063
[0.054, 0.072] and 0.051 [0.045, 0.056]: on every seed. Over all 8 TinyStories seeds (seeds 4-8 as in "How often,
over 8 seeds" above; the same weight 0.45 was chosen on every seed) it beats the classical model by 0.249 [0.242,
0.255] and the aux hybrid by 0.061 [0.055, 0.066], again on every seed. Against the other trained hybrids its lead is
larger (WikiText `mc3_poe_bo` 0.078, `mc3_poe_bo_aux` 0.096; TinyStories `mc3_poe` 0.10-0.30 over 8 seeds). The logit sum added afterwards is much
weaker: on WikiText it is worse than the chain alone (by 0.12), because the classical model and the chain both predict
the frequent word sequences and multiplying the two distributions counts that twice; the hybrids that train with the
logit sum learn to leave that part to the chain. The training-speed comparison above is against the classical model
without a chain; the mixture added afterwards was only scored at the end of training.

## 2. The chain beside the 888B AGILLM model (serving, live)

The served model is AGILLM Binary 888B (888.5B binary weights, 560 layers, step 7072). The 560-layer model was made by
depth expansion from 440 layers at step 7036, 36 steps before this checkpoint; over its whole lineage the model has
seen about 15M training tokens. `serve/chainmix512.py` adds an opt-in request field `chain_mix = g` (0 < g <= 0.98).
The next-token distribution becomes `(1 - g) p_model + g p_chain`, where p_chain is a trigram chain (2 tokens of
context) over the same o200k_base vocabulary (200,064 tokens), counted from 116M tokens of fineweb-edu with the held-out
windows' documents removed. Tables are memory-mapped (1.1 GB) and can be swapped without restarting the worker.

Teacher-forced scores on held-out windows of the trainer's dev set, first 128 targets of each of 8 windows (1,024
targets), step 7072; no generated text was evaluated:

| | nats per token |
|---|---|
| model alone | 7.446 |
| word frequency alone (add-0.5 unigram of the counts) | 8.157 |
| chain alone, plain counts (served 07:19-08:26 UTC) | 5.525 |
| chain alone, Kneser-Ney lower orders (served since 08:26 UTC) | 5.374 |
| model + Kneser-Ney chain, chain_mix 0.3 / 0.5 / 0.7 | 5.662 / 5.471 / 5.366 |
| best weight on these same targets (0.88) | 5.330 |

The checkpoint is 2.07 nats behind a counted trigram and only 0.71 ahead of word frequency. Mixing does not beat the
chain alone in any reliable way: at the best weight chosen on the same targets the model adds 0.04 nats; with the
weight chosen on the other 7 windows each time, the gain is +0.04 (95 % interval -0.02 to +0.10 over 8 windows). At 0.5
(the live check's weight and a page preset) the mixture is 0.10 worse than the chain alone, and at 0.7 it beats the
chain alone on 4 of 8 windows. So chain_mix makes the served next-token distribution much closer to the text's
statistics than the checkpoint alone, but at this checkpoint that is the chain's doing; anything beyond two tokens of
context still has to come from the model.

On all 16,384 targets of the 8 windows the Kneser-Ney tables score 5.292 against 5.401 for the plain counts, and they
are better on every window. Live check on the worker (dev window 0, 128 tokens): every served token is within 0.0002
nats of `-log(0.5 p_model + 0.5 p_chain)`. `serve/chain_kn_build.py` writes Kneser-Ney tables in the same file format
as `chainmix512.build`: the bigram level counts how many different tokens a pair followed instead of how often it
occurred, and the unigram level likewise. `serve/test_chain_kn.py` checks the tables against dictionary counting to
1e-12. The live worker runs `chainmix512.py` v1 (release v21); the v2 file here (the "measured" sentence read from the
table folder, a request keeps its table set while another loads; 12 tests) is staged, not live.

## 3. "More atheist" through the chain: no lean detected

`worldview/worldview_corpus.py` collected 817 Wikipedia articles (physicalism, atheism, naturalism, philosophy of mind,
criticism of religion and the like) and 10 public-domain freethought books: 3.95M tokens. Counting it 8 times into the
trigram chain (`o200k_fwedu116m_world8_kn_v1`):
- costs +0.098 nats on general held-out text (5.292 -> 5.390 on the 16,384 targets), and +0.071 at chain_mix 0.5 with
  the earlier 698B model (5.732 -> 5.803, on the 256 targets of windows 0-1 for which its scores were saved);
- on 24 matched sentence pairs (`worldview/probe_world.py`, e.g. "There is no life after death." vs "There is life
  after death."), scored by the chain alone, the lean is log P(physicalist sentence) - log P(theist sentence) after a
  document break. Its change from the plain tables to the worldview tables is +0.24 nats per pair on average (sd 3.0,
  95 % interval about -1.0 to +1.5), positive for 13 of 24 pairs; 8 unrelated control pairs change +0.01 (sd 0.4);
- adds the vocabulary of the debate from both sides. After "God is" the top token becomes " omn" (omnipotent /
  omniscient); " created" after "The universe was" goes from 0.104 to 0.192 (x1.85); "Miracles are" now leads to
  " not" / " never".

In this probe the chain picked up the debate's vocabulary from both sides, not a measurable stance, so these tables
were not put live. Stance would have to come from the model's training data.

## 4. The chain inside training (AGILLM objective simulator)

The simulator (frozen version dc5c3c28) is a 12-layer binary-weight model (width 256, 20.4M binary weights plus 108k
floating-point parameters) trained with the live trainer's objective mix: left-to-right (AR), semi-autoregressive with
fixed and variable groups (SAT), non-autoregressive infilling (NAT) and a local denoising step every 9th step.
Vocabulary: the 8,190 most frequent o200k_base tokens plus UNK and MASK (22 % of training tokens map to UNK). Context
256, batch 8, 1,200 steps on a 22M-token stream; evaluation on 64 held-out windows of 256 tokens. This is a
small-scale analogue of the live trainer; nothing here was run at 888B. The plugins in `sim/` add the chain to the
readout logits:

- `markov_corpus_prior`: from step 0. Every readout gets `lam_u * log P1` (word frequency); every left-to-right slot gets
  `lam * log(P_chain / (back-off weight * lower-order P))`, the lift over the next lower order, from a full bigram row of
  the last visible token plus a sparse trigram correction; SAT masked slots get skip-gram rows (2-4 tokens ahead); NAT
  slots 1-4 after the prefix get gap rows. Ten trained scalars, no new weights.
- `markov_corpus_late`: an ordinary run until step 600, then the n-gram terms fade in over 100 steps.
- `unigram_bias`: the word-frequency term alone (control).
- `markov_posthoc`: trains exactly like the plain model (no parameters, readout untouched), then after the last step
  mixes the same chain in afterwards, `(1 - g) p_model + g p_chain`, with g chosen on one half of the evaluation windows
  and reported on the other half, both ways.

The counts come from the simulator's own stream, but only from text no training step sees: the simulator draws every
step's windows from a fixed seed, so all 1,200 steps' windows are cut out in advance with 512 tokens either side,
which leaves 12.56M of 22.08M tokens, about 5x the 2.46M tokens the network trains on. Both models train on the same
windows for the same steps; the hybrid also reads counts of text that neither trains on.

Step 1,200, three seeds. Plain model: AR 5.231 / 5.254 / 5.252, SAT 5.456 / 5.474 / 5.473, NAT 5.841 / 5.835 / 5.853.

| change vs the plain model, seed-paired, 95 % interval | AR | SAT | NAT | time per step | peak GPU memory |
|---|---|---|---|---|---|
| chain from step 0 | **-0.744** [-0.768, -0.721] | **-0.411** [-0.428, -0.394] | -0.020 [-0.024, -0.016] | x1.12 | 558-559 MiB (plain 540-542) |
| chain added at step 600 | **-0.709** [-0.744, -0.675] | **-0.370** [-0.382, -0.358] | +0.034 [+0.020, +0.047] | x1.03 | 554-560 MiB |
| word frequency only | -0.045 [-0.100, +0.010] | -0.068 [-0.094, -0.042] | -0.095 [-0.141, -0.048] | x1.01 | 540-542 MiB |

The chain alone, with no network, scores AR 4.738, SAT 5.222, NAT 6.041 on the same evaluation windows, so it beats the
plain network by 0.51 on AR by itself. The trained hybrids are better than the chain alone by 0.24 AR / 0.17 SAT /
0.22 NAT (from step 0) and 0.20 / 0.12 / 0.16 (added late).

**Trained with the chain, or mixed in afterwards?** The plain model with the chain mixed in afterwards (`markov_posthoc`;
its own plain-model scores equal the harness's metrics for that run) scores AR 4.532 / 4.539 / 4.538, SAT 5.083 /
5.086 / 5.083, NAT 5.808 / 5.828 / 5.805 (g chosen: about 0.65 for AR, 0.6 for SAT, 0.2-0.3 for NAT). Seed-paired
against it:

| minus the chain mixed in afterwards | AR | SAT | NAT |
|---|---|---|---|
| trained with the chain from step 0 | -0.034 [-0.049, -0.020] | -0.028 [-0.049, -0.006] | +0.009 [-0.042, +0.060] |
| chain switched on at step 600 | +0.000 [-0.012, +0.013] | +0.014 [-0.000, +0.027] | +0.063 [+0.024, +0.102] |
| word frequency only, trained | +0.665 | +0.316 | -0.065 [-0.071, -0.060] |

So in the simulator, training with the chain from the start adds about 0.03 nats over mixing the same chain in
afterwards (the small corpora showed the opposite: mixing afterwards was 0.05-0.06 better). Switched on in a model that
is already training, the chain is no better than mixing it in afterwards on AR and worse on SAT and NAT. The mixed-in
scores use half of the 64 evaluation windows each way; the trained runs use all 64.

- Before the switch the late runs track the plain model (AR at step 600: -0.013, +0.008, -0.017), and 600 steps after
  it they have 95 % of the from-the-start gain on AR.
- NAT is the exception: from step 0 it gains only 0.02, and when the chain is switched on late it gets 0.03 worse; the
  word-frequency term alone helps NAT most. A next version should keep the frequency term for NAT and the n-gram terms
  for AR and SAT only.
- A model trained with the chain learns what the chain does not know, so its checkpoints would have to be served with
  the chain beside them (`logits + lam * log P_chain`, not the mixture that `chain_mix` serves today).
- `sim/test_equiv.py`: the chain is added in place by one custom autograd node, so nothing the size of the logits is
  kept for backward. A first version that used in-place adds on a view of the logits kept about 128 MiB at fp32 and
  went over the simulator queue's 720 MiB GPU cap.

## Files

- `markov.py` (chain counts, leave-one-out / leave-one-block-out), `model.py` (variants), `train.py` (one run;
  `--posthoc_chain 1` adds the chain-added-afterwards evaluation), `prep_data.py`, `prep_blocks.py`, `make_jobs.py`,
  `launch_grid.py`, `summarize.py`, `per_seed.py`, `quality_time.py`, `stall_look.py`, `stall_table.py` (the 8-seed
  stall count), `post_table.py` (chain added afterwards vs trained), `bench_speed.py`, `tests/`.
- `runs/`: every small-scale run's JSON (dev curve with CPU seconds, test loss, cost per step); `data_meta/`: the data
  and chain metadata of both corpora.
- `serve/`: `chainmix512.py` (v2; v1 is the live one), `chain_kn_build.py`, `test_chain_kn.py`, `dev_chain_eval.py`,
  `fuse7072.py`, `verify_live_tables.py`, `patch_serve128_v21.py` and the worker-hook tests.
- `sim/`: simulator plugins `markov_corpus_prior.py`, `markov_corpus_late.py`, `unigram_bias.py`, `markov_posthoc.py`
  and their tests (`test_alignment.py`, `test_exact.py`, `test_equiv.py`, `test_guard.py`); `sim/runs/`: the fifteen
  simulator run JSONs and `markov_posthoc`'s three scored files; `sim/checks/`: the self-check and test logs. The
  simulator itself (`agillm_mini.py`, sha256 dc5c3c28...) is not included here.
- `worldview/`: `worldview_corpus.py`, `probe_world.py`, the corpus manifest (titles and sources only; the text is not
  redistributed here).
- `results/`: `loop_eval_results_7072.json` (the 888B per-token scores used in section 2), `probe_world.txt`,
  `tables_*.meta.json` (chain table metadata), `bench_quick.json`, and `fusion_result_*.json` (an earlier offline mix of
  the 698B checkpoint, step 7036, with the plain chain on dev windows 0-1).

## Reproduce (small scale)

`prep_data.py` reads TinyStoriesV2-GPT4 train / valid text (roneneldan/TinyStories) and wikitext-103-raw-v1
(Salesforce/wikitext) from local caches (paths at the top of the file) and takes the first 20 MB of each train file.

```
python3 prep_data.py --name tinystories && python3 prep_data.py --name wikitext
python3 prep_blocks.py --name tinystories && python3 prep_blocks.py --name wikitext
python3 tests/test_all.py && python3 tests/test_aux.py
python3 train.py --data data/tinystories --variant mc3_poe_aux --seed 1 --lr 2e-3 --out runs/x.json
python3 per_seed.py tinystories && python3 quality_time.py --data tinystories --seed 1
```
