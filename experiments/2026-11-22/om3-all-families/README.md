# OM3: ten model families on one matched benchmark, and one all-metrics index (22 Nov 2026)

OM3 extends the OM2 benchmark (`../om2-mamba-v2`) to ten of the twelve model families ranked by the AGI Architecture Index (https://zqx.lat/agi-architectures/; all but discrete diffusion and Perceiver AR, for the reasons in the card's closing note). Eight more designs are trained on exactly the same data stream, budget and harness, and all sixteen models are measured on every metric and combined into one index, so the trade-offs at this size can be read at a glance. No single model is best under every way of counting; the findings say which conclusions survive.

## The eight new arms

| arm | design | source | check |
|---|---|---|---|
| `hyb` | Transformer-Mamba hybrid (Jamba-style interleave) | Lieber et al. 2024, arXiv:2403.19887; Mamba-2: Dao & Gu 2024, arXiv:2405.21060 | harness only |
| `griffin` | Griffin: RG-LRU recurrence + local multi-query attention | De et al. 2024, arXiv:2402.19427 | harness only |
| `gdn` | Gated DeltaNet (token mixer only, no SwiGLU MLP) | Yang, Kautz & Hatamizadeh 2025, arXiv:2412.06464 | reviewed: ok |
| `titans` | Titans-style MAG with linear long-term memory | Behrouz, Zhong & Mirrokni 2025, arXiv:2501.00663 | reviewed: "fix" verdict; the major finding (linear instead of deep memory) was not implemented, so the arm is MAG with a linear memory, which the paper does not report; the minor step-size bound was applied |
| `mlstm` | xLSTM, mLSTM blocks (xLSTM[1:0]) | Beck et al. 2024, arXiv:2405.04517 | reviewed: ok |
| `retnet` | RetNet (multi-scale retention) | Sun et al. 2023, arXiv:2307.08621 | reviewed: ok |
| `rwkv7` | RWKV-7-style time and channel mixing | Peng et al. 2025, arXiv:2503.14456 | reviewed: fixed (init) |
| `hyena` | Hyena operator, order 2 | Poli et al. 2023, arXiv:2302.10866 | harness only |

Each arm is written from its paper (reference code consulted only where the paper is silent; each file's docstring lists its sources and every simplification). `REVIEWS.md` records the implementers' summaries, the independent paper-fidelity reviews and what was changed after them.

## Protocol

- Same as OM2: P056 corpus (WikiText-2 articles, public-domain Project Gutenberg fiction, OpenAssistant oasst1 dialogue; hashes in `data/`, text not redistributed); 4 layers, d_model 128, byte vocabulary, tied embeddings, about 471K parameters (13 arms within 1.1% of 471,264; the placebo sidecar and the two exact-store arms 504,289, +7.0%); 2,000 AdamW updates of 8 x 256 bytes (lr 3e-3, warmup 100, cosine to 10%, weight decay 0.01 on matrices, clip 1.0); the same sampled byte stream for every arm at each seed (5617, 5649; the two seeds' streams differ); final checkpoint, no test-set selection.
- Quality: held-out bits per byte over the 303 test windows, the mean of the three domains (31 article, 32 fiction, 240 dialogue windows). Repeat-span accuracy (OM2's "exact recall"; `recall` in the data files): next-byte accuracy on bytes 5-32 of a 32-byte span repeated after 16-176 bytes of another document. No model predicts the repeat better than its first occurrence, so it measures next-byte skill, not recall.
- Checks for every arm (`test_arm.py`, results in `results/checks/`): parameter count, forward vs recurrent-step parity (< 1e-4) and exact causality, at initialisation and again after a 12-step training smoke, plus peak memory. Every training run also records forward-vs-step parity after its 2,000 steps on two 96-byte test sequences (`parity_max_abs` in `results/runs/*.json`).
- Speed and memory: one machine (Intel(R) Xeon(R) CPU E5-2697 v3 @ 2.60GHz), one thread, every model in its own process (`speed_all.py`), two passes of 3 rounds with alternating order. Training s/step (batch 8 x 256), prefill tok/s (1 x 256) and decode ms/token (median per-token time of 192 greedy steps after a 64-byte prompt) were measured in both passes; training memory above the process baseline and the per-sequence decode state after 256 tokens (batch-2 cache minus batch-1 cache, which leaves out input-independent tables such as Hyena's filter taps) in the second (`results/speed_table_per_process.json`; first pass `results/speed_table_first_pass.json`). Every timing is the median over the rounds of both passes (raw values in `results/speed_table.json`, built by `merge_speed.py`). The shared machine switched between a fast and a slow mode within tens of seconds, so single rounds vary (a model's decode time by up to a third); the timings from either pass alone, the best rounds and the interleaved timing are alternative ways of counting, and the spreads include the round-to-round variation. The state-gated store uses only the second pass, whose decode is timed with its trained gate (its first-pass decode ran with an untrained gate). The `tfplain` control (second pass only) is the Transformer with its attention written out in ordinary ops instead of PyTorch's fused CPU kernel (`check_tfplain.py`: the same function, `results/checks/tfplain_equivalence.json`). `latency_interleaved.py` timed all seventeen models interleaved in one process (20 iterations, `results/latency_interleaved.json`); every model alternated between a fast and a slow mode, mostly in the same iterations; it is one of the alternative ways of counting. The training runs themselves ran on two machines (PyTorch 2.10 for m2, om1, om2a, om2g; 2.7.1 for the rest), so their wall times are not compared.
- The state-gated store (om2g) is scored as designed, reading only where its gate exceeds 0.5; its soft-read score is reported alongside (arm `om2gsoft` in the contrasts). Its dev curves, used only for minutes to the dev target, are soft-read.
- Paired contrasts (`results/contrasts.json`): A minus B in test bits/byte, mean of both seeds, with a descriptive 95% interval from a domain-stratified document bootstrap (2,000 resamples; same method as OM2), which leaves out seed-to-seed variation.
- Review: the first draft of the card and findings (first speed pass, median timings) was checked by five independent reviewers, the second draft (both passes, best-round prefill and decode) by three more and the third draft (timings paired with the Transformer's round) by two more; `VERIFICATION.md` lists all their findings and what changed. The final version (plain medians over both passes, with the single-pass, best-round and interleaved timings as alternatives) follows the third round's recommendations and was not reviewed again.

## One index

| rank | model | family | **combined** (spread) | rank under 16 ways of counting | quality (bpb) | repeat-span accuracy | training speed (s/step) | decode (ms/token) | prefill (tok/s) | inference memory: fp32 weights + state at 256 tokens (state) | training memory above baseline | minutes to common dev target |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Transformer | Transformer | **100.0** (96.9-101.2) | 1-11 | 100 (2.379) | 100 (52%) | 100 (0.22) | 100 (2.4) | 100 (24.2k) | 100 (2.82 MiB; 1024 KiB) | 100 (290 MiB) | 5.1 |
| 2≈ | Transformer-Mamba hybrid | Transformer-Mamba hybrid | **99.9** (98.5-101.1) | 1-8 | 114 (2.189) | 106 (55%) | 74 (0.30) | 100 (2.4) | 78 (19.0k) | 119 (2.37 MiB; 584 KiB) | 73 (394 MiB) | 3.1 |
| 3 | Mamba-2 | Mamba | **96.7** (93.1-98.1) | 1-4 | 115 (2.178) | 108 (56%) | 59 (0.37) | 104 (2.3) | 62 (15.1k) | 145 (1.94 MiB; 143 KiB) | 52 (561 MiB) | 3.3 |
| 4≈ | Our Mamba v2, positive decay | Mamba (ours) | **93.8** (90.0-94.6) | 3-7 | 116 (2.168) | 107 (56%) | 56 (0.39) | 90 (2.7) | 59 (14.3k) | 145 (1.94 MiB; 143 KiB) | 50 (574 MiB) | 3.7 |
| 5≈ | RetNet | RetNet | **93.6** (90.5-95.2) | 2-14 | 94 (2.475) | 95 (50%) | 93 (0.23) | 82 (2.9) | 69 (16.8k) | 151 (1.86 MiB; 66 KiB) | 69 (418 MiB) | 7.8 |
| 6≈ | Our Mamba v2 | Mamba (ours) | **90.8** (88.1-92.6) | 4-8 | 116 (2.168) | 108 (56%) | 54 (0.40) | 79 (3.0) | 55 (13.3k) | 145 (1.94 MiB; 143 KiB) | 45 (637 MiB) | 3.6 |
| 7≈ | Hyena | Hyena | **90.4** (87.5-92.0) | 5-15 | 102 (2.355) | 102 (53%) | 79 (0.28) | 82 (2.9) | 60 (14.6k) | 100 (2.82 MiB; 1036 KiB) | 71 (408 MiB) | 5.7 |
| 8≈ | Griffin (RG-LRU + local attention) | Griffin / RecurrentGemma | **90.3** (87.4-91.8) | 1-10 | 109 (2.257) | 103 (54%) | 59 (0.37) | 71 (3.4) | 68 (16.5k) | 155 (1.82 MiB; 24 KiB) | 55 (530 MiB) | 5.9 |
| 9≈ | v2 + placebo sidecar | Mamba (ours) + sidecar | **89.0** (86.2-90.4) | 5-10 | 116 (2.163) | 107 (56%) | 53 (0.41) | 74 (3.2) | 54 (13.0k) | 137 (2.06 MiB; 143 KiB) | 46 (633 MiB) | 3.8 |
| 10≈ | Gated DeltaNet (mixer only) | Gated DeltaNet | **87.7** (85.2-89.0) | 6-11 | 113 (2.209) | 106 (55%) | 42 (0.52) | 84 (2.8) | 46 (11.2k) | 147 (1.91 MiB; 117 KiB) | 60 (484 MiB) | 6.1 |
| 11≈ | RWKV-7-style | RWKV-7 | **85.1** (81.4-86.0) | 9-14 | 109 (2.258) | 104 (54%) | 49 (0.44) | 74 (3.2) | 48 (11.7k) | 146 (1.92 MiB; 132 KiB) | 40 (729 MiB) | 6.9 |
| 12≈ | v2 + always-on exact store | Mamba (ours) + exact store | **84.8** (82.3-88.6) | 7-13 | 117 (2.153) | 108 (56%) | 51 (0.43) | 58 (4.1) | 51 (12.4k) | 129 (2.19 MiB; 271 KiB) | 45 (643 MiB) | 3.8 |
| 13≈ | v2 + state-gated exact store ("Mamba-4" idea, gate as designed) | Mamba (ours) + exact store | **84.7** (84.0-88.8) | 8-13 | 113 (2.198) | 108 (56%) | 53 (0.41) | 60 (4.0) | 50 (12.1k) | 129 (2.19 MiB; 271 KiB) | 45 (640 MiB) | 3.7 |
| 14≈ | xLSTM (mLSTM) | xLSTM | **83.7** (81.5-84.7) | 9-14 | 116 (2.170) | 107 (56%) | 40 (0.55) | 71 (3.4) | 43 (10.3k) | 137 (2.06 MiB; 272 KiB) | 48 (603 MiB) | 5.8 |
| 15 | Titans-style (MAG, linear memory) | Google Titans | **73.2** (70.2-73.7) | 15-16 | 105 (2.310) | 102 (53%) | 36 (0.61) | 46 (5.2) | 29 (6.9k) | 132 (2.14 MiB; 345 KiB) | 60 (485 MiB) | 10.9 |
| 16 | Our Mamba v1 | Mamba (ours) | **61.1** (59.5-63.0) | 2-16 | 117 (2.151) | 108 (56%) | 6 (3.95) | 124 (1.9) | 20 (4.8k) | 152 (1.85 MiB; 44 KiB) | 10 (3040 MiB) | 34.2 |

Each column is an index with the Transformer = 100 (higher is better), raw value in brackets. Combined = weighted geometric mean of the columns: quality 35%, repeat-span accuracy 15%, training speed 15%, decode speed 15%, inference memory 10%, prefill speed 5%, training memory 5%. The spread is the lowest and highest combined score over the seed x timing-round pairs (each round's own timings against the Transformer's medians; each seed's quality against the Transformer's at that seed). "≈" marks a row whose spread overlaps the row above (not separated from it). All models with a quality index of at least 80 are ranked. Speed and memory: one machine (Intel(R) Xeon(R) CPU E5-2697 v3 @ 2.60GHz), one thread, every arm in its own process; every timing is the median over the rounds of two passes. "Rank under 16 ways of counting" is the best and worst rank under the stated weights and 15 alternatives: equal weights (quality 14%, repeat-span accuracy 14%, training speed 14%, decode speed 14%, inference memory 14%, prefill speed 14%, training memory 14%); quality 50% (quality 50%, repeat-span accuracy 10%, training speed 10%, decode speed 10%, inference memory 10%, prefill speed 5%, training memory 5%); speed-heavy (quality 25%, repeat-span accuracy 10%, training speed 25%, decode speed 20%, inference memory 10%, prefill speed 5%, training memory 5%); without repeat-span accuracy (quality 41%, training speed 18%, decode speed 18%, inference memory 12%, prefill speed 6%, training memory 6%); training cost as time to quality (quality 35%, repeat-span accuracy 15%, decode speed 15%, inference memory 10%, prefill speed 5%, training memory 5%, time to quality 15%); inference only (no training cost) (quality 45%, repeat-span accuracy 15%, decode speed 20%, inference memory 15%, prefill speed 5%); bf16 weights in inference memory (quality 35%, repeat-span accuracy 15%, training speed 15%, decode speed 15%, prefill speed 5%, training memory 5%, inference memory with bf16 weights 10%); inference memory at batch 8 (quality 35%, repeat-span accuracy 15%, training speed 15%, decode speed 15%, prefill speed 5%, training memory 5%, inference memory at batch 8 10%); decode state only as memory (OM2 definition) (quality 35%, repeat-span accuracy 15%, training speed 15%, decode speed 15%, prefill speed 5%, training memory 5%, decode state only 10%); whole-process training memory (OM2 definition) (quality 35%, repeat-span accuracy 15%, training speed 15%, decode speed 15%, inference memory 10%, prefill speed 5%, whole-process training memory 5%); whole decode cache as memory (incl. input-independent tables) (quality 35%, repeat-span accuracy 15%, training speed 15%, decode speed 15%, prefill speed 5%, training memory 5%, inference memory with the whole decode cache 10%); best round instead of the median for prefill and decode (quality 35%, repeat-span accuracy 15%, training speed 15%, inference memory 10%, training memory 5%, decode speed (best round) 15%, prefill speed (best round) 5%); timings from the first pass only (quality 35%, repeat-span accuracy 15%, inference memory 10%, training memory 5%, training speed (first pass) 15%, decode speed (first pass) 15%, prefill speed (first pass) 5%); timings from the second pass only (quality 35%, repeat-span accuracy 15%, inference memory 10%, training memory 5%, training speed (second pass) 15%, decode speed (second pass) 15%, prefill speed (second pass) 5%); prefill and decode timed interleaved in one process (quality 35%, repeat-span accuracy 15%, training speed 15%, inference memory 10%, training memory 5%, decode speed (interleaved) 15%, prefill speed (interleaved) 5%).

Control: the Transformer with its attention written out in ordinary PyTorch ops instead of the fused kernel takes 1.16x as long per training step, 1.68x the training memory, 1.07x the decode time (not resolved: its decode rounds overlap the Transformer's) and 1.23x the prefill time (second-pass medians), and would score 93.3 against the same reference.

## Findings

- **No single best model.** Under the stated weights the Transformer (100.0; spread 96.9–101.2 over the seed x timing-round
  pairs) and the Transformer-Mamba hybrid (99.9; 98.5–101.1) are level at the top: their spreads overlap and the timing on this
  shared machine cannot separate them. Mamba-2 is third (96.7; 93.1–98.1); our v2 with positive decay (93.8) and RetNet
  (93.6) follow ("≈" in the table marks rows whose spread overlaps the row above). Under the other ways of counting ("level"
  = overlapping spreads): equal weights, Transformer leads (100.0; next Transformer-Mamba hybrid 93.2); quality 50%, Transformer-Mamba hybrid leads (103.2; next Mamba-2 100.8); speed-heavy, Transformer and Transformer-Mamba hybrid are level at the top (100.0 and 95.4); without repeat-span
  accuracy, Transformer and Transformer-Mamba hybrid are level at the top (100.0 and 98.9); training cost as time to quality, Transformer-Mamba hybrid and Mamba-2 are level at the top (112.6 and 111.8); inference only, Mamba-2 and Our Mamba v1 are level at the top (112.0 and 111.3); bf16 weights, Transformer-Mamba hybrid and Transformer are level at the top (100.8 and 100.0);
  batch 8, Mamba-2 and RetNet are level at the top (105.2 and 103.7); state-only memory (OM2's definition), Griffin (RG-LRU + local attention) leads (125.8; next RetNet 118.1); whole-process training memory, Transformer-Mamba hybrid and Transformer are level at the top (100.8 and 100.0);
  whole decode cache, Transformer and Transformer-Mamba hybrid are level at the top (100.0 and 99.9); best decode and prefill round, Transformer and Transformer-Mamba hybrid are level at the top (100.0 and 99.4); first timing pass only, Transformer and Transformer-Mamba hybrid are level at the top (100.0 and 98.8);
  second timing pass only, Transformer-Mamba hybrid and Transformer are level at the top (103.7 and 100.0); interleaved timing, Transformer-Mamba hybrid leads (102.5; next Mamba-2 100.5). In the top four under all 16: Mamba-2 (ranks 1–4). The
  hybrid is in the top four under 15 (8th with state-only memory), the Transformer under 12
  (6th with time to quality, 11th with inference only, 6th at batch 8, 10th with
  state-only memory); our v2 with positive decay ranges 3–7, RetNet 2–14.
- **Attention kernel.** The attention layers (Transformer, hybrid, Griffin, Titans-style, the exact store) use PyTorch's fused CPU
  scaled_dot_product_attention; the recurrent mixers run as ordinary PyTorch ops (Mamba-2 on this study's chunked kernel). The
  `tfplain` control, the Transformer with its attention written out (same function to 5.7e-7, `results/checks/tfplain_equivalence.json`),
  takes 1.16x as long per training step, 1.68x the training memory and 1.23x the prefill time
  (second-pass medians); its decode difference (1.07x) is not resolved (its rounds 2.47–3.17 ms against the Transformer's
  2.30–2.92 ms in the same pass). It would score 93.3, inside the spreads of the rows ranked 3–5.
- **Quality** (held-out test bits/byte, mean of three domains, lower is better): our Mamba v1 2.151; v2 + always-on store
  2.153; v2 + placebo 2.163; our v2 2.168; v2 positive decay 2.168; xLSTM mLSTM 2.170; Mamba-2
  2.178; hybrid 2.189; v2 + state-gated store as designed 2.198 (soft-read 2.157); Gated DeltaNet 2.209;
  Griffin 2.257; RWKV-7-style 2.258; Titans-style 2.310; Hyena 2.355; Transformer 2.379; RetNet
  2.475. Paired contrasts (A minus B, mean of both seeds, with descriptive 95% intervals from resampling test documents
  within domains; they leave out seed-to-seed variation, so per-seed values are given where the seeds disagree):
  v1 - v2 + always-on store -0.002 [-0.005, +0.002]; mLSTM - Mamba-2 -0.008 [-0.012, -0.003]; mLSTM - our v2 +0.003 [-0.002, +0.007] (seeds +0.009 and -0.004);
  our v1 - mLSTM -0.019 [-0.024, -0.015]; mLSTM - v2 + always-on store +0.018 [+0.014, +0.022]; hybrid - Mamba-2 +0.011 [+0.005, +0.016]; Gated DeltaNet -
  Mamba-2 +0.031 [+0.026, +0.036]; Gated DeltaNet - hybrid +0.020 [+0.015, +0.024]; Griffin - RWKV-7-style -0.002 [-0.006, +0.003] (seeds
  +0.013 and -0.016); Griffin - Mamba-2 +0.078 [+0.071, +0.086]; RWKV-7-style - Gated DeltaNet +0.050 [+0.044, +0.055]; Hyena - Transformer
  -0.024 [-0.028, -0.019]; RetNet - Transformer +0.097 [+0.091, +0.102]. (Titans-style - Gated DeltaNet is +0.101 [+0.096, +0.107], but see below: it says
  nothing about Titans against Gated DeltaNet.)
- **Against the papers.** Gated DeltaNet is reported ahead of Mamba-2 at 1.3B parameters and 100B tokens (Yang et al. 2025, Table 3:
  Wiki perplexity 16.42 against 16.56) but trails it here; we did not test why. Both arms here are mixer-only with 32-wide key heads,
  narrower than the Gated DeltaNet paper uses (its ablation tested 64 to 256 and chose 128), and the Gated DeltaNet arm has half as
  many heads and a 25% smaller recurrent state (6,144 against 8,192 values per layer). The RetNet paper's perplexity scaling
  comparison with Transformers is at 1.3B, 2.7B and 6.7B parameters (its training-cost table adds 13B); it calls the results
  comparable and reports RetNet ahead above about 2B, so it says nothing direct about 0.47M parameters, and this arm's heads, value
  width and FFN are narrower than the paper's. Titans reports gains at ordinary 4K training context (170M to 760M parameters, with
  MAG ahead of Gated DeltaNet and Transformer++ at 340M, 400M and 760M) and at very long contexts, all with a deep MLP memory; its
  linear-memory ablation is of the memory-only model (Wiki perplexity 28.49 against 27.01). This arm is MAG with a small linear
  memory, a combination the paper does not report; with linear memory its update is the same kind of gated delta rule as Gated
  DeltaNet's with a third of the memory state, so it tests neither Titans' deep memory nor Titans against Gated DeltaNet. xLSTM's
  mLSTM is level with our v2 but behind v2 with the always-on store; it is one of the slowest to train on this CPU (0.55
  s/step), with no chunkwise or fused kernel.
- **Speed and memory.** Per training step (median over rounds): Transformer 0.22 s, RetNet 0.23, Hyena 0.28,
  hybrid 0.30, Mamba-2 0.37, our v1 3.95 (slowest). Time to the common dev-set target (2.427 bits/byte, the worst
  final dev score of any single run; dev scores run lower than test scores): hybrid 3.1 min, Mamba-2 3.3, Transformer
  5.1; to the Transformer's own final dev score (2.325): hybrid 4.4, Mamba-2 5.0, Transformer 7.2.
  These are read off the dev curves of one 2,000-step cosine schedule, interpolated between evaluations every 250 steps; separately
  scheduled shorter runs were not trained, and the state-gated store's dev curves are soft-read. Decoding (median ms/token): our v1
  1.9, Mamba-2 2.3, Transformer 2.4, hybrid 2.4, Titans-style 5.2 (slowest). Per-sequence decode
  state after 256 tokens: Griffin 24 KiB (local attention, fixed size), our v1 44, RetNet 66,
  Gated DeltaNet 117, RWKV-7-style 132, Mamba-2 143, mLSTM 272, Titans-style 345,
  hybrid 584, Transformer 1024, Hyena 1036 (the history of its long-convolution inputs; its decoder also
  caches 1024 KiB of input-independent filter taps, left out). Training memory above the ~0.46 GiB process baseline:
  Transformer 290 MiB, hybrid 394, Mamba-2 561, our v1 3040. The shared machine switched between a fast and a
  slow mode within tens of seconds; between rounds a model's decode time varied by up to 35% and its training step by up to
  23%, which the spreads include.
- **The exact-token store ("Mamba-4" idea, from OM2).** Always-on store minus placebo -0.010 [-0.014, -0.007]. The state-gated version,
  scored as designed (reads only where the gate exceeds 0.5), reads on 0.04% of tokens and scores 2.198, worse than plain
  v2 (2.168); by the end of training its gate had not become selective (mean gate weight 0.05 and 0.09). Read softly at every
  token it scores 2.157 (soft minus placebo -0.006 [-0.010, -0.002]). Its decode is timed with its trained gate. It fails the
  page's test for state-gated retrieval on held-out loss alone; this benchmark has no working recall measure, since no model copies.
- **No model shows in-context copying.** Copy gain (first-occurrence minus repeat bits/byte) is -0.14 to -0.01 over every model, seed and
  gap: every model predicts a span repeated after 16-176 bytes of another document worse than its first occurrence, which follows its
  own text, so any copying is smaller than the cost of the change of context. The repeat-span column (50–56% next-byte accuracy)
  is ordinary next-byte skill, a second quality score; leaving it out changes no model's rank by more than 2.
- **Choices made after seeing the results, and their effect.** (1) Memory: OM2's card (8 models) scored it by the decode state
  alone, which put Mamba-2 first and the Transformer 7th of 8; on these 16 models that puts Griffin (RG-LRU + local attention) first, RetNet
  2nd, the hybrid 8th and the Transformer 10th, because its 10% column
  multiplies Griffin's score by 1.45x while the 35% quality column moves no model by more than 6% (the training-speed
  column, which multiplies our v1's score by 0.65x, was kept as a real difference in training time). Memory now
  counts fp32 weights plus the per-sequence decode state. (2) That leaves out Hyena's 1024 KiB of input-independent filter
  taps; counting them, Hyena is 9th instead of 7th. (3) Training memory is measured above the process
  baseline; with the whole process, as in OM2, Transformer-Mamba hybrid and Transformer are level at the top (100.8 and 100.0). (4) Timings are medians over two passes, decided after both were
  seen: with the first pass alone Transformer and Transformer-Mamba hybrid are level at the top (100.0 and 98.8); with the second alone Transformer-Mamba hybrid and Transformer are level at the top (103.7 and 100.0); with each model's best decode and prefill round
  (as in the revised draft) Transformer and Transformer-Mamba hybrid are level at the top (100.0 and 99.4); with prefill and decode timed interleaved in one process (`results/latency_interleaved.json`)
  Transformer-Mamba hybrid leads (102.5; next Mamba-2 100.5). (5) The state-gated store is scored with its gate as designed; with its soft read it would be
  11th (85.5) instead of 13th. (6) The recall column was renamed. Alternatives (1)-(4) are among
  the 16 ways of counting. Because these choices came after the results, and the top of the table matches the site's family
  ranking (Transformer, hybrid, Mamba), that agreement is not independent support for that ranking.
- **Checks.** All eight new arms pass forward-vs-step parity and exact causality at initialisation and after a 12-step smoke
  (`results/checks/`), and every run passes parity again after its 2,000 steps (largest logit difference 2.3e-05, on two
  96-byte test sequences). Five arms were reviewed independently against their papers: RWKV-7-style's initialisation was fixed
  before training; for Titans-style the major finding (linear instead of deep memory) was not implemented and only the minor fix (a
  bounded inner step size) was applied. The hybrid, Griffin and Hyena were not reviewed independently (`REVIEWS.md`). Training ran
  on two machines (Mamba-2, our v1 and the two exact-store arms on PyTorch 2.10, the other twelve on 2.7.1) with the same data
  stream for every arm at each seed and the same settings. Hyena runs its forward FFT in float64 and the Titans-style arm takes an
  exact inner gradient step per token, both slower than their papers' recipes.
- **Short context only.** Every speed and memory figure is at 256 tokens or fewer, where the Transformer's KV cache is still
  smaller than its weights; long contexts are not tested.

The numbers in these findings and on the site card are filled in from `results/` by `render_findings.py` (templates: `findings_om3.md`, `card_findings.html`); facts about the papers and the code are written by hand.

## Reproduce

```
python3 test_arm.py ARM                       # standard checks for arms/ARM.py
python3 om_queue_vast.py 16 jobs_om3.json     # 16 runs: 8 arms x 2 seeds (om3bench.py run --arm ARM --seed S ...)
python3 -c "import torch, safetensors.torch as s; torch.save(s.load_file('../om2-mamba-v2/checkpoints/om2g-5617.safetensors'), 'om2g-5617.pt')"
OM_TRAINED_OM2G=om2g-5617.pt \
  python3 speed_all.py tf,tfplain,m2k,om1,om2,om2pos,om2p,om2a,om2g,hyb,griffin,gdn,titans,mlstm,retnet,rwkv7,hyena speed.json 3
#   run speed_all.py twice (two passes); the second run is speed.json, the first speed_first_pass.json
python3 latency_interleaved.py tf,tfplain,m2k,om1,om2,om2pos,om2p,om2a,om2g,hyb,griffin,gdn,titans,mlstm,retnet,rwkv7,hyena \
  latency.json --trained om2g=om2g-5617.pt     # reference only
python3 merge_speed.py speed.json speed_first_pass.json latency.json results/speed_table.json
python3 check_tfplain.py results/checks/tfplain_equivalence.json
python3 index_all.py ../om2-mamba-v2/results/runs,results/runs results/speed_table.json index/
python3 contrasts_all.py ../om2-mamba-v2/results/runs,results/runs contrasts.json
```

`om2bench.py` is an unchanged copy of the OM2 harness (it defines training, evaluation, probes and the OM2 arms); `om3bench.py` loads the extra arms from `arms/` and adds the `m2k` and `tfplain` speed arms. The data files are not included; their sha256 are in `data/SHA256SUMS`. `index_all.py` and `contrasts_all.py` read either `<arm>-<seed>/result.json` folders or flat run files.

## Files

- `arms/*.py`: the eight new models; `om3bench.py`, `om2bench.py`: harness; `test_arm.py`, `run_capped.sh`, `check_tfplain.py`: checks
- `om_queue_vast.py`, `jobs_om3.json`: job running; `speed_all.py`, `latency_interleaved.py`, `merge_speed.py`, `index_all.py`, `contrasts_all.py`, `render_findings.py`, `make_site_card.py`: analysis and write-up
- `results/`: `INDEX.md` / `index.json` (the one index, all 16 models), `speed_table.json` (and the per-pass and interleaved tables), `contrasts.json`, `runs/*.json` (the 16 OM3 runs, with per-document scores and dev curves; the OM2 runs are in `../om2-mamba-v2/results/runs/`), `checks/*.json`
- `checkpoints/`: 16 safetensors state dicts for the OM3 runs (load into `om3bench.build(arm)`), `CHECKPOINTS.json` with sha256
- `REVIEWS.md` (implementation reviews), `VERIFICATION.md` (reviews of the card and findings), `THIRD_PARTY_NOTICES.md`, `data/`, `SHA256SUMS`

## Limits

- Small models (about 0.47M parameters), one corpus, two seeds, 2,000 updates, CPU: a matched comparison of these designs at this size and budget, not a ranking of the families at scale. Several designs are known to gain most from scale or long context, and every speed and memory figure here is at 256 tokens or fewer.
- Each family is one implementation at one width allocation, scaled down to the shared parameter budget; the simplifications are listed per file. The Titans-style arm is MAG with a linear memory, a combination the Titans paper does not report.
- Index weights are a stated choice; the per-metric columns and the alternative ways of counting are given so other weightings can be judged. Speed is one CPU model and one thread: the attention layers (Transformer, hybrid, Griffin, Titans-style, the exact store) use PyTorch's fused CPU attention kernel and the recurrent mixers ordinary PyTorch ops, so GPU rankings, or rankings with fused recurrent kernels, can differ.

