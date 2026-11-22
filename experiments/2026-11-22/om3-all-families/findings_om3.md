- **No single best model.** Under the stated weights the Transformer ({comb:tf}; spread {sp:tf} over the seed x timing-round
  pairs) and the Transformer-Mamba hybrid ({comb:hyb}; {sp:hyb}) are level at the top: their spreads overlap and the timing on this
  shared machine cannot separate them. Mamba-2 is third ({comb:m2}; {sp:m2}); our v2 with positive decay ({comb:om2pos}) and RetNet
  ({comb:retnet}) follow ("≈" in the table marks rows whose spread overlaps the row above). Under the other ways of counting ("level"
  = overlapping spreads): equal weights, {leadc:equal}; quality 50%, {leadc:q50}; speed-heavy, {leadc:speed}; without repeat-span
  accuracy, {leadc:norec}; training cost as time to quality, {leadc:ttq}; inference only, {leadc:infonly}; bf16 weights, {leadc:bf16};
  batch 8, {leadc:b8}; state-only memory (OM2's definition), {leadc:stateonly}; whole-process training memory, {leadc:rsswhole};
  whole decode cache, {leadc:statetotal}; best decode and prefill round, {leadc:latbest}; first timing pass only, {leadc:pass1};
  second timing pass only, {leadc:pass2}; interleaved timing, {leadc:latil}. In the top four under all {nways}: {alltop4rr}. The
  hybrid is in the top four under {top4:hyb} ({aro:hyb:stateonly} with state-only memory), the Transformer under {top4:tf}
  ({aro:tf:ttq} with time to quality, {aro:tf:infonly} with inference only, {aro:tf:b8} at batch 8, {aro:tf:stateonly} with
  state-only memory); our v2 with positive decay ranges {rr:om2pos}, RetNet {rr:retnet}.
- **Attention kernel.** The attention layers (Transformer, hybrid, Griffin, Titans-style, the exact store) use PyTorch's fused CPU
  scaled_dot_product_attention; the recurrent mixers run as ordinary PyTorch ops (Mamba-2 on this study's chunked kernel). The
  `tfplain` control, the Transformer with its attention written out (same function to {tfeq}, `results/checks/tfplain_equivalence.json`),
  takes {ctl:train}x as long per training step, {ctl:train_memory}x the training memory and {ctl:prefill}x the prefill time
  (second-pass medians); its decode difference ({ctl:decode}x) is not resolved (its rounds {ctl:dpl} ms against the Transformer's
  {ctl:dtf} ms in the same pass). It would score {ctl:comb}, inside the spreads of the rows ranked {ctl:near}.
- **Quality** (held-out test bits/byte, mean of three domains, lower is better): our Mamba v1 {bpb:om1}; v2 + always-on store
  {bpb:om2a}; v2 + placebo {bpb:om2p}; our v2 {bpb:om2}; v2 positive decay {bpb:om2pos}; xLSTM mLSTM {bpb:mlstm}; Mamba-2
  {bpb:m2}; hybrid {bpb:hyb}; v2 + state-gated store as designed {bpb:om2g} (soft-read {bpbs:om2g}); Gated DeltaNet {bpb:gdn};
  Griffin {bpb:griffin}; RWKV-7-style {bpb:rwkv7}; Titans-style {bpb:titans}; Hyena {bpb:hyena}; Transformer {bpb:tf}; RetNet
  {bpb:retnet}. Paired contrasts (A minus B, mean of both seeds, with descriptive 95% intervals from resampling test documents
  within domains; they leave out seed-to-seed variation, so per-seed values are given where the seeds disagree):
  v1 - v2 + always-on store {ci:om1:om2a}; mLSTM - Mamba-2 {ci:mlstm:m2}; mLSTM - our v2 {ci:mlstm:om2} (seeds {ps:mlstm:om2});
  our v1 - mLSTM {ci:om1:mlstm}; mLSTM - v2 + always-on store {ci:mlstm:om2a}; hybrid - Mamba-2 {ci:hyb:m2}; Gated DeltaNet -
  Mamba-2 {ci:gdn:m2}; Gated DeltaNet - hybrid {ci:gdn:hyb}; Griffin - RWKV-7-style {ci:griffin:rwkv7} (seeds
  {ps:griffin:rwkv7}); Griffin - Mamba-2 {ci:griffin:m2}; RWKV-7-style - Gated DeltaNet {ci:rwkv7:gdn}; Hyena - Transformer
  {ci:hyena:tf}; RetNet - Transformer {ci:retnet:tf}. (Titans-style - Gated DeltaNet is {ci:titans:gdn}, but see below: it says
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
  mLSTM is level with our v2 but behind v2 with the always-on store; it is one of the slowest to train on this CPU ({sps:mlstm}
  s/step), with no chunkwise or fused kernel.
- **Speed and memory.** Per training step (median over rounds): Transformer {sps:tf} s, RetNet {sps:retnet}, Hyena {sps:hyena},
  hybrid {sps:hyb}, Mamba-2 {sps:m2}, our v1 {sps:om1} (slowest). Time to the common dev-set target ({target} bits/byte, the worst
  final dev score of any single run; dev scores run lower than test scores): hybrid {ttt:hyb} min, Mamba-2 {ttt:m2}, Transformer
  {ttt:tf}; to the Transformer's own final dev score ({tffinal}): hybrid {tttf:hyb}, Mamba-2 {tttf:m2}, Transformer {tttf:tf}.
  These are read off the dev curves of one 2,000-step cosine schedule, interpolated between evaluations every 250 steps; separately
  scheduled shorter runs were not trained, and the state-gated store's dev curves are soft-read. Decoding (median ms/token): our v1
  {dec:om1}, Mamba-2 {dec:m2}, Transformer {dec:tf}, hybrid {dec:hyb}, Titans-style {dec:titans} (slowest). Per-sequence decode
  state after 256 tokens: Griffin {state:griffin} KiB (local attention, fixed size), our v1 {state:om1}, RetNet {state:retnet},
  Gated DeltaNet {state:gdn}, RWKV-7-style {state:rwkv7}, Mamba-2 {state:m2}, mLSTM {state:mlstm}, Titans-style {state:titans},
  hybrid {state:hyb}, Transformer {state:tf}, Hyena {state:hyena} (the history of its long-convolution inputs; its decoder also
  caches {tabs:hyena} KiB of input-independent filter taps, left out). Training memory above the ~0.46 GiB process baseline:
  Transformer {rss:tf} MiB, hybrid {rss:hyb}, Mamba-2 {rss:m2}, our v1 {rss:om1}. The shared machine switched between a fast and a
  slow mode within tens of seconds; between rounds a model's decode time varied by up to {maxjit}% and its training step by up to
  {trjit}%, which the spreads include.
- **The exact-token store ("Mamba-4" idea, from OM2).** Always-on store minus placebo {ci:om2a:om2p}. The state-gated version,
  scored as designed (reads only where the gate exceeds 0.5), reads on {read:om2g}% of tokens and scores {bpb:om2g}, worse than plain
  v2 ({bpb:om2}); by the end of training its gate had not become selective (mean gate weight {gate:om2g}). Read softly at every
  token it scores {bpbs:om2g} (soft minus placebo {ci:om2gsoft:om2p}). Its decode is timed with its trained gate. It fails the
  page's test for state-gated retrieval on held-out loss alone; this benchmark has no working recall measure, since no model copies.
- **No model shows in-context copying.** Copy gain (first-occurrence minus repeat bits/byte) is {cg} over every model, seed and
  gap: every model predicts a span repeated after 16-176 bytes of another document worse than its first occurrence, which follows its
  own text, so any copying is smaller than the cost of the change of context. The repeat-span column ({recrange}% next-byte accuracy)
  is ordinary next-byte skill, a second quality score; leaving it out changes no model's rank by more than {maxmove:norec}.
- **Choices made after seeing the results, and their effect.** (1) Memory: OM2's card (8 models) scored it by the decode state
  alone, which put Mamba-2 first and the Transformer 7th of 8; on these 16 models that puts {lead:stateonly} first, RetNet
  {aro:retnet:stateonly}, the hybrid {aro:hyb:stateonly} and the Transformer {aro:tf:stateonly}, because its 10% column
  multiplies Griffin's score by {spct:griffin}x while the 35% quality column moves no model by more than {qpct}% (the training-speed
  column, which multiplies our v1's score by {mult:om1:train_speed}x, was kept as a real difference in training time). Memory now
  counts fp32 weights plus the per-sequence decode state. (2) That leaves out Hyena's {tabs:hyena} KiB of input-independent filter
  taps; counting them, Hyena is {aro:hyena:statetotal} instead of {ro:hyena}. (3) Training memory is measured above the process
  baseline; with the whole process, as in OM2, {leadc:rsswhole}. (4) Timings are medians over two passes, decided after both were
  seen: with the first pass alone {leadc:pass1}; with the second alone {leadc:pass2}; with each model's best decode and prefill round
  (as in the revised draft) {leadc:latbest}; with prefill and decode timed interleaved in one process (`results/latency_interleaved.json`)
  {leadc:latil}. (5) The state-gated store is scored with its gate as designed; with its soft read it would be
  {wi:om2g_soft_rank} ({wi:om2g_soft_comb}) instead of {ro:om2g}. (6) The recall column was renamed. Alternatives (1)-(4) are among
  the {nways} ways of counting. Because these choices came after the results, and the top of the table matches the site's family
  ranking (Transformer, hybrid, Mamba), that agreement is not independent support for that ranking.
- **Checks.** All eight new arms pass forward-vs-step parity and exact causality at initialisation and after a 12-step smoke
  (`results/checks/`), and every run passes parity again after its 2,000 steps (largest logit difference {paritymax}, on two
  96-byte test sequences). Five arms were reviewed independently against their papers: RWKV-7-style's initialisation was fixed
  before training; for Titans-style the major finding (linear instead of deep memory) was not implemented and only the minor fix (a
  bounded inner step size) was applied. The hybrid, Griffin and Hyena were not reviewed independently (`REVIEWS.md`). Training ran
  on two machines (Mamba-2, our v1 and the two exact-store arms on PyTorch 2.10, the other twelve on 2.7.1) with the same data
  stream for every arm at each seed and the same settings. Hyena runs its forward FFT in float64 and the Titans-style arm takes an
  exact inner gradient step per token, both slower than their papers' recipes.
- **Short context only.** Every speed and memory figure is at 256 tokens or fewer, where the Transformer's KV cache is still
  smaller than its weights; long contexts are not tested.
