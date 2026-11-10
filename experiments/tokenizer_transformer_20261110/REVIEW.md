# Independent read-only review
Reviewed prepare.py, train.py and PROTOCOL.md before the first complete paired result.

No must-fix training or scoring bug found:
- Causal attention and shifted targets score every text token. Padding labels are ignored.
- Same raw sample/order at128 updates; summed canonical NLL / identical UTF8 byte counts.
- Same backbone initialization per seed.
- Training CPU counter excludes evaluation/tokenization as declared.

Interpretation cautions:
1. Padded GPT2 has unused output classes receiving probability and gradients. It equalizes shapes, not useful vocabulary or optimization difficulty.
2. GPT2 and padded GPT2 shared-ID embedding initialization differs because embedding construction consumes vocabulary-dependent RNG before explicit initialization. Only trunk initialization is matched.
3. CPU limit can overshoot by one optimizer step; report actual seconds.
4. Require all arms to reach128 updates before claiming equal-data endpoint.
5. <=63-token filtering under both vocabularies selects short mutually tokenizable fragments. First-N sampling and128 test fragments can cluster by article.
6. EOS start context is given and EOS target excluded; metric is canonical-path fragment likelihood, not marginal document probability.
7. Fixed shared hyperparameters need not be equally optimal across vocabularies.
8. Three seeds quantify limited optimizer variability, not corpus/domain uncertainty. No equivalence or frontier-scale claim follows.
