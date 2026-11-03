# v8: test the recurrent result on Hugging Face WikiText-2

2026-11-03 follow-up to v7. This is actual isolated CPU neural training on an established real-text dataset, with a stronger Transformer baseline and equal training/tuning budgets. It is not an intelligence benchmark or an economic feasibility result.

## Result

The recurrent baseline again has lower held-out loss in all three paired seeds, but this remains a tiny short-context language-model result:

| Model | Seed1 bpb | Seed2 bpb | Seed3 bpb | Mean bpb | Seed sample SD |
|---|---:|---:|---:|---:|---:|
| Normalized four-head Transformer | 3.517405 | 3.490569 | 3.440045 | 3.482673 | 0.039280 |
| Vanilla recurrent LM | 3.279704 | 3.443202 | 3.282271 | 3.335059 | 0.093664 |

Mean recurrent improvement is 0.147614 bits/byte, **4.24% lower cross-entropy**. This is not an accuracy or intelligence percentage. Recurrent seed variation is materially larger than the Transformer's in this three-seed sample. No confidence claim across models or datasets follows. The result extends the narrow v7 observation to a second real-text corpus, not to general reasoning or long-range retention.

## Dataset and metric

Source: [Salesforce/wikitext on Hugging Face](https://huggingface.co/datasets/Salesforce/wikitext), config `wikitext-2-raw-v1`, pinned revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Official train/validation/test splits, 36,718/3,760/4,358 rows. All original rows, including empty rows, are joined by one LF and encoded as UTF-8 without filtering or normalization. `data_manifest.json` preserves source URLs and exact Parquet/serialized-text SHA256s. Source dataset card lists CC-BY-SA-3.0/GFDL; no source corpus is redistributed in this repository.

Fixed vocabulary256 models every UTF-8 byte without unknown-token substitution. Scores are **bits per byte (bpb)**, lower is better. They are NOT canonical WikiText word perplexity and are not directly comparable with v7's character scores. Byte representation preserves all text, but changes sequence granularity.

Training split: 10,951,562 bytes. Validation: 1,148,007 bytes; test: 1,292,013 bytes. Evaluation covers all complete64-byte segments, not a selected tiny prefix: 1,147,968 validation targets and1,291,968 test targets. The first byte of each split lacks a preceding target context and is not scored; trailing38 validation and44 test target bytes form incomplete final segments and are excluded identically for both models. State resets every64 bytes. Full-split scoring still does not test long-context retention.

## Fixed controlled protocol

- Models unchanged from v7: normalized four-head one-block causal Transformer, width32, affine pre-LayerNorm attention and4x tanh FFN with residuals, final LayerNorm and sinusoidal positions. Vanilla tanh recurrent baseline with learned embedding, bias, recurrence and output projection.
- Transformer29,024 parameters; recurrent width51 has28,764, 0.896% fewer. Width selected solely by nearest parameter count before training.
- Each run:400 Adam updates ×8 sequences ×64 bytes =204,800 training-target presentations. Identical sampled windows in paired runs. Global gradient clipping1, beta1 .9, beta2 .999, epsilon1e-8, float64.
- Three learning rates .001/.003/.01, tuning seed101 for each architecture. Select by full validation bpb, save `selection.json`, then final seeds1/2/3 with test scoring. Test bytes may be loaded before selection, but test scores are not used for selection. No early stopping or best-seed reporting.
- Six runs per architecture including tuning,1,228,800 training-target presentations each. This is not unique data coverage and is deliberately very small relative to the corpus. Equal parameters/tokens/tuning runs do not imply equal FLOPs, training maturity, or globally optimized hyperparameters. Both selected learning rates are .01, the upper grid endpoint, so neither optimum is established.

`protocol.json` was written before training. `results.json` preserves every tuning/final run, training traces, exact data/source/weight hashes and actual CPU timings. Training time is not serving latency or all-in cost.

## Verification and reproduction

Python3.10+, NumPy, PyArrow19.0.1 for dataset extraction. PyArrow was installed from the ordinary package registry into isolated scratch storage; no paid hardware or production environment changed.

    python prepare_data.py
    OPENBLAS_NUM_THREADS=1 python test_models.py
    python run.py

`prepare_data.py` downloads only pinned Hugging Face URLs and verifies explicit expected Parquet SHA256 values. It also computes serialized text hashes. The independent audit reserializes every original row, checks byte round-trip fidelity, model identity with v7, finite-difference gradient checks, causal and batch invariance, equal budgets, selection/test separation, and token-weighted full-heldout evaluation including the smaller last batch.

## Scope

This tests whether the v7 small-model advantage transfers to a different real-text corpus under the same bounded training procedure. It cannot establish general intelligence, superiority to typical large modern Transformers, long-range memory, frontier capability, half-price API economics or a profitable $500 subscription. No hidden reasoning allowance, customer workload or required quality has been reduced to make a cost claim; no such claim is made.
