"""Rebuild the v44 data contract from pinned public sources and check it against the published v44 ledger.

Train text: WikiText-2-raw-v1 (Salesforce/wikitext @ b08601e0...), rows joined with "\\n" (this reproduces the published train hash), line-deduplicated exactly as the published
source_v38.py `sources()` does, first 2,000,000 bytes. Held-out: the 183 WikiText test windows and 99 FineWeb-Edu
windows named in the published v44 ledger.json (513 bytes each; last 128 targets are scored)."""
import sys, json, hashlib, pathlib
import numpy as np, pyarrow.parquet as pq
hf, v44, out = (pathlib.Path(p) for p in sys.argv[1:4])
dg = lambda b: hashlib.sha256(b).hexdigest()
ledger = json.loads((v44 / "ledger.json").read_text())
raw = {s: "\n".join(pq.read_table(hf / f"wt2_{s}.parquet").column("text").to_pylist()).encode("utf-8") for s in ("train", "validation", "test")}
src = {}; train_seen = set()
for split in ("train", "validation", "test"):
    seen = set(); rows = []
    for line in raw[split].splitlines(keepends=True):
        k = line.strip()
        if k and (k in seen or split != "train" and k in train_seen): continue
        if k: seen.add(k)
        rows.append(line)
    if split == "train": train_seen = seen
    src[split] = b"".join(rows)
train = src["train"][:2000000]
rep = {"raw_bytes": {k: len(v) for k, v in raw.items()}, "dedup_bytes": {k: len(v) for k, v in src.items()},
       "train_sha256": dg(train), "train_matches_v44_ledger": dg(train) == ledger["train_sha256"]}
wiki = []; bad = 0
for w in ledger["wiki"]:
    b = src["test"][w["start"]:w["start"] + 513]
    ok = len(b) == 513 and dg(b) == w["window_sha256"] and dg(b[-128:]) == w["target_sha256"]; bad += (not ok); wiki.append(list(b))
rep["wiki_windows"] = len(wiki); rep["wiki_window_hash_mismatches"] = bad
fr = hf / "fineweb_rows.json"; rep["fineweb_response_sha256_ok"] = dg(fr.read_bytes()) == ledger["fineweb_response_sha256"]
rows = {r["row_idx"]: r["row"]["text"] for r in json.loads(fr.read_text())["rows"]}
fine = []; badf = 0
for w in ledger["fineweb"]:
    t = rows[w["doc"]].encode("utf-8"); b = t[w["start"]:w["start"] + 513]
    ok = len(b) == 513 and dg(b) == w["window_sha256"] and dg(t) == w["full_text_sha256"]; badf += (not ok); fine.append(list(b))
rep["fineweb_windows"] = len(fine); rep["fineweb_window_hash_mismatches"] = badf
if bad == 0 and badf == 0:
    arrays = {"wiki": np.array(wiki, dtype=np.uint8), "fineweb": np.array(fine, dtype=np.uint8)}
    rep["array_hashes"] = {k: dg(v.tobytes()) for k, v in arrays.items()}
    rep["array_hashes_match_v44_ledger"] = rep["array_hashes"] == ledger["array_hashes"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "evaluation.npz", **arrays); np.save(out / "train_u8.npy", np.frombuffer(train, dtype=np.uint8))
    rep["wiki_articles"] = len({w["doc"] for w in ledger["wiki"]})
(out / "DATA_CHECK.json").write_text(json.dumps(rep, indent=1) + "\n"); print(json.dumps(rep))
