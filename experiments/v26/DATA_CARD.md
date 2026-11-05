# Data provenance and bounded use

Evaluation: public ungated HuggingFaceFW/fineweb-edu, sample-10BT, train split held out here; revision87f09149ef4734204d70ed1d046ddc9ca3f2b8f9. Educationally filtered Common Crawl. Dataset ODC-BY1.0, underlying text rights apply. Pinned card freshly retrieved and hashed in data_audit.json. No gated agreement or paid access.

https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md

Fresh viewer rows1000–1099, first4097 UTF8 bytes/document, no normalization/joins/content selection.100 documents,276787 capped bytes;68 eligible documents,97 nonoverlapping2048-input windows,12416 scored final-byte targets. Byte vocabulary0–255. Revision header checked fail-closed. Response,full/capped document,original/donor-window hashes retained. Viewer endpoint itself is not immutable. Corpus excluded from archive/repository.

Inherited training: Salesforce/wikitext,wikitext-2-raw-v1,train revision b08601e04326c79dfdd32d625aee71d232d685c3.36,718 LF-joined rows encoded UTF8,10,951,562 bytes as pinned v14 preparation. Source Wikipedia Good/Featured articles. Upstream metadata lists CC BY-SA3.0 and GFDL, while prose says CC BY-SA4.0: unresolved upstream inconsistency, preserve attribution/share-alike obligations. Fresh card retrieval timed out, recorded verbatim in data_audit.json; v25 retained pinned card was inspected and inherited verification is explicit. No license interpretation changed.

https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md

No new training. v23 context2048 Transformer and v19 local/GRU frozen three-seed checkpoints. Histories and compute differ across architectures; no matched-budget architecture winner claimed. Reproduction requires lawful inherited data/checkpoint preparation, not included weights.

No within-cohort exact full-document duplicates, no matches to previous v15–v25 full-document hashes, no complete evaluation document nor complete scored128-byte target span in inherited WikiText training. Shorter/near-duplicate overlaps not excluded. This is a minimum-length deterministic convenience cohort, not random representative sampling.

Donor is the next eligible distinct-full-document-hash row cyclically; each original window first1408 bytes replaced, final641 bytes retained. Donors can be recipients elsewhere; bootstrap recipient clusters do not fully capture donor dependence. Subgroup uses only observed histories, not current target identity, and requires a historical follower inside replaced prefix. Artificial boundary can disrupt predictions. No corpus, checkpoints or raw cards published.
