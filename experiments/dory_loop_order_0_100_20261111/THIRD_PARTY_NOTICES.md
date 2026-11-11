# Sources, rights and license limits

## Independent Dory-inspired NAT motif

Architectural motivation: NVIDIA's official Dory model card, previously inspected November 11, 2026 with displayed revision b366b87:
https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/blob/main/README.md

The tiny split/whole recurrent-order NAT code is an independent implementation. No NVIDIA code or pretrained checkpoint is redistributed; no faithful replication, endorsement or upstream license grant is implied. Repository licensing and permissions are unchanged. This notice assigns no new license to sources, checkpoints or data.

## WikiText

Source: Salesforce/wikitext, configuration wikitext-2-raw-v1, revision b08601e04326c79dfdd32d625aee71d232d685c3.
https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3

Previously inspected metadata for this pinned dataset reports CC-BY-SA-3.0 and GFDL; other licensing prose links CC-BY-SA-4.0. Preserve original-source rights and this unresolved licensing inconsistency. These provenance statements are carried forward from the initial comparison's inspected source records, not a new legal determination. The corpus, local caches, original text, and text excerpts are excluded from this report bundle. Source identification and hashes do not grant redistribution rights.

## FineWeb-Edu

Source: HuggingFaceFW/fineweb-edu, revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9, source rows 1500–1599. The dataset card reports ODC-BY; underlying web text may retain third-party rights. No FineWeb corpus, downloaded source response, local cache, or original-text excerpts are bundled. The 100-document cohort was used in earlier research and is reused, not a newly acquired validation source.
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

## Historical NAT receipts

Five numerical/status artifacts are reused byte-for-byte from the earlier public NAT release for the unchanged control.py readiness check. Their exact hashes and source attribution are in PROVENANCE.json. They include historical Transformer/Mamba measurements and are not Dory results. No prior NAT model checkpoint or implementation is bundled. The prior Mamba architecture reference was state-spaces/mamba at commit e9594ce1c732d97440f0332fdc43170a2294dbfa (Apache-2.0):
https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/LICENSE

## Dependencies

PyTorch is distributed under its BSD-style license, NumPy under BSD-3-Clause, and PyArrow under Apache-2.0. Obtain dependencies from their normal official/reputable distributions; their binaries/source are not included. Training/evaluation uses PyTorch 2.6.0 CPU and NumPy 2.3.5; the WikiText download script uses PyArrow 19.0.1. The two report generators use Python's standard library and emit SVG. Rendering software is optional and separately licensed. No Matplotlib dependency is needed for these Dory reports.
