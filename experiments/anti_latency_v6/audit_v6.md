Independent read-only audit passes; no blocking issue.

All24 positive paired retention improvements, three-seed means, timing values/ratios and compute accounting reproduce from raw results. All four reconstructions have maximum weight error0. All eight checkpoint validations pass, retaining Adam steps1000 versus resetting to200; optimized/full-cache parity errors are approximately1e-14.

Decode uses a full n-token prefix then times one next-token call. Each timing array has41 samples plus5 warmups. Disposable training probes present235,520 targets; quality training presents6,553,600. Total6,789,120 excludes inference/evaluation work.

Caveats remain: three seeds and reused small heldouts; correlated FineWeb windows; seed2/fresh-Adam timing, not retained-optimizer timing; no retained128 control or full optimizer/context interaction; no broad speed, frontier-quality or economic pass. Reviewer ran no competing benchmarks and made no code changes.
