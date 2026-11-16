# Attribution and changes

AdaptiveGateNet-AFP is extracted from GAIR-NLP/ASI-Arch, data/106.json, record index 712 at commit 3113c51b04a1d4a7520a3b0238c6a08d3194d58f. Its repository license is Apache License 2.0, included unchanged as ASI_ARCH_LICENSE.

upstream712.py is unchanged. reference712_dtype_preserving.py changes only the torch.compile decorator and inverse dtype cast, as asserted by the tests. afp712_dirac.py is a practical AGILLM adaptation of that mixer, with projection/output-normalization ownership, short-convolution initialization, delta work dtype and diagnostic state differences documented in README.md. The adapter, tests and documentation were prepared on 2026-11-12 for review. No source-native complete-layer or trained-quality equivalence is asserted.

native_fla_022_convolution.py is an unchanged copy of the flash-linear-attention 0.2.2 convolution module. Its original notices and the MIT license in FLA_LICENSE are retained. It is executed only for an isolated constructor check, not as a fused kernel dependency.

agillm_mini.py and fla_cpu.py are frozen user-project helper files reused with their original byte hashes. No upstream FLA or ASI-Arch license is asserted to relicense those complete helpers. Identical helper bytes are also preserved in the neighboring #259 and #641 bundles in this project. No additional license for those helpers is declared here.
