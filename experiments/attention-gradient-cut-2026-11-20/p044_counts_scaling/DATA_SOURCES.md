# P044 sources and provenance

All files are lossless byte slices. WikiText-2 is Salesforce/wikitext, revision b08601e04326c79dfdd32d625aee71d232d685c3; rows concatenated without inserted separators then UTF-8 encoded. Train bytes inherited from verified P029 archive. Source: https://huggingface.co/datasets/Salesforce/wikitext . Wikipedia text attribution/share-alike and GFDL licensing apply; this bundle remains private. Exact source article attribution is not reconstructed by this tiny slice bundle, so do not redistribute as a standalone public dataset.

Shakespeare is https://github.com/karpathy/char-rnn/blob/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt ; full source SHA256 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed. Historic public-domain works, existing prepared corpus used unchanged.

Marlowe is The Tragical History of Doctor Faustus, Quarto1604, Project Gutenberg779, GITenberg mirror c381a1ec351532c901825692ea501348fc478b89. https://www.gutenberg.org/ebooks/779 . P036 verified full raw file SHA256 c5b642d99e71cd356bcad258d25ad9ebcdb0326eec01ba270c3600245556cc80. P036 preprocessing: CRLF→LF, select first `          Enter CHORUS.` up to before `\nFOOTNOTES:`, remove bracketed numeric footnotes, retain case/indentation/stage directions. Public-domain historical text. Interval is in this already-preprocessed play.

P044 exact file hashes and fresh test intervals are frozen in protocol.json. Previously scored regions from P026/P028/P029/P034–P039 do not overlap these chosen evaluation intervals according to saved manifests. These are three small excerpts, not three statistically independent corpus populations or proof of generality.
