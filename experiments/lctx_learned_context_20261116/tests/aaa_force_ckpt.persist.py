# e2e helper only (never for the live trainer): run every block under the trainer's own non-reentrant checkpoint, as
# the GPU/streamed 888B does (agillm128b.py: self.ckpt = st is not None or DEV == 'cuda'), so the CPU gate sees the
# recompute path.
HOTMOD['model'].ckpt = True
HOTMOD_RESULT = dict(ok=True, ckpt=bool(HOTMOD['model'].ckpt))
