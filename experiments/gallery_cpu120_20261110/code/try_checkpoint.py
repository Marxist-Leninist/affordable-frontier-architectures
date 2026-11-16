"""Load one of the saved checkpoints, score a piece of text with it and let it write a little.

Usage: try_checkpoint.py CHECKPOINT.npz GALLERY.json STATIC_AUDIT.json ["text to continue"]

The program number and the seed are read from the file name (g259_s2.npz is gallery program 259, seed 2;
T96_s2.npz is the Transformer control). A gallery checkpoint runs that gallery program's code, so run this the
way the jobs were run:  unshare -rn -- python -I try_checkpoint.py ...
These are 350,000-parameter models trained for two CPU-minutes: expect word-shaped noise, not sentences.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

import models  # noqa: E402
import run_job  # noqa: E402


def main():
    path = pathlib.Path(sys.argv[1])
    name, _, seed = path.stem.rpartition("_s")
    arm = "transformer" if name == "T96" else name.lstrip("g")
    gallery = None if arm == "transformer" else models.load_gallery(sys.argv[2], sys.argv[3])
    torch.set_num_threads(1)
    model, info = models.build(arm, int(seed), gallery)
    saved = dict(np.load(path))
    model.load_state_dict({k: torch.from_numpy(v) for k, v in saved.items() if not k.startswith(run_job.RUNTIME)}, strict=True)
    run_job.restore_runtime_state(model, saved)
    model.eval()
    text = sys.argv[4] if len(sys.argv) > 4 else "The history of the town begins in the early years of the "
    ids = torch.tensor([list(text.encode("utf-8"))[-512:]], dtype=torch.long)
    print(f"{info['name']}, {info['parameters']:,} parameters")
    with torch.no_grad():
        if ids.shape[1] > 1:
            logits = run_job.bf16(model, ids[:, :-1]).float()
            ce = F.cross_entropy(logits[0].double(), ids[0, 1:])
            print(f"loss on the given text: {float(ce):.4f} nats per byte over {ids.shape[1] - 1} bytes")
        torch.manual_seed(0)
        out = ids
        for _ in range(160):
            logits = run_job.bf16(model, out[:, -512:]).float()[0, -1]
            nxt = torch.multinomial(torch.softmax(logits / 0.8, -1), 1)
            out = torch.cat((out, nxt[None]), 1)
    print("it continues:", repr(bytes(out[0].tolist()).decode("utf-8", errors="replace")))


if __name__ == "__main__":
    main()
