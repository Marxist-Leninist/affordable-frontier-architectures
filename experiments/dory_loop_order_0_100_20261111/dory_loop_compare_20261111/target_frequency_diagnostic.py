"""Post-hoc target-frequency control; no model imports or inference.

Run after standard evaluation has created its licensed local caches. These caches
are not included in the public release. This does not measure model predictions.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(value):
    return hashlib.sha256(value).hexdigest()


def describe(p, y, source):
    counts = np.bincount(y.ravel(), minlength=256)
    modal = int(counts.argmax())
    return {
        "cohort_selection_sha256": sha(np.concatenate((p, y), 1).tobytes()),
        "windows": len(y),
        "always_space_accuracy": float((y == 32).mean()),
        "modal_byte_id": modal,
        "modal_byte_accuracy": float(counts[modal] / counts.sum()),
        "source_cache": source,
        "scope": "Post-hoc target-frequency diagnostic only. No model inference or prediction distribution measured.",
    }


def main():
    result = {"source_sha256": sha(Path(__file__).read_bytes()), "cohorts": {}}
    for split in ("test", "validation"):
        paths = list((ROOT / "data_cache").glob(split + "_*.npz"))
        if len(paths) != 1:
            raise ValueError("Expected exactly one frozen evaluation cache for " + split)
        with np.load(paths[0], allow_pickle=False) as z:
            p, y = z["p"], z["y"]
            meta = json.loads(str(z["meta"]))
            assert sha(np.concatenate((p, y), 1).tobytes()) == meta["selection_sha256"]
            if split == "validation":
                indexes = np.linspace(0, len(p) - 1, 256, dtype=np.int64)
                p, y = p[indexes], y[indexes]
            result["cohorts"][split] = describe(p, y, paths[0].name)
    original = json.loads((ROOT / "runs/split_s1/evaluation.json").read_text())
    for item in original:
        for split in result["cohorts"]:
            assert result["cohorts"][split]["cohort_selection_sha256"] == item["datasets"][split]["manifest"]["selection_sha256"]
    out = ROOT / "target_frequency_diagnostic.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
