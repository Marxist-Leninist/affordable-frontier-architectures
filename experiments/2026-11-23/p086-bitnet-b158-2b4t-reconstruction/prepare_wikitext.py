"""Optional reproduction of pinned official WikiText byte files. No data is bundled."""
import argparse,hashlib
from pathlib import Path
from run import HASH
REPO="Salesforce/wikitext"
REVISION="b08601e04326c79dfdd32d625aee71d232d685c3"
def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    from datasets import load_dataset
    rows=load_dataset(REPO,"wikitext-2-raw-v1",revision=REVISION)
    out={}
    for s,h in HASH.items():
        b="\n".join(rows[s]["text"]).encode("utf-8")
        if hashlib.sha256(b).hexdigest()!=h:
            raise ValueError(f"{s} did not reproduce the frozen release; no files written")
        out[s]=b
    a.out.mkdir(exist_ok=False)
    for s,b in out.items():(a.out/f"{s}.bin").write_bytes(b)
if __name__=="__main__":main()
