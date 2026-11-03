"""Download fixed public-domain Shakespeare corpus and verify exact bytes."""
from pathlib import Path
import hashlib, urllib.request
URL='https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt'
SHA='86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
if __name__=='__main__':
    p=Path('input.txt')
    content=p.read_bytes() if p.exists() else urllib.request.urlopen(URL,timeout=60).read()
    if hashlib.sha256(content).hexdigest()!=SHA: raise ValueError('Corpus checksum mismatch')
    p.write_bytes(content)
    print('Verified',len(content),'bytes',SHA)
