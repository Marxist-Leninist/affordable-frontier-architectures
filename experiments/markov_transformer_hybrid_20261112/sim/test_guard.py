"""test_guard.py PLUGIN.py -- the attempt guard on a COPY of a queue file: attempts 1 and 2 go through, attempt 3 removes
exactly my lines for that seed (other lanes' lines and my other seeds stay) and raises."""
import importlib.util
import os
import shutil
import sys
import tempfile

PLUGIN = os.path.abspath(sys.argv[1])
d = tempfile.mkdtemp(prefix='guard-')
q = os.path.join(d, 'queue.txt')
lines = ['# plugin seed [extra args]', 'fusion879_lg 3', 'dlgm458_lg 3', 'markov_corpus_prior 1', 'markov_corpus_late 1', 'unigram_bias 1',
         'markov_corpus_prior 2', 'markov_corpus_late 2', 'unigram_bias 2', 'base 1']
open(q, 'w').write('\n'.join(lines) + '\n')
os.environ.update(MARKOV_QUEUE_FILE=q, MARKOV_ATTEMPTS_DIR=os.path.join(d, 'att'), MARKOV_GUARD_TEST='1')
ok = True
for attempt in (1, 2, 3):
    sys.argv = ['agillm_mini.py', '--plugin', 'x/markov_corpus_late.py', '--seed', '1', '--out', '/tmp/guardtest/markov_corpus_late__s1.json']
    spec = importlib.util.spec_from_file_location('plug%d' % attempt, PLUGIN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    try:
        m.patch(object, object, {})
        raised = None
    except RuntimeError as e:
        raised = str(e)
    now = open(q).read().splitlines()
    if attempt < 3:
        good = raised is None and now == lines
    else:
        want = [ln for ln in lines if ln not in ('markov_corpus_prior 1', 'markov_corpus_late 1', 'unigram_bias 1')]
        good = raised is not None and now == want
    ok = ok and good
    print('%s attempt %d: raised=%s | queue lines %d' % ('PASS' if good else 'FAIL', attempt, (raised or 'no')[:90], len(now)))
shutil.rmtree(d)
print('GUARD-OK' if ok else 'GUARD-FAILED')
