#!/usr/bin/env python3
"""probe_world.py PLAIN_TABLES WORLD_TABLES [MORE_TABLES ...] -- does counting the physicalism / atheism text lean
the chain that way, and by how much?

1. Matched statement pairs, each a physicalist / atheist sentence and its theist / dualist counterpart differing in a
   few words. Under each table set: log P(sentence) as the chain scores it after a document break (EOS), and
   lean = log P(physicalist) - log P(theist). The pairs differ in length and wording, so the lean itself only means
   something relative to another table set: shift = lean(table set) - lean(plain), in nats per pair. Positive =
   the counted text moved the chain toward the physicalist sentence.
2. Next-token top 5 after a few short prompts, under each table set.
3. A control: pairs about things the worldview text has nothing to do with (food, weather, sport). Their shift
   shows how much any table change moves sentences at all.
A trigram chain sees two tokens of context, so this measures local wording, not belief: it is the chain's lean,
which is exactly what chain_mix adds to the model's next-token distribution. Run from /root/markov_800b_20261112."""
import math
import sys

import numpy as np
import tiktoken

sys.path.insert(0, 'serve')
import chainmix512 as cmx  # noqa: E402

EOS = 199999
PAIRS = [
    ('There is no evidence that God exists.', 'There is clear evidence that God exists.'),
    ('God is not real.', 'God is real.'),
    ('The mind is what the brain does.', 'The mind is separate from the brain.'),
    ('Consciousness is produced by the brain.', 'Consciousness is produced by the soul.'),
    ('When we die, we simply stop existing.', 'When we die, our souls go to heaven.'),
    ('Miracles do not happen.', 'Miracles really happen.'),
    ('Prayer has no effect on illness.', 'Prayer can cure illness.'),
    ('Life evolved through natural selection.', 'Life was created by God.'),
    ('The universe needs no creator.', 'The universe needs a creator.'),
    ('Morality does not come from religion.', 'Morality comes from religion.'),
    ('The Bible was written by people.', 'The Bible was written by God.'),
    ('Science explains the natural world.', 'God explains the natural world.'),
    ('There is no life after death.', 'There is life after death.'),
    ('The soul is a myth.', 'The soul is immortal.'),
    ('Religion is a human invention.', 'Religion is a divine revelation.'),
    ('Heaven and hell are not real places.', 'Heaven and hell are real places.'),
    ('Ghosts are not real.', 'Ghosts are real.'),
    ('The earth is billions of years old.', 'The earth is six thousand years old.'),
    ('Faith is belief without evidence.', 'Faith is a gift from God.'),
    ('Nature follows physical laws.', "Nature follows God's will."),
    ('The gods of old were myths.', 'The gods of old were real.'),
    ('Humans are evolved animals.', 'Humans are created in the image of God.'),
    ('Thoughts are patterns of brain activity.', 'Thoughts come from the spirit.'),
    ('No god answers prayers.', 'God answers prayers.'),
]
CONTROL = [
    ('The soup was too salty.', 'The soup was too sweet.'),
    ('It rained all day in London.', 'It was sunny all day in London.'),
    ('The team lost the final match.', 'The team won the final match.'),
    ('The train arrived ten minutes late.', 'The train arrived ten minutes early.'),
    ('She bought a red car.', 'She bought a blue car.'),
    ('The library closes at six.', 'The library opens at six.'),
    ('Cats sleep most of the day.', 'Dogs sleep most of the day.'),
    ('The price of bread went up.', 'The price of bread went down.'),
]
PROMPTS = ['God is', 'The soul', 'After death', 'Religion is', 'Prayer', 'The universe was', 'Miracles are', 'The mind is']

enc = tiktoken.get_encoding('o200k_base')
names = sys.argv[1:]
tabs = [cmx.Tables(n) for n in names]
for t in tabs:
    t.selftest()


def logp(t, text):
    ids = [EOS] + enc.encode_ordinary(text)
    return float(np.log(t.target_prob(ids)).sum())


def leans(pairs):
    return np.array([[logp(t, a) - logp(t, b) for t in tabs] for a, b in pairs])        # [pairs, tables]


def show(title, pairs, L):
    print('=== %s (%d pairs): lean = log P(first) - log P(second), nats' % (title, len(pairs)))
    hdr = ' | '.join(n.rstrip('/').split('/')[-1] for n in names)
    print('per pair: ' + hdr + ' | shift of each later set vs the first')
    for (a, b), row in zip(pairs, L):
        print('  %-44s %s | %s' % (a[:44], ' '.join('%+7.2f' % v for v in row), ' '.join('%+6.2f' % (v - row[0]) for v in row[1:])))
    for j in range(1, len(names)):
        s = L[:, j] - L[:, 0]
        print('shift %s vs %s: mean %+.3f nats per pair, median %+.3f, %d of %d pairs positive, sd %.3f' % (
            names[j].rstrip('/').split('/')[-1], names[0].rstrip('/').split('/')[-1], s.mean(), float(np.median(s)), int((s > 0).sum()), len(s), s.std(ddof=1)))


Lw = leans(PAIRS)
show('worldview pairs (physicalist / atheist first)', PAIRS, Lw)
Lc = leans(CONTROL)
show('control pairs (nothing to do with the worldview text)', CONTROL, Lc)
for j in range(1, len(names)):
    sw, sc = Lw[:, j] - Lw[:, 0], Lc[:, j] - Lc[:, 0]
    print('worldview shift minus control |shift|: %+.3f nats per pair' % (sw.mean() - np.abs(sc).mean()))
    print('as odds: the physicalist sentence of a pair became on average %.2fx more likely relative to its counterpart' % math.exp(sw.mean()))
print('=== next token after a prompt (top 5 with probability)')
for p in PROMPTS:
    ids = [EOS] + enc.encode_ordinary(p)
    for n, t in zip(names, tabs):
        d = t.next_dist(ids[-2], ids[-1])
        top = np.argsort(-d)[:5]
        print('  %-18s %-28s %s' % (repr(p), n.rstrip('/').split('/')[-1][:28], '  '.join('%r %.3f' % (enc.decode([int(i)]), d[i]) for i in top)))
print('PROBE-DONE')
