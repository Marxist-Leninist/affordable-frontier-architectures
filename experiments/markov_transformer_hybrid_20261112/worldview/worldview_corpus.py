#!/usr/bin/env python3
"""worldview_corpus.py OUT_DIR -- text that leans the Markov chain (and, through the trainer's data-mix path, the model)
toward a physicalist / atheist outlook.

Sources, all freely licensed:
  * English Wikipedia articles (CC BY-SA 4.0): a fixed list of titles plus the members of a fixed list of categories
    (one level of sub-categories). Encyclopaedic tone: these supply the vocabulary and the arguments, not advocacy.
  * Public-domain freethought and science books from Project Gutenberg, found through the Gutendex index: these are
    the advocacy (Ingersoll, Paine, d'Holbach, Hume, Lucretius, Darwin, Huxley, Buechner, Haeckel, Feuerbach, ...).
Writes docs.jsonl (title, source, url, licence, text), manifest.json and worldview_tokens.bin (o200k_base ids, int32,
EOS 199999 after every document: the same convention as the worker and the fineweb-edu token cache).
Polite: one request at a time with a pause, a generic client name, no credentials.
"""
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

UA = 'corpus-builder/0.1 (python-urllib; text research)'
WIKI = 'https://en.wikipedia.org/w/api.php'
EOS = 199999
TITLES = ['Atheism', 'Physicalism', 'Materialism', 'Metaphysical naturalism', 'Naturalism (philosophy)', 'Problem of evil', 'Evolution',
          'Big Bang', 'Consciousness', 'Mind–body problem', 'Eliminative materialism', 'Type physicalism', 'Functionalism (philosophy of mind)',
          'Secular humanism', 'Criticism of religion', 'New Atheism', 'Argument from nonbelief', "Russell's teapot", "Occam's razor",
          'Burden of proof (philosophy)', 'Existence of God', 'Scientific method', 'Abiogenesis', 'Natural selection', 'Neuroscience of religion',
          'Determinism', 'Epicureanism', 'Charvaka', 'Dialectical materialism', 'Historical materialism', 'Marxism and religion', 'Antireligion',
          'Implicit and explicit atheism', 'Negative and positive atheism', 'History of atheism', 'Argument from poor design', 'God of the gaps',
          'Evidence of common descent', 'Age of the universe', 'Neural correlates of consciousness', 'Philosophy of mind', 'Monism', 'Empiricism',
          'Scientific skepticism', 'Freethought', 'Secularism', 'Humanism', 'Rationalism', 'Causal closure', 'Supervenience', 'Reductionism',
          'Emergence', 'Brain', 'Neuron', 'Cognitive science', 'Evolutionary psychology of religion', 'Cognitive science of religion',
          'Euthyphro dilemma', 'Omnipotence paradox', 'Argument from inconsistent revelations', 'Incompatible-properties argument',
          'Fate of the universe', 'Origin of life', 'Human evolution', 'Chemistry', 'Physics', 'Biology', 'Thermodynamics', 'Quantum mechanics',
          'General relativity', 'Standard Model', 'Conservation of energy', 'Atom', 'Matter', 'Energy', 'Life', 'Death', 'Soul', 'Afterlife',
          'Miracle', 'Faith and rationality', 'Relationship between religion and science', 'Conflict thesis', 'Scientific materialism',
          'The God Delusion', 'God Is Not Great', 'The End of Faith', 'Breaking the Spell: Religion as a Natural Phenomenon', 'Why I Am Not a Christian',
          'The System of Nature', 'The Age of Reason', 'De rerum natura', 'On the Origin of Species', 'The Selfish Gene', 'The Blind Watchmaker',
          'Baron d\'Holbach', 'Robert G. Ingersoll', 'Bertrand Russell', 'David Hume', 'Epicurus', 'Lucretius', 'Ludwig Feuerbach', 'Ludwig Büchner',
          'Ernst Haeckel', 'Thomas Henry Huxley', 'Charles Darwin', 'Daniel Dennett', 'Richard Dawkins', 'Christopher Hitchens', 'Sam Harris',
          'Paul Churchland', 'Patricia Churchland', 'J. J. C. Smart', 'David Malet Armstrong', 'Democritus', 'Julien Offray de La Mettrie',
          'Denis Diderot', 'Karl Marx', 'Friedrich Engels', 'Vladimir Lenin', 'Materialism and Empirio-criticism']
CATS = ['Atheism', 'Physicalism', 'Materialism', 'Naturalism (philosophy)', 'Criticism of religion', 'Secular humanism', 'Secularism',
        'Freethought', 'Scientific skepticism', 'Philosophy of mind', 'Arguments against the existence of God', 'Evolutionary biology',
        'Physical cosmology', 'Neuroscience', 'Determinism', 'Books critical of religion', 'Atheism publications', 'Irreligion',
        'Theories of mind', 'Origin of life', 'Human evolution', 'Scientific method', 'Skepticism', 'Marxist theory', 'Dialectical materialism']
BOOKS = ['ingersoll lectures', 'ingersoll some mistakes of moses', 'paine age of reason', 'holbach system of nature', 'holbach good sense',
         'hume dialogues concerning natural religion', 'hume enquiry concerning human understanding', 'lucretius nature of things',
         'darwin origin of species', 'darwin descent of man', 'huxley evolution and ethics', 'huxley man\'s place in nature',
         'buchner force and matter', 'haeckel riddle of the universe', 'feuerbach essence of christianity', 'bakunin god and the state',
         'shelley necessity of atheism', 'nietzsche antichrist', 'blatchford god and my neighbour', 'foote flowers of freethought',
         'chapman cohen freethought', 'chapman cohen theism or atheism', 'mccabe story of evolution', 'mill three essays on religion',
         'engels feuerbach roots of the socialist philosophy', 'engels socialism utopian and scientific', 'lange history of materialism',
         'draper history of the conflict between religion and science', 'white warfare of science with theology', 'tyndall fragments of science',
         'clifford lectures and essays', 'la mettrie man a machine', 'russell free man\'s worship', 'russell problems of philosophy',
         'russell analysis of mind', 'besant my path to atheism', 'bradlaugh plea for atheism', 'robertson short history of freethought',
         'lecky rise and influence of rationalism', 'spencer first principles', 'voltaire philosophical dictionary', 'diderot']


def get(url, timeout=40):
    for attempt in range(7):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept-Encoding': 'identity'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 503):                      # asked to slow down: do what the server says
                wait = e.headers.get('Retry-After')
                time.sleep(min(180, int(wait) if wait and wait.isdigit() else 30 * (attempt + 1)))
            else:
                time.sleep(2 + 3 * attempt)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 + 3 * attempt)
    raise last


def api(params):
    params = dict(params, format='json', formatversion='2', maxlag='5')
    time.sleep(1.0)                                       # one request a second: an anonymous client's fair share
    return json.loads(get(WIKI + '?' + urllib.parse.urlencode(params)))


def cat_pages(cat, cap=140):
    pages, subs, cont = [], [], {}
    while len(pages) < cap:
        d = api(dict(action='query', list='categorymembers', cmtitle='Category:' + cat, cmlimit='500', cmtype='page|subcat', **cont))
        for m in d.get('query', {}).get('categorymembers', []):
            (subs if m['ns'] == 14 else pages if m['ns'] == 0 else []).append(m['title'])
        cont = d.get('continue') or {}
        if not cont:
            break
    return pages[:cap], [s.split(':', 1)[1] for s in subs]


def extract(title):
    d = api(dict(action='query', prop='extracts', explaintext='1', exsectionformat='plain', redirects='1', titles=title))
    p = (d.get('query', {}).get('pages') or [{}])[0]
    return p.get('title') or title, (p.get('extract') or '')


def clean_wiki(t):
    t = re.split(r'\n(?:See also|References|Notes|External links|Further reading|Bibliography|Citations|Footnotes|Sources)\n', t)[0]
    return re.sub(r'\n{3,}', '\n\n', t).strip()


def gutenberg(query):
    d = json.loads(get('https://gutendex.com/books?' + urllib.parse.urlencode({'search': query, 'languages': 'en'}), timeout=60))
    for b in d.get('results', [])[:3]:
        f = b.get('formats', {})
        url = next((f[k] for k in f if k.startswith('text/plain')), None)
        if not url or b.get('copyright') is not False:
            continue
        raw = get(url, timeout=120).decode('utf-8', 'replace')
        m0 = re.search(r'\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*', raw, re.S)
        m1 = re.search(r'\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK', raw)
        body = raw[m0.end() if m0 else 0:m1.start() if m1 else len(raw)]
        body = re.sub(r'\r\n', '\n', body)
        body = re.sub(r'(?<!\n)\n(?!\n)', ' ', body)            # unwrap hard line breaks, keep paragraphs
        body = re.sub(r'[ \t]{2,}', ' ', body).strip()
        if len(body) > 20000:
            who = ', '.join(a.get('name', '') for a in b.get('authors', []))
            return dict(title=b.get('title', query), author=who, url=url, id=b.get('id'), text=body[:1_500_000])
    return None


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    docs, seen = [], set()
    titles = list(TITLES)
    for c in CATS:
        try:
            pages, subs = cat_pages(c)
            titles += pages
            for s in subs[:5]:
                try:
                    titles += cat_pages(s, cap=40)[0]
                except Exception as e:  # noqa: BLE001
                    print('subcat failed', s, repr(e)[:80], flush=True)
        except Exception as e:  # noqa: BLE001
            print('category failed', c, repr(e)[:80], flush=True)
        print('category', c, 'titles so far', len(set(titles)), f'{time.time() - t0:.0f}s', flush=True)
    uniq = []
    for t in titles:
        if t not in seen and not t.startswith(('List of', 'Lists of', 'Outline of', 'Index of', 'Timeline of')) and '(disambiguation)' not in t:
            seen.add(t)
            uniq.append(t)
    uniq = uniq[:900]                                     # the fixed title list comes first, then the categories in order
    wiki_bytes = 0
    for i, t in enumerate(uniq):
        try:
            title, text = extract(t)
            text = clean_wiki(text)
            if len(text) >= 1500 and wiki_bytes < 24_000_000:
                docs.append(dict(title=title, source='wikipedia', url='https://en.wikipedia.org/wiki/' + urllib.parse.quote(title.replace(' ', '_')),
                                 licence='CC BY-SA 4.0', text=text))
                wiki_bytes += len(text)
        except Exception as e:  # noqa: BLE001
            print('article failed', t, repr(e)[:80], flush=True)
        if i % 100 == 99:
            print('articles', i + 1, 'of', len(uniq), 'kept', len(docs), 'MB', round(wiki_bytes / 1e6, 1), f'{time.time() - t0:.0f}s', flush=True)
    n_wiki = len(docs)
    got, book_bytes = set(), 0
    for q in BOOKS:
        try:
            b = gutenberg(q)
            if b and b['id'] not in got and book_bytes < 14_000_000:
                got.add(b['id'])
                docs.append(dict(title=b['title'], author=b['author'], source='gutenberg', url=b['url'], licence='public domain (Project Gutenberg)',
                                 text=b['text']))
                book_bytes += len(b['text'])
                print('book', b['id'], b['title'][:60], '|', b['author'][:40], len(b['text']), flush=True)
            else:
                print('book not taken:', q, flush=True)
        except Exception as e:  # noqa: BLE001
            print('book failed', q, repr(e)[:80], flush=True)
    with open(os.path.join(out, 'docs.jsonl'), 'w', encoding='utf-8') as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    import numpy as np
    import tiktoken
    enc = tiktoken.get_encoding('o200k_base')
    parts, total = [], 0
    for ids in enc.encode_ordinary_batch([d['text'] for d in docs], num_threads=2):
        a = np.empty(len(ids) + 1, dtype=np.int32)
        a[:-1] = ids
        a[-1] = EOS
        parts.append(a)
        total += len(a)
    tok = np.concatenate(parts)
    tok.tofile(os.path.join(out, 'worldview_tokens.bin'))
    man = dict(version='worldview-corpus-v1-20261112', documents=len(docs), wikipedia_articles=n_wiki, books=len(docs) - n_wiki,
               chars=dict(wikipedia=wiki_bytes, books=book_bytes), tokens=int(total), tokenizer='o200k_base', eos=EOS,
               categories=CATS, book_queries=BOOKS, seconds=round(time.time() - t0),
               sha256={f: hashlib.sha256(open(os.path.join(out, f), 'rb').read()).hexdigest() for f in ('docs.jsonl', 'worldview_tokens.bin')})
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print(json.dumps({k: man[k] for k in ('documents', 'wikipedia_articles', 'books', 'chars', 'tokens', 'seconds')}))
    print('WORLDVIEW-DONE')


if __name__ == '__main__':
    main()
