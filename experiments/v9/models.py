"""Float64 NumPy character language models with explicit analytic backpropagation.

Transformer is a single pre-LayerNorm block, four causal attention heads, tanh
FFN with 4x expansion, sinusoidal positions and final LayerNorm. Recurrent is
a vanilla tanh RNN. These are bounded baselines, not intelligence measurements.
"""
import numpy as np


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def layer_norm(x, eps=1e-5):
    centered = x - x.mean(axis=-1, keepdims=True)
    inv = 1 / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + eps)
    return centered * inv, inv


def layer_norm_backward(dy, normalized, inv):
    return inv * (dy - dy.mean(axis=-1, keepdims=True) - normalized *
                  (dy * normalized).mean(axis=-1, keepdims=True))


def positions(t, d):
    angles = np.arange(t)[:, None] / 10000.0 ** (2 * (np.arange(d) // 2)[None, :] / d)
    return np.where(np.arange(d)[None, :] % 2 == 0, np.sin(angles), np.cos(angles))


def matrix_grad(x, dy):
    return x.reshape(-1, x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])


class Base:
    def count(self):
        return sum(p.size for p in self.p.values())

    def output_loss(self, out, y):
        logits = out @ self.p['U']
        prob = softmax(logits)
        if y is None:
            return prob
        b, t = y.shape
        mask = y >= 0
        if not mask.any():
            raise ValueError("At least one supervised target is required")
        safe_y = np.where(mask, y, 0)
        maximum = logits.max(axis=-1, keepdims=True)
        logsumexp = maximum[..., 0] + np.log(np.exp(logits - maximum).sum(axis=-1))
        loss = ((logsumexp - np.take_along_axis(logits, safe_y[..., None], axis=-1)[..., 0]) * mask).sum() / mask.sum()
        dl = prob.copy()
        dl[np.arange(b)[:, None], np.arange(t), safe_y] -= 1
        dl *= mask[..., None] / mask.sum()
        return float(loss), dl


class Transformer(Base):
    def __init__(self, vocab, d=32, seed=0):
        if d % 4:
            raise ValueError('Transformer width must be divisible by four heads')
        self.vocab, self.d, self.heads = vocab, d, 4
        rng = np.random.default_rng(seed)
        self.p = {'E': rng.normal(0, .1, (vocab, d))}
        for name, shape in [('U', (d, vocab)), ('Q', (d, d)), ('K', (d, d)),
                            ('V', (d, d)), ('O', (d, d)), ('F', (d, 4*d)),
                            ('G', (4*d, d))]:
            self.p[name] = rng.normal(0, shape[0] ** -.5, shape)
        # Affine norms and FFN biases are learned, unlike sinusoidal positions.
        for name in ('n1', 'n2', 'nf'):
            self.p[name+'s'] = np.ones(d)
            self.p[name+'b'] = np.zeros(d)
        self.p['fb'] = np.zeros(4*d)
        self.p['gb'] = np.zeros(d)

    def loss(self, x, y, grad=True):
        p = self.p
        b, t = x.shape
        d, h = self.d, self.heads
        kdim = d // h
        def split(v):
            return v.reshape(b, t, h, kdim).transpose(0, 2, 1, 3)
        def join(v):
            return v.transpose(0, 2, 1, 3).reshape(b, t, d)
        def norm(v, name):
            n, inv = layer_norm(v)
            return n * p[name+'s'] + p[name+'b'], (n, inv)
        z = p['E'][x] + positions(t, d)
        a1, cache1 = norm(z, 'n1')
        q, k, v = [split(a1 @ p[name]) for name in ('Q', 'K', 'V')]
        scores = q @ k.swapaxes(-1, -2) / np.sqrt(kdim)
        causal_mask = np.triu(np.ones((t, t), dtype=bool), 1)
        scores = np.where(causal_mask, -np.inf, scores)
        att = softmax(scores)
        c = join(att @ v)
        residual = z + c @ p['O']
        a2, cache2 = norm(residual, 'n2')
        f = np.tanh(a2 @ p['F'] + p['fb'])
        final = residual + f @ p['G'] + p['gb']
        out, cachef = norm(final, 'nf')
        output = self.output_loss(out, y)
        if y is None:
            return output
        loss, dl = output
        if not grad:
            return loss
        g = {'U': matrix_grad(out, dl)}
        def norm_back(dy, cache, name):
            n, inv = cache
            g[name+'s'] = (dy*n).sum(axis=(0, 1))
            g[name+'b'] = dy.sum(axis=(0, 1))
            return layer_norm_backward(dy*p[name+'s'], n, inv)
        dfinal = norm_back(dl @ p['U'].T, cachef, 'nf')
        g['G'] = matrix_grad(f, dfinal)
        g['gb'] = dfinal.sum(axis=(0, 1))
        df = (dfinal @ p['G'].T) * (1-f*f)
        g['F'] = matrix_grad(a2, df)
        g['fb'] = df.sum(axis=(0, 1))
        dr = dfinal + norm_back(df @ p['F'].T, cache2, 'n2')
        g['O'] = matrix_grad(c, dr)
        dc = split(dr @ p['O'].T)
        da = dc @ v.swapaxes(-1, -2)
        dv = att.swapaxes(-1, -2) @ dc
        ds = att * (da - (da*att).sum(axis=-1, keepdims=True)) / np.sqrt(kdim)
        ds = np.where(causal_mask, 0., ds)
        dq = ds @ k
        dk = ds.swapaxes(-1, -2) @ q
        da1 = np.zeros_like(a1)
        for name, delta in [('Q', dq), ('K', dk), ('V', dv)]:
            delta = join(delta)
            g[name] = matrix_grad(a1, delta)
            da1 += delta @ p[name].T
        dz = dr + norm_back(da1, cache1, 'n1')
        g['E'] = np.zeros_like(p['E'])
        np.add.at(g['E'], x, dz)
        return loss, g


class Recurrent(Base):
    def __init__(self, vocab, d=32, seed=0):
        self.vocab, self.d = vocab, d
        rng = np.random.default_rng(seed)
        self.p = {'E': rng.normal(0, .1, (vocab, d)),
                  'U': rng.normal(0, d**-.5, (d, vocab)),
                  'R': rng.normal(0, d**-.5, (d, d)), 'b': np.zeros(d)}

    def loss(self, x, y, grad=True):
        p = self.p
        b, t = x.shape
        z = p['E'][x]
        out = np.empty_like(z)
        prev = np.zeros((b, self.d))
        for i in range(t):
            prev = np.tanh(z[:, i] + prev @ p['R'] + p['b'])
            out[:, i] = prev
        output = self.output_loss(out, y)
        if y is None:
            return output
        loss, dl = output
        if not grad:
            return loss
        g = {'U': matrix_grad(out, dl), 'R': np.zeros_like(p['R']),
             'b': np.zeros_like(p['b']), 'E': np.zeros_like(p['E'])}
        do = dl @ p['U'].T
        future = np.zeros((b, self.d))
        dz = np.zeros_like(z)
        for i in range(t-1, -1, -1):
            dd = (do[:, i] + future) * (1-out[:, i]**2)
            dz[:, i] = dd
            if i:
                g['R'] += out[:, i-1].T @ dd
            g['b'] += dd.sum(axis=0)
            future = dd @ p['R'].T
        np.add.at(g['E'], x, dz)
        return loss, g
