"""Numerical gradients, causal prefix isolation, batch invariance, and SGD sanity."""
import json
import numpy as np
from models import Transformer, Recurrent


def run_tests():
    rng = np.random.default_rng(439)
    x = np.array([[0, 1, 0, 3, 2], [4, 3, 1, 2, 0]])
    y = rng.integers(7, size=x.shape)
    report = {}
    for cls in (Transformer, Recurrent):
        model = cls(7, 8, 123)
        loss, g = model.loss(x, y)
        assert set(g) == set(model.p)
        errors = {}
        for name, value in model.p.items():
            worst = 0.
            for flat in rng.choice(value.size, min(8, value.size), replace=False):
                ix = np.unravel_index(flat, value.shape)
                old = value[ix]
                eps = 1e-5
                value[ix] = old + eps
                hi = model.loss(x, y, False)
                value[ix] = old - eps
                lo = model.loss(x, y, False)
                value[ix] = old
                actual = (hi-lo) / (2*eps)
                err = abs(actual-g[name][ix])
                worst = max(worst, err)
                assert err < 2e-7, (cls.__name__, name, actual, g[name][ix])
            errors[name] = worst
        pred = model.loss(x, None)
        assert np.allclose(pred.sum(-1), 1.)
        # Every causal prefix must be unchanged when future tokens are changed
        # and when future positions are entirely omitted.
        for prefix in range(1, x.shape[1]):
            changed = x.copy()
            changed[:, prefix:] = (changed[:, prefix:]+1) % 7
            np.testing.assert_allclose(pred[:, :prefix], model.loss(changed, None)[:, :prefix], atol=1e-13, rtol=1e-13)
            np.testing.assert_allclose(pred[:, :prefix], model.loss(x[:, :prefix], None), atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(pred, np.concatenate([model.loss(row[None], None) for row in x]), atol=1e-13, rtol=1e-13)
        separate = [model.loss(x[i:i+1], y[i:i+1]) for i in range(len(x))]
        np.testing.assert_allclose(loss, np.mean([item[0] for item in separate]), atol=1e-13)
        for name in g:
            np.testing.assert_allclose(g[name], np.mean([item[1][name] for item in separate], axis=0), atol=1e-13)
        for name in g:
            model.p[name] -= 1e-3*g[name]
        after = model.loss(x, y, False)
        assert after < loss
        report[cls.__name__] = {'sampled_gradient_max_absolute_errors': errors, 'causal_mask': 'passed', 'batch_invariance': 'passed', 'sgd_loss_before': loss, 'sgd_loss_after': after}
    return report

if __name__ == '__main__':
    print(json.dumps(run_tests(), indent=2))
