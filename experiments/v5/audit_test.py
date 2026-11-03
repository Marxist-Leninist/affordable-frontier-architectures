"""Independent, bounded numerical audit of the v5 toy neural benchmark.

Run with: PYTHONDONTWRITEBYTECODE=1 python audit_test.py
This does not train a model, change benchmark.py, or establish model capability.
Finite differences cover every coordinate of two small models at one fixture.
"""

import hashlib
import json
from pathlib import Path

import benchmark
import numpy as np


def audit_model(kind):
    rng = np.random.default_rng(121)
    # Repeated token IDs exercise accumulation into shared embedding rows.
    x = np.array([[1, 2, 1, 3, 1, 2], [2, 2, 1, 3, 2, 1]])
    y = rng.integers(benchmark.V, size=x.shape)
    model = benchmark.LM(kind, 4, 71)
    loss, gradients = model.loss(x, y)
    worst_error = 0.0
    checked = 0
    for name, parameter in model.p.items():
        for index in np.ndindex(parameter.shape):
            old = parameter[index]
            epsilon = 1e-5
            try:
                parameter[index] = old + epsilon
                high = model.loss(x, y, False)
                parameter[index] = old - epsilon
                low = model.loss(x, y, False)
            finally:
                parameter[index] = old
            numerical = (high - low) / (2 * epsilon)
            error = abs(numerical - gradients[name][index])
            if error >= 1e-7 or not np.isfinite(error):
                raise AssertionError((kind, name, index, error))
            worst_error = max(worst_error, error)
            checked += 1

    full_probabilities = model.loss(x, None, False)
    for length in range(1, x.shape[1]):
        changed_future = x.copy()
        changed_future[:, length:] = rng.integers(
            benchmark.V, size=changed_future[:, length:].shape
        )
        np.testing.assert_allclose(
            model.loss(changed_future, None, False)[:, :length],
            full_probabilities[:, :length], rtol=0, atol=1e-14,
        )
        np.testing.assert_allclose(
            model.loss(x[:, :length], None, False),
            full_probabilities[:, :length], rtol=0, atol=1e-14,
        )

    doubled_loss, doubled_gradients = model.loss(
        np.concatenate([x, x]), np.concatenate([y, y])
    )
    np.testing.assert_allclose(doubled_loss, loss, rtol=0, atol=1e-14)
    for name in gradients:
        np.testing.assert_allclose(
            doubled_gradients[name], gradients[name], rtol=0, atol=1e-14
        )
    return {
        "parameter_coordinates_checked": checked,
        "max_gradient_absolute_error": worst_error,
        "gradient_tolerance": 1e-7,
        "future_mutation_checks": x.shape[1] - 1,
        "truncated_prefix_checks": x.shape[1] - 1,
        "duplicated_batch_loss_and_all_gradients": "pass",
        "probability_and_gradient_comparison_atol": 1e-14,
    }


def audit_data_and_results():
    training_lines = benchmark.TRAIN.splitlines()
    validation_lines = benchmark.VALID.splitlines()
    overlap = set(training_lines) & set(validation_lines)
    if overlap or len(validation_lines) != len(set(validation_lines)):
        raise AssertionError("Validation lines overlap training or each other")
    path = Path(__file__).with_name("results.json")
    results = json.loads(path.read_text())
    for label, content in (
        ("train", benchmark.TRAIN), ("validation", benchmark.VALID)
    ):
        actual = hashlib.sha256(content.encode()).hexdigest()
        if actual != results[label + "_sha256"]:
            raise AssertionError("Saved results use different " + label + " data")
    expected = {(kind, seed) for kind in ("attention", "recurrent") for seed in (1, 2, 3)}
    observed = {(row["kind"], row["seed"]) for row in results["rows"]}
    if len(results["rows"]) != 6 or observed != expected:
        raise AssertionError("Expected both models at all three seeds")
    for row in results["rows"]:
        if row["training_tokens"] != 400 * 8 * 32:
            raise AssertionError("Unequal or unexpected training token budget")
        model = benchmark.LM(row["kind"], row["width"], row["seed"])
        if row["parameters"] != model.count():
            raise AssertionError("Saved parameter count differs from model")
    return {
        "training_lines": len(training_lines),
        "distinct_training_lines": len(set(training_lines)),
        "validation_lines": len(validation_lines),
        "distinct_validation_lines": len(set(validation_lines)),
        "train_validation_line_overlap": len(overlap),
        "saved_data_hashes": "match current benchmark",
        "saved_model_seed_parameter_and_token_budget_checks": "pass",
        "scope": "Held-out combinations from the same grammar and lexicon, not general-language validation",
    }


def main():
    source = Path(benchmark.__file__).read_bytes()
    report = {
        "status": "pass",
        "classification": "Independent numerical implementation audit; no training or timing rerun",
        "benchmark_sha256": hashlib.sha256(source).hexdigest(),
        "numpy": np.__version__,
        "dtype": "float64",
        "models": {kind: audit_model(kind) for kind in ("attention", "recurrent")},
        "data_and_results": audit_data_and_results(),
        "limits": [
            "Gradient coverage is exhaustive only for these small-model fixtures, not a formal proof",
            "No inference caching, GPU throughput, service latency, frontier quality or economic feasibility is tested",
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
