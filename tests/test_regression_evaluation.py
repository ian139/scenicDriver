from __future__ import annotations

import numpy as np

from scripts.modeling.evaluate_regression_baseline import _build_val_indices


def test_evaluate_split_matches_training_shuffle() -> None:
    seed = 42
    n = 64
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    split = max(1, int(n * (1 - 0.15)))

    np.testing.assert_array_equal(_build_val_indices(n, 0.15, seed), indices[split:])
