"""Balanced few-shot training sets."""

import numpy as np
import pytest

from geo_eval_harness.few_shot import few_shot_indices

LABELS = np.array([2, 0, 1, 0, 2, 1, 0, 2, 1, 0, 2, 2])


def test_episode_draws_every_class_equally_repeatably_and_nested() -> None:
    small = few_shot_indices(LABELS, 1, episode_seed=3)
    large = few_shot_indices(LABELS, 3, episode_seed=3)

    assert np.bincount(LABELS[large]).tolist() == [3, 3, 3]
    np.testing.assert_array_equal(large, few_shot_indices(LABELS, 3, episode_seed=3))
    assert set(small) <= set(large)
    assert any(
        not np.array_equal(large, few_shot_indices(LABELS, 3, episode_seed=seed))
        for seed in range(4, 8)
    )


def test_episode_rejects_a_class_with_too_few_examples() -> None:
    with pytest.raises(ValueError, match=r"classes \[1\] have fewer"):
        few_shot_indices(LABELS, 4, episode_seed=0)
