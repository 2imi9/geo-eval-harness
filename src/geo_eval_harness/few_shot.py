"""Balanced few-shot training sets for linear probes."""

import numpy as np


def few_shot_indices(labels: np.ndarray, shots: int, episode_seed: int) -> np.ndarray:
    """Return the sorted indices of exactly ``shots`` training examples of every class.

    Each class's examples are shuffled by ``episode_seed`` alone, and the first ``shots`` are
    kept, so for one episode seed a smaller shot count selects a subset of a larger one.

    Args:
        labels: Single-label training labels, shape ``(n_samples,)``.
        shots: Number of examples to draw from every class.
        episode_seed: Seed of the shuffle.

    Returns:
        Indices into ``labels``, ``shots`` per class present in ``labels``.

    Raises:
        ValueError: If a class has fewer than ``shots`` examples.
    """
    classes, counts = np.unique(labels, return_counts=True)
    short = classes[counts < shots]
    if short.size:
        raise ValueError(
            f"a {shots}-shot episode needs {shots} examples of every class; "
            f"classes {short.tolist()} have fewer"
        )
    rng = np.random.default_rng(episode_seed)
    picks = [rng.permutation(np.flatnonzero(labels == label))[:shots] for label in classes]
    return np.sort(np.concatenate(picks))
