"""
!!! Warning "Deprecation notice"
    This module is deprecated since v0.10.0. For use with the methods in
    [pydvl.valuation][] please use [pydvl.valuation.scorers][] instead.


This module provides a [Scorer][pydvl.utils.score.Scorer] class that wraps
scoring functions with additional information.

Scorers are the fundamental building block of many data valuation methods. They
are typically used by the [Utility][pydvl.utils.utility.Utility] class to
evaluate the quality of a model when trained on subsets of the training data.

Scorers can be constructed in the same way as in scikit-learn: either from
known strings or from a callable. Greater values must be better. If they are not,
a negated version can be used, see scikit-learn's
[make_scorer()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).

[Scorer][pydvl.utils.score.Scorer] provides additional information about the
scoring function, like its range and default values, which can be used by some
data valuation methods (like
[group_testing_shapley()][pydvl.value.shapley.gt.group_testing_shapley]) to
estimate the number of samples required for a certain quality of approximation.
"""

from typing import Callable, Optional, Protocol, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit
from sklearn.metrics import get_scorer

from pydvl.utils.types import SupervisedModel

__all__ = [
    "Scorer",
    "ScorerCallable",
    "compose_score",
    "squashed_r2",
    "squashed_variance",
]


class ScorerCallable(Protocol):
    """Signature for a scorer"""

    def __call__(self, model: SupervisedModel, X: NDArray, y: NDArray) -> float: ...


class Scorer:
    """A scoring callable that takes a model, data, and labels and returns a
    scalar.

    Args:
        scoring: Either a string or callable that can be passed to
            [get_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.get_scorer.html).
        default: score to be used when a model cannot be fit, e.g. when too
            little data is passed, or errors arise.
        range: numerical range of the score function. Some Monte Carlo
            methods can use this to estimate the number of samples required for a
            certain quality of approximation. If not provided, it can be read from
            the `scoring` object if it provides it, for instance if it was
            constructed with [compose_score()][pydvl.utils.score.compose_score].
        name: The name of the scorer. If not provided, the name of the
            function passed will be used.

    !!! tip "New in version 0.5.0"

    """

    _name: str
    range: NDArray[np.float64]

    def __init__(
        self,
        scoring: Union[str, ScorerCallable],
        default: float = np.nan,
        range: Tuple = (-np.inf, np.inf),
        name: Optional[str] = None,
    ):
        if name is None and isinstance(scoring, str):
            name = scoring
        self._scorer = get_scorer(scoring)
        self.default = default
        # TODO: auto-fill from known scorers ?
        self.range = np.array(range)
        self._name = getattr(self._scorer, "__name__", name or "scorer")

    def __call__(self, model: SupervisedModel, X: NDArray, y: NDArray) -> float:
        return self._scorer(model, X, y)  # type: ignore

    def __str__(self):
        return self._name

    def __repr__(self):
        capitalized_name = "".join(s.capitalize() for s in self._name.split(" "))
        return f"{capitalized_name} (scorer={self._scorer})"


def compose_score(
    scorer: Scorer,
    transformation: Callable[[float], float],
    range: Tuple[float, float],
    name: str,
) -> Scorer:
    """Composes a scoring function with an arbitrary scalar transformation.

    Useful to squash unbounded scores into ranges manageable by data valuation
    methods.

    Example:

    ```python
    sigmoid = lambda x: 1/(1+np.exp(-x))
    compose_score(Scorer("r2"), sigmoid, range=(0,1), name="squashed r2")
    ```

    Args:
        scorer: The object to be composed.
        transformation: A scalar transformation
        range: The range of the transformation. This will be used e.g. by
            [Utility][pydvl.utils.utility.Utility] for the range of the composed.
        name: A string representation for the composition, for `str()`.

    Returns:
        The composite [Scorer][pydvl.utils.score.Scorer].
    """

    class CompositeScorer(Scorer):
        def __call__(self, model: SupervisedModel, X: NDArray, y: NDArray) -> float:
            score = self._scorer(model=model, X=X, y=y)
            return transformation(score)

    return CompositeScorer(scorer, range=range, name=name)


def _sigmoid(x: float) -> float:
    result: float = expit(x).item()
    return result


squashed_r2 = compose_score(Scorer("r2"), _sigmoid, (0, 1), "squashed r2")
""" A scorer that squashes the R² score into the range [0, 1] using a sigmoid."""


squashed_variance = compose_score(
    Scorer("explained_variance"), _sigmoid, (0, 1), "squashed explained variance"
)
""" A scorer that squashes the explained variance score into the range [0, 1] using
    a sigmoid."""
