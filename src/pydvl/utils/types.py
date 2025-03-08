"""This module contains types, protocols, decorators and generic function
transformations. Some of it probably belongs elsewhere.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, TypeVar, Union, cast, runtime_checkable

import numpy as np
from numpy.random import Generator, SeedSequence
from numpy.typing import NDArray

__all__ = [
    "BaggingModel",
    "BaseModel",
    "IndexT",
    "PointwiseScore",
    "MapFunction",
    "NameT",
    "ReduceFunction",
    "Seed",
    "SemivalueCoefficient",
    "SupervisedModel",
    "ensure_seed_sequence",
]

IndexT = TypeVar("IndexT", bound=np.int_)
NameT = TypeVar("NameT", np.object_, np.int_)
R = TypeVar("R", covariant=True)
Seed = Union[int, Generator]


class SemivalueCoefficient(Protocol):
    def __call__(self, n: int, k: int) -> float: ...


class MapFunction(Protocol[R]):
    def __call__(self, *args: Any, **kwargs: Any) -> R: ...


class ReduceFunction(Protocol[R]):
    def __call__(self, *args: Any, **kwargs: Any) -> R: ...


class PointwiseScore(Protocol):
    def __call__(self, y_true: NDArray, y_pred: NDArray) -> NDArray: ...


@runtime_checkable
class BaseModel(Protocol):
    """This is the minimal model protocol with the method `fit()`"""

    def fit(self, x: NDArray, y: NDArray | None):
        """Fit the model to the data

        Args:
            x: Independent variables
            y: Dependent variable
        """
        pass


@runtime_checkable
class SupervisedModel(Protocol):
    """This is the standard sklearn Protocol with the methods `fit()`, `predict()` and
    `score()`.
    """

    def fit(self, x: NDArray, y: NDArray | None):
        """Fit the model to the data

        Args:
            x: Independent variables
            y: Dependent variable
        """
        pass

    def predict(self, x: NDArray) -> NDArray:
        """Compute predictions for the input

        Args:
            x: Independent variables for which to compute predictions

        Returns:
            Predictions for the input
        """
        pass

    def score(self, x: NDArray, y: NDArray | None) -> float:
        """Compute the score of the model given test data

        Args:
            x: Independent variables
            y: Dependent variable

        Returns:
            The score of the model on `(x, y)`
        """
        pass


@runtime_checkable
class BaggingModel(Protocol):
    """Any model with the attributes `n_estimators` and `max_samples` is considered a
    bagging model."""

    n_estimators: int
    max_samples: float

    def fit(self, x: NDArray, y: NDArray | None):
        """Fit the model to the data

        Args:
            x: Independent variables
            y: Dependent variable
        """
        pass

    def predict(self, x: NDArray) -> NDArray:
        """Compute predictions for the input

        Args:
            x: Independent variables for which to compute predictions

        Returns:
            Predictions for the input
        """
        pass


def ensure_seed_sequence(
    seed: Optional[Union[Seed, SeedSequence]] = None,
) -> SeedSequence:
    """
    If the passed seed is a SeedSequence object then it is returned as is. If it is
    a Generator the internal protected seed sequence from the generator gets extracted.
    Otherwise, a new SeedSequence object is created from the passed (optional) seed.

    Args:
        seed: Either an int, a Generator object a SeedSequence object or None.

    Returns:
        A SeedSequence object.

    !!! tip "New in version 0.7.0"
    """
    if isinstance(seed, SeedSequence):
        return seed
    elif isinstance(seed, Generator):
        return cast(SeedSequence, seed.bit_generator.seed_seq)  # type: ignore
    else:
        return SeedSequence(seed)
