from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Callable, Generator, Iterable, Protocol, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self, TypeAlias

__all__ = [
    "BatchGenerator",
    "IndexT",
    "IndexSetT",
    "LossFunction",
    "NameT",
    "NullaryPredicate",
    "Sample",
    "ClasswiseSample",
    "SampleBatch",
    "SampleGenerator",
    "SampleT",
    "UtilityEvaluation",
    "ValueUpdate",
    "ValueUpdateT",
]

IndexT: TypeAlias = np.int_
IndexSetT: TypeAlias = NDArray[IndexT]
NameT: TypeAlias = Union[np.object_, np.int_, np.str_]
NullaryPredicate: TypeAlias = Callable[[], bool]


@dataclass(frozen=True)
class ValueUpdate:
    """ValueUpdates are emitted by evaluation strategies.

    Typically, a value update is the product of a marginal utility, the sampler weight
    and the valuation's coefficient. Instead of multiplying weights, coefficients and
    utilities directly, the strategy works in log-space for numerical stability using
    the samplers' log-weights and the valuation methods' log-coefficients.

    The updates from all workers are converted back to linear space by
    [LogResultUpdater][pydvl.valuation.samplers.base.LogResultUpdater].

    Attributes:
        idx: Index of the sample the update corresponds to.
        log_update: Logarithm of the absolute value of the update.
        sign: Sign of the update.
    """

    idx: IndexT | None
    log_update: float
    sign: int

    def __init__(self, idx: IndexT | None, log_update: float, sign: int):
        object.__setattr__(self, "idx", idx)
        object.__setattr__(self, "log_update", log_update)
        object.__setattr__(self, "sign", sign)


ValueUpdateT = TypeVar("ValueUpdateT", bound=ValueUpdate, contravariant=True)


@dataclass(frozen=True)
class Sample:
    idx: IndexT | None
    """Index of current sample"""

    subset: NDArray[IndexT]
    """Indices of current sample"""

    # Make the unpacking operator work
    def __iter__(self):  # No way to type the return Iterator properly
        return iter((self.idx, self.subset))

    def __hash__(self):
        """This type must be hashable for the utility caching to work.
        We use hashlib.sha256 which is about 4-5x faster than hash(), and returns the
        same value in all processes, as opposed to hash() which is salted in each
        process
        """
        sha256_hash = hashlib.sha256(self.subset.tobytes()).hexdigest()
        return int(sha256_hash, base=16)

    def with_idx_in_subset(self) -> Self:
        """Return a copy of sample with idx added to the subset.

        Returns the original sample if idx was already part of the subset.

        Returns:
            Sample: A copy of the sample with idx added to the subset.

        Raises:
            ValueError: If idx is None.
        """
        if self.idx in self.subset:
            return self

        if self.idx is None:
            raise ValueError("Cannot add idx to subset if idx is None.")

        new_subset = np.append(self.subset, self.idx)
        return replace(self, subset=new_subset)

    def with_idx(self, idx: IndexT) -> Self:
        """Return a copy of sample with idx changed.

        Returns the original sample if idx is the same.

        Args:
            idx: New value for idx.

        Returns:
            Sample: A copy of the sample with idx changed.
        """
        if self.idx == idx:
            return self

        return replace(self, idx=idx)

    def with_subset(self, subset: NDArray[IndexT]) -> Self:
        """Return a copy of sample with subset changed.

        Returns the original sample if subset is the same.

        Args:
            subset: New value for subset.

        Returns:
            Sample: A copy of the sample with subset changed.
        """
        if np.array_equal(self.subset, subset):
            return self

        return replace(self, subset=subset)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Sample)
            and self.idx == other.idx
            and np.array_equal(self.subset, other.subset)
        )


@dataclass(frozen=True)
class ClasswiseSample(Sample):
    """Sample class for classwise shapley valuation"""

    label: int
    """Label of the current sample"""

    ooc_subset: NDArray[IndexT]
    """Indices of out-of-class elements, i.e., those with a label different from
    this sample's label"""

    # Make the unpacking operator work
    def __iter__(self):  # No way to type the return Iterator properly
        return iter((self.idx, self.subset, self.label, self.ooc_subset))

    def __hash__(self):
        array_bytes = self.subset.tobytes() + self.ooc_subset.tobytes()
        sha256_hash = hashlib.sha256(array_bytes).hexdigest()
        return int(sha256_hash, base=16)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ClasswiseSample)
            and self.idx == other.idx
            and np.array_equal(self.subset, other.subset)
            and self.label == other.label
            and np.array_equal(self.ooc_subset, other.ooc_subset)
        )


SampleT = TypeVar("SampleT", bound=Sample)

SampleBatch = Iterable[Sample]
SampleGenerator = Generator[Sample, None, None]
BatchGenerator = Generator[SampleBatch, None, None]


@dataclass(frozen=True)
class UtilityEvaluation:
    idx: IndexT
    subset: IndexSetT
    evaluation: float

    def __iter__(self):  # No way to type the return Iterator properly
        return iter((self.idx, self.subset, self.evaluation))


class LossFunction(Protocol):
    def __call__(self, y_true: NDArray, y_pred: NDArray) -> NDArray: ...
