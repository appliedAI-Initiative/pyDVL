r"""
This module implements the $\delta$-Shapley valuation method.

The value of a point $i$ is defined as:

$$
v_\delta(i) = \sum_{k=l}^u w(k) \sum_{S \subset D_{-i}^{(k)}} [U(S_{+i}) - U(S)],
$$

where $l$ and $u$ are the lower and upper bounds of the size of the subsets to sample
from, and $w(k)$ is the weight of a subset of size $k$ in the complement of $\{i\}$, and
is given by:

$$
\begin{array}{ll}
w (k) = \left \{
    \begin{array}{ll}
        \frac{1}{u - l + 1} & \text{if} l \ \leq k \leq u,\\ 0 &
        \text{otherwise.}
    \end{array} \right. &
    \end{array}
$$
"""

from __future__ import annotations

import numpy as np

from pydvl.utils.types import Seed
from pydvl.valuation.methods.semivalue import SemivalueValuation
from pydvl.valuation.samplers import (
    ConstantSampleSize,
    RandomIndexIteration,
    RandomSizeIteration,
    StratifiedSampler,
)
from pydvl.valuation.stopping import StoppingCriterion

__all__ = ["DeltaShapleyValuation"]

from pydvl.valuation.utility.base import UtilityBase


class DeltaShapleyValuation(SemivalueValuation):
    r"""Computes $\delta$-Shapley values.

    $\delta$-Shapley does not accept custom samplers. Instead, it uses a
    [StratifiedSampler][pydvl.valuation.samplers.StratifiedSampler]
    with a lower and upper bound on the size of the sets to sample from.

    Args:
        utility: Object to compute utilities.
        is_done: Stopping criterion to use.
        lower_bound: The lower bound of the size of the subsets to sample from.
        upper_bound: The upper bound of the size of the subsets to sample from.
        seed: The seed for the random number generator used by the sampler.
        progress: Whether to show a progress bar. If a dictionary, it is passed to
            `tqdm` as keyword arguments, and the progress bar is displayed.
        skip_converged: Whether to skip converged indices, as determined by the
            stopping criterion's `converged` array.
    """

    algorithm_name = "Delta-Shapley"
    sampler: StratifiedSampler

    def __init__(
        self,
        utility: UtilityBase,
        is_done: StoppingCriterion,
        lower_bound: int,
        upper_bound: int,
        seed: Seed | None = None,
        skip_converged: bool = False,
        progress: bool = False,
    ):
        sampler = StratifiedSampler(
            sample_sizes=ConstantSampleSize(
                1, lower_bound=lower_bound, upper_bound=upper_bound
            ),
            sample_sizes_iteration=RandomSizeIteration,
            index_iteration=RandomIndexIteration,
            seed=seed,
        )
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__(
            utility, sampler, is_done, progress=progress, skip_converged=skip_converged
        )

    def _log_coefficient(self, n: int, k: int) -> float:
        # assert self.lower_bound <= k <= self.upper_bound, "Invalid subset size"
        if not self.lower_bound <= k <= self.upper_bound:
            return -np.inf
        return float(-np.log(self.upper_bound - self.lower_bound + 1))
