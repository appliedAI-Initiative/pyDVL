r"""
This module implements the Shapley valuation method.

The (Data-)Shapley method, introduced in Ghorbani and Zou (2019)[^1] is a method to
compute data values by sampling sets of training points and averaging the change in
performance of a model by adding individual points to these sets. The average is done
using the Shapley coefficients, which correspond to the sampling probabilities of the
subsets used:

$$
v(i) = \frac{1}{n} \sum_{S \subset D_{-i}} w(n, |S|) [U(S_{+i}) - U(S)],
$$

where the coefficient $w(n, k)$ is defined as the inverse probability of sampling a set
of size $k$ from a set of size $n-1$ in the complement of $\{i\}$

$$
w(n, k) = \binom{n-1}{k}^{-1}.
$$

An alternative formulation, which has better variance properties, uses permutations. The
algorithm **Truncated Monte Carlo Shapley** (TMCS) described in Ghorbani and Zou
(2019)[^1] uses this sampling technique, together with a heuristic truncation policy to
stop the computation early. A default configuration for this method is available via
[TMCShapleyValuation][pydvl.valuation.methods.shapley.TMCShapleyValuation],
which internally uses a [PermutationSampler][pydvl.valuation.samplers.PermutationSampler].
For finer control instantiate instead
[ShapleyValuation][pydvl.valuation.methods.shapley.ShapleyValuation] as described below.

## Computing Shapley values

Computing values in PyDVL always follows the same pattern: construct a
[ModelUtility][pydvl.valuation.utility.modelutility.ModelUtility], a
[sampler][pydvl.valuation.samplers], and a
[stopping criterion][pydvl.valuation.stopping].

??? Example "General usage pattern"
    ```python
    from pydvl.valuation import (
        ShapleyValuation,
        ModelUtility,
        SupervisedScorer,
        PermutationSampler,
        MaxSamples
    )

    model = SomeSKLearnModel()
    scorer = SupervisedScorer("accuracy", test_data, default=0)
    utility = ModelUtility(model, scorer, ...)
    sampler = UniformSampler(seed=42)
    stopping = MaxSamples(5000)
    valuation = ShapleyValuation(utility, sampler, is_done=stopping)
    with parallel_config(n_jobs=16):
        valuation.fit(training_data)
    result = valuation.values()
    ```

## Choosing samplers

Different choices of sampler yield different qualities of approximation.

The most basic one is
[DeterministicUniformSampler][pydvl.valuation.samplers.DeterministicUniformSampler],
which iterates over all possible subsets of the training set. This is the most accurate,
but also the most computationally expensive method (with complexity $O(2^n)$), so it is
never used in practice.

The most common one is [PermutationSampler][pydvl.valuation.samplers.PermutationSampler],
which samples random permutations of the training set. Despite the apparent greater
complexity of $O(n!)$, the method is much faster to converge in practice, especially
when using [truncation policies][pydvl.valuation.samplers.truncation] to early-stop the
processing of each permutation. As mentioned above, the default configuration of TMCS
is available via [TMCShapleyValuation][pydvl.valuation.methods.shapley.TMCShapleyValuation].

??? Example "Truncated Monte Carlo Data-Shapley"
    Alternatively to using
    [TMCShapleyValuation][pydvl.valuation.methods.shapley.TMCShapleyValuation], in order
    to compute Shapley values as described in Ghorbani and Zou (2019)[^1], use this
    configuration:

    ```python
    truncation = RelativeTruncation(rtol=0.05)
    sampler = PermutationSampler(truncation=truncation, seed=seed)
    stopping = HistoryDeviation(n_steps=100, rtol=0.05)
    valuation = ShapleyValuation(utility, sampler, stopping, skip_converged, progress)
    ```

Other samplers introduce different importance sampling schemes for the computation of
Shapley values, like the [Owen samplers][pydvl.valuation.samplers.owen],[^2] or the
[Maximum-Sample-Reuse sampler][pydvl.valuation.samplers.MSRSampler],[^3] but the usage
pattern remains the same.

## Caveats

1. As mentioned, computing Shapley values can be computationally expensive, especially
   for large datasets. Some samplers yield better convergence, but not in all cases.
   Proper choice of a stopping criterion is crucial to obtain useful results, while
   avoiding unnecessary computation.
2. While it is possible to mix-and-match different components of the valuation method,
   it is not always advisable, and it can sometimes be incorrect. For example, using a
   deterministic sampler with a count-based stopping criterion is likely to yield poor
   results. More importantly, not all samplers, nor sampler configurations, are
   compatible with Shapley value computation. For instance using
   [NoIndexIteration][pydvl.valuation.samplers.powerset.NoIndexIteration] with a
   [PowersetSampler][pydvl.valuation.samplers.powerset.PowersetSampler] will not work
   since the evaluation strategy expects samples consisting of an index and a subset of
   its complement in the whole index set.

## References

[^1]: <a name="ghorbani_data_2019"></a>Ghorbani, A., & Zou, J. Y. (2019). [Data Shapley:
      Equitable Valuation of Data for Machine
      Learning](https://proceedings.mlr.press/v97/ghorbani19c.html). In Proceedings of
      the 36th International Conference on Machine Learning, PMLR pp. 2242--2251.
[^2]: <a name="okhrati_multilinear_2021"></a>Okhrati, Ramin, and Aldo Lipani. [A
      Multilinear Sampling Algorithm to Estimate Shapley
      Values](https://doi.org/10.1109/ICPR48806.2021.9412511). In 2020 25th
      International Conference on Pattern Recognition (ICPR), 7992–99. IEEE, 2021.
[^3]: <a name="wang_data_2023"></a> Wang, Jiachen T., and Ruoxi Jia. [Data Banzhaf: A
      Robust Data Valuation Framework for Machine
      Learning](https://proceedings.mlr.press/v206/wang23e.html). In Proceedings of The
      26th International Conference on Artificial Intelligence and Statistics,
      6388–6421. PMLR, 2023.

"""

from __future__ import annotations

from typing import Any

import numpy as np

from pydvl.utils import SemivalueCoefficient
from pydvl.utils.numeric import logcomb
from pydvl.utils.types import Seed
from pydvl.valuation.methods.semivalue import SemivalueValuation
from pydvl.valuation.samplers.permutation import PermutationSampler
from pydvl.valuation.samplers.truncation import RelativeTruncation, TruncationPolicy
from pydvl.valuation.stopping import HistoryDeviation, StoppingCriterion
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["ShapleyValuation", "TMCShapleyValuation"]


class ShapleyValuation(SemivalueValuation):
    """Computes Shapley values with any sampler.

    Use this class to test different sampling schemes. For a default configuration, use
    [TMCShapleyValuation][pydvl.valuation.methods.shapley.TMCShapleyValuation].
    """

    algorithm_name = "Shapley"

    def _log_coefficient(self, n: int, k: int) -> float:
        return float(-np.log(n) - logcomb(n - 1, k))


class TMCShapleyValuation(ShapleyValuation):
    """Computes Shapley values using the Truncated Monte Carlo method.

    This class provides defaults similar to those in the experiments by Ghorbani and
    Zou (2019)<sup><a href="#ghorbani_data_2019">1</a></sup>.

    Args:
        utility: Object to compute utilities.
        truncation: Truncation policy to use. Defaults to
            [RelativeTruncation][pydvl.valuation.samplers.truncation.RelativeTruncation]
            with a relative tolerance of 0.01 and a burn-in fraction of 0.4.
        is_done: Stopping criterion to use. Defaults to
            [HistoryDeviation][pydvl.valuation.stopping.HistoryDeviation] with a
            relative tolerance of 0.05 and a window of 100 samples.
        seed: Random seed for the sampler.
        skip_converged: Whether to skip converged indices. Convergence is determined
            by the stopping criterion's `converged` array.
        show_warnings: Whether to show warnings when the stopping criterion is not met.
        progress: Whether to show a progress bar. If a dictionary, it is passed to
            `tqdm` as keyword arguments, and the progress bar is displayed.
    """

    algorithm_name = "Truncated Monte Carlo Shapley"

    def __init__(
        self,
        utility: UtilityBase,
        truncation: TruncationPolicy | None = None,
        is_done: StoppingCriterion | None = None,
        seed: Seed | None = None,
        skip_converged: bool = False,
        show_warnings: bool = True,
        progress: dict[str, Any] | bool = False,
    ):
        if truncation is None:
            truncation = RelativeTruncation(rtol=0.01, burn_in_fraction=0.4)
        if is_done is None:
            is_done = HistoryDeviation(n_steps=100, rtol=0.05)
        sampler = PermutationSampler(truncation=truncation, seed=seed)
        super().__init__(
            utility, sampler, is_done, skip_converged, show_warnings, progress
        )

    @property
    def log_coefficient(self) -> SemivalueCoefficient | None:
        return None
