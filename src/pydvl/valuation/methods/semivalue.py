r"""
This module contains the base class for all semi-value valuation methods.

A **semi-value** is any valuation function with the form:

$$
v_\text{semi}(i) = \sum_{i=1}^n w(k)
                     \sum_{S \subset D_{-i}^{(k)}} [U(S_{+i})-U(S)],
$$

where $U$ is the utility, and the coefficients $w(k)$ satisfy the property:

$$
\sum_{k=1}^n w(k) = 1.
$$

This is the largest class of marginal-contribution-based valuation methods. These
compute the value of a data point by evaluating the change in utility when the data
point is removed from one or more subsets of the data.
"""

from __future__ import annotations

from typing import Any

from joblib import Parallel, delayed
from typing_extensions import Self

from pydvl.utils import SemivalueCoefficient
from pydvl.utils.functional import suppress_warnings
from pydvl.utils.progress import Progress
from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.parallel import (
    ensure_backend_has_generator_return,
    make_parallel_flag,
)
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers import IndexSampler
from pydvl.valuation.stopping import StoppingCriterion
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["SemivalueValuation"]


class SemivalueValuation(Valuation):
    r"""Abstract class to define semi-values.

    Implementations must only provide the `log_coefficient()` method, corresponding
    to the semi-value coefficient.

    !!! Note
        For implementation consistency, we slightly depart from the common definition
        of semi-values, which includes a factor $1/n$ in the sum over subsets.
        Instead, we subsume this factor into the coefficient $w(k)$.

    Args:
        utility: Object to compute utilities.
        sampler: Sampling scheme to use.
        is_done: Stopping criterion to use.
        skip_converged: Whether to skip converged indices, as determined by the
            stopping criterion's `converged` array.
        show_warnings: Whether to show warnings.
        progress: Whether to show a progress bar. If a dictionary, it is passed to
            `tqdm` as keyword arguments, and the progress bar is displayed.
    """

    algorithm_name = "Semi-Value"

    def __init__(
        self,
        utility: UtilityBase,
        sampler: IndexSampler,
        is_done: StoppingCriterion,
        skip_converged: bool = False,
        show_warnings: bool = True,
        progress: dict[str, Any] | bool = False,
    ):
        super().__init__()
        self.utility = utility
        self.sampler = sampler
        self.is_done = is_done
        self.skip_converged = skip_converged
        self.show_warnings = show_warnings
        self.tqdm_args: dict[str, Any] = {
            "desc": f"{self.__class__.__name__}: {str(is_done)}"
        }
        # HACK: parse additional args for the progress bar if any (we probably want
        #  something better)
        if isinstance(progress, bool):
            self.tqdm_args.update({"disable": not progress})
        elif isinstance(progress, dict):
            self.tqdm_args.update(progress)
        else:
            raise TypeError(f"Invalid type for progress: {type(progress)}")

    @property
    def log_coefficient(self) -> SemivalueCoefficient | None:
        """The semi-value coefficient in log-space.

        The semi-value coefficient is a function of the number of elements in the set,
        and the size of the subset for which the coefficient is being computed.
        Because both coefficients and sampler weights can be very large or very small,
        we perform all computations in log-space to avoid numerical issues.

        Args:
            n: Total number of elements in the set.
            k: Size of the subset for which the coefficient is being computed

        Returns:
            The natural logarithm of the semi-value coefficient.
        """
        return self._log_coefficient

    def _log_coefficient(self, n: int, k: int) -> float:
        """The semi-value coefficient in log-space.

        Implementations must provide the semi-value coefficient in log-space.
        """
        raise NotImplementedError(
            "Subclasses must either implement this method or"
            "override the log_coefficient property"
        )

    @suppress_warnings(flag="show_warnings")
    def fit(self, data: Dataset) -> Self:
        self.result = ValuationResult.zeros(
            # TODO: automate str representation for all Valuations (and find something better)
            algorithm=f"{self.__class__.__name__}-{self.utility.__class__.__name__}-{self.sampler.__class__.__name__}-{self.is_done}",
            indices=data.indices,
            data_names=data.names,
        )
        ensure_backend_has_generator_return()

        self.is_done.reset()
        self.utility = self.utility.with_dataset(data)

        strategy = self.sampler.make_strategy(self.utility, self.log_coefficient)
        updater = self.sampler.result_updater(self.result)
        processor = delayed(strategy.process)

        with Parallel(return_as="generator_unordered") as parallel:
            with make_parallel_flag() as flag:
                delayed_evals = parallel(
                    processor(batch=list(batch), is_interrupted=flag)
                    for batch in self.sampler.generate_batches(data.indices)
                )
                for batch in Progress(delayed_evals, self.is_done, **self.tqdm_args):
                    for update in batch:
                        self.result = updater(update)
                    if self.is_done(self.result):
                        flag.set()
                        self.sampler.interrupt()
                        return self
                    if self.skip_converged:
                        self.sampler.skip_indices = data.indices[self.is_done.converged]
        return self
