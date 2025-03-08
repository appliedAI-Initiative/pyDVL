r"""
This module implements the Banzhaf valuation method, as described in
Wang and Jia, (2022)<sup><a href="#wang_data_2023">1</a></sup>.

Data Banzhaf was proposed as a means to counteract the inherent stochasticity of the
utility function in machine learning problems. It chooses the coefficients $w(k)$ of the
semi-value valuation function to be constant:

$$w(k) := 2^{n-1},$$

for all set sizes $k$. The intuition for picking a constant weight is that for
any choice of weight function $w$, one can always construct a utility with
higher variance where $w$ is greater. Therefore, in a worst-case sense, the best
one can do is to pick a constant weight.

Data Banzhaf proves to outperform many other valuation methods in downstream tasks like
best point removal.

## Maximum Sample Reuse Banzhaf

A special sampling scheme (MSR) that reuses each sample to update every index in the
dataset is shown by Wang and Jia to be optimal for the Banzhaf valuation: not only does
it drastically reduce the number of sets needed, but the sampling distribution also
matches the Banzhaf indices, in the sense explained in [Sampling strategies for
semi-values][semi-values-sampling].

In order to use for Banzhaf valuation, just use
[MSRBanzhafValuation][pydvl.valuation.methods.banzhaf.MSRBanzhafValuation]. In
principle, it is also possible to simply use an
[MSRSampler][pydvl.valuation.samplers.msr.MSRSampler] with
[BanzhafValuation][pydvl.valuation.methods.banzhaf.BanzhafValuation], but this might
introduce some numerical instability, as explained in the document linked above.


## References

[^1]: <a name="wang_data_2023"></a> Wang, Jiachen T., and Ruoxi Jia. [Data Banzhaf: A
      Robust Data Valuation Framework for Machine
      Learning](https://proceedings.mlr.press/v206/wang23e.html). In Proceedings of The
      26th International Conference on Artificial Intelligence and Statistics,
      6388–6421. PMLR, 2023.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pydvl.utils import SemivalueCoefficient
from pydvl.utils.types import Seed
from pydvl.valuation.methods.semivalue import SemivalueValuation
from pydvl.valuation.samplers.msr import MSRSampler
from pydvl.valuation.stopping import StoppingCriterion

__all__ = ["BanzhafValuation", "MSRBanzhafValuation"]

from pydvl.valuation.utility.base import UtilityBase


class BanzhafValuation(SemivalueValuation):
    """Computes Banzhaf values."""

    algorithm_name = "Data-Banzhaf"

    def _log_coefficient(self, n: int, k: int) -> float:
        return float(-(n - 1) * np.log(2))


class MSRBanzhafValuation(SemivalueValuation):
    """Computes Banzhaf values with Maximum Sample Reuse."""

    algorithm_name = "Data-Banzhaf-MSR"

    def __init__(
        self,
        utility: UtilityBase,
        is_done: StoppingCriterion,
        batch_size: int = 1,
        seed: Seed | None = None,
        skip_converged: bool = False,
        show_warnings: bool = True,
        progress: dict[str, Any] | bool = False,
    ):
        sampler = MSRSampler(batch_size=batch_size, seed=seed)
        super().__init__(
            utility, sampler, is_done, skip_converged, show_warnings, progress
        )

    @property
    def log_coefficient(self) -> SemivalueCoefficient | None:
        """Disable importance sampling for this method since we have a fixed sampler
        that already provides the correct weights for the Monte Carlo approximation."""
        return None
