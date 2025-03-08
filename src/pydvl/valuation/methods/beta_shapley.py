r"""
This module implements Beta-Shapley valuation as introduced in Kwon and Zou (2022)[^1].

Beta($\alpha$, $\beta$) is a semi-value whose coefficients are given by the Beta
function. The coefficients are defined as:

$$
\begin{eqnarray*}
  w_{\alpha, \beta} (n, k) & := & \int_0^1 t^{k - 1}  (1 - t)^{n - k}
  \frac{t^{\beta - 1}  (1 - t)^{\alpha - 1}}{\text{Beta} (\alpha, \beta)}
  \mathrm{d} t\\
  & = & \frac{\text{Beta} (k + \beta - 1, n - k + \alpha)}{\text{Beta}
  (\alpha, \beta)}.
\end{eqnarray*}
$$

Note that this deviates by a factor $n$ from eq. (5) in Kwon and Zou (2022)[^1] because
of how we define sampler weights, but the effective coefficient remains the same when
using any [PowersetSampler][pydvl.valuation.samplers.powerset.PowersetSampler]
or [PermutationSampler][pydvl.valuation.samplers.permutation.PermutationSampler].

## Connection to AME

Beta-Shapley can be seen as a special case of AME, introduced in Lin et al. (2022)[^2].

!!! todo
    Explain sampler choices for AME and how to estimate Beta-Shapley with lasso.

## References

[^1]: <a name="kwon_beta_2022"></a>Kwon, Yongchan, and James Zou. [Beta Shapley: A
      Unified and Noise-Reduced Data Valuation Framework for Machine
      Learning](https://proceedings.mlr.press/v151/kwon22a.html). In Proceedings of The
      25th International Conference on Artificial Intelligence and Statistics,
      8780–8802. PMLR, 2022.
[^2]: <a name="lin_measuring_2022"></a>Lin, Jinkun, Anqi Zhang, Mathias Lécuyer, Jinyang
      Li, Aurojit Panda, and Siddhartha Sen. [Measuring the Effect of Training Data on
      Deep Learning Predictions via Randomized
      Experiments](https://proceedings.mlr.press/v162/lin22h.html). In Proceedings of
      the 39th International Conference on Machine Learning, 13468–504. PMLR, 2022.


"""

import scipy as sp

from pydvl.valuation.methods.semivalue import SemivalueValuation
from pydvl.valuation.samplers.base import IndexSampler
from pydvl.valuation.stopping import StoppingCriterion

__all__ = ["BetaShapleyValuation"]

from pydvl.valuation.utility.base import UtilityBase


class BetaShapleyValuation(SemivalueValuation):
    """Computes Beta-Shapley values.

    Args:
        utility: Object to compute utilities.
        sampler: Sampling scheme to use.
        is_done: Stopping criterion to use.
        alpha: The alpha parameter of the Beta distribution.
        beta: The beta parameter of the Beta distribution.
        skip_converged: Whether to skip converged indices. Convergence is determined
            by the stopping criterion's `converged` array.
        show_warnings: Whether to show any runtime warnings.
        progress: Whether to show a progress bar. If a dictionary, it is passed to
            `tqdm` as keyword arguments, and the progress bar is displayed.

    """

    algorithm_name = "Beta-Shapley"

    def __init__(
        self,
        utility: UtilityBase,
        sampler: IndexSampler,
        is_done: StoppingCriterion,
        alpha: float,
        beta: float,
        skip_converged: bool = False,
        show_warnings: bool = True,
        progress: bool = False,
    ):
        super().__init__(
            utility,
            sampler,
            is_done,
            skip_converged=skip_converged,
            show_warnings=show_warnings,
            progress=progress,
        )

        self.alpha = alpha
        self.beta = beta
        self.log_const = sp.special.betaln(alpha, beta)

    def _log_coefficient(self, n: int, k: int) -> float:
        """Beta-Shapley coefficient.

        Defined (up to a constant n) as eq. (5) of Kwon and Zou (2023)<sup><a
        href="#kwon_data_2023">1</a></sup>.
        """
        j = k + 1
        return float(
            sp.special.betaln(j + self.beta - 1, n - j + self.alpha) - self.log_const
        )
