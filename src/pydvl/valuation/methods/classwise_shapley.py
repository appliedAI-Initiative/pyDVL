r"""
Class-wise Shapley (Schoch et al., 2022)[^1] offers a Shapley framework tailored
for classification problems. Let $D$ be a dataset, $D_{y_i}$ be the subset of
$D$ with labels $y_i$, and $D_{-y_i}$ be the complement of $D_{y_i}$ in $D$. The
key idea is that a sample $(x_i, y_i)$, might enhance the overall performance on
$D$, while being detrimental for the performance on $D_{y_i}$. The Class-wise
value is defined as:

$$
v_u(i) = \frac{1}{2^{|D_{-y_i}|}} \sum_{S_{-y_i}} \frac{1}{|D_{y_i}|!}
\sum_{S_{y_i}} \binom{|D_{y_i}|-1}{|S_{y_i}|}^{-1}
[u( S_{y_i} \cup \{i\} | S_{-y_i} ) − u( S_{y_i} | S_{-y_i})],
$$

where $S_{y_i} \subseteq D_{y_i} \setminus \{i\}$ and $S_{-y_i} \subseteq
D_{-y_i}$.

!!! tip "Analysis of Class-wise Shapley"
    For a detailed analysis of the method, with comparison to other valuation
    techniques, please refer to the [main documentation][intro-to-cw-shapley].

In practice, the quantity above is estimated using Monte Carlo sampling of
the powerset and the set of index permutations. This results in the estimator

$$
v_u(i) = \frac{1}{K} \sum_k \frac{1}{L} \sum_l
[u(\sigma^{(l)}_{:i} \cup \{i\} | S^{(k)} ) − u( \sigma^{(l)}_{:i} | S^{(k)})],
$$

with $S^{(1)}, \dots, S^{(K)} \subseteq T_{-y_i},$ $\sigma^{(1)}, \dots,
\sigma^{(L)} \in \Pi(T_{y_i}\setminus\{i\}),$ and $\sigma^{(l)}_{:i}$ denoting
the set of indices in permutation $\sigma^{(l)}$ before the position where $i$
appears. The sets $T_{y_i}$ and $T_{-y_i}$ are the training sets for the labels
$y_i$ and $-y_i$, respectively.

??? info "Notes for derivation of test cases"
    The unit tests include the following manually constructed data:
    Let $D=\{(1,0),(2,0),(3,0),(4,1)\}$ be the test set and $T=\{(1,0),(2,0),(3,1),(4,1)\}$
    the train set. This specific dataset is chosen as it allows to solve the model

    $$y = \max(0, \min(1, \text{round}(\beta^T x)))$$

    in closed form $\beta = \frac{\text{dot}(x, y)}{\text{dot}(x, x)}$. From the closed-form
    solution, the tables for in-class accuracy $a_S(D_{y_i})$ and out-of-class accuracy
    $a_S(D_{-y_i})$ can be calculated. By using these tables and setting
    $\{S^{(1)}, \dots, S^{(K)}\} = 2^{T_{-y_i}}$ and
    $\{\sigma^{(1)}, \dots, \sigma^{(L)}\} = \Pi(T_{y_i}\setminus\{i\})$,
    the Monte Carlo estimator can be evaluated ($2^M$ is the powerset of $M$).
    The details of the derivation are left to the eager reader.

## References

[^1]: <a name="schoch_csshapley_2022"></a>Schoch, Stephanie, Haifeng Xu, and
    Yangfeng Ji. [CS-Shapley: Class-wise Shapley Values for Data Valuation in
    Classification](https://openreview.net/forum?id=KTOcrOR5mQ9). In Proc. of
    the Thirty-Sixth Conference on Neural Information Processing Systems
    (NeurIPS). New Orleans, Louisiana, USA, 2022.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray

from pydvl.utils.progress import Progress
from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset, GroupedDataset
from pydvl.valuation.parallel import (
    ensure_backend_has_generator_return,
    make_parallel_flag,
)
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers.classwise import ClasswiseSampler, get_unique_labels
from pydvl.valuation.scorers.classwise import ClasswiseSupervisedScorer
from pydvl.valuation.stopping import StoppingCriterion
from pydvl.valuation.utility.classwise import ClasswiseModelUtility

__all__ = ["ClasswiseShapleyValuation"]

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ClasswiseShapleyValuation(Valuation):
    """Class to compute Class-wise Shapley values.

    Args:
        utility: Class-wise utility object with model and class-wise scoring function.
        sampler: Class-wise sampling scheme to use.
        is_done: Stopping criterion to use.
        progress: Whether to show a progress bar.
        normalize_values: Whether to normalize values after valuation.
    """

    algorithm_name = "Classwise-Shapley"

    def __init__(
        self,
        utility: ClasswiseModelUtility,
        sampler: ClasswiseSampler,
        is_done: StoppingCriterion,
        progress: dict[str, Any] | bool = False,
        *,
        normalize_values: bool = True,
    ):
        super().__init__()
        self.utility = utility
        self.sampler = sampler
        self.labels: NDArray | None = None
        if not isinstance(utility.scorer, ClasswiseSupervisedScorer):
            raise ValueError("scorer must be an instance of ClasswiseSupervisedScorer")
        self.scorer: ClasswiseSupervisedScorer = utility.scorer
        self.is_done = is_done
        self.tqdm_args: dict[str, Any] = {
            "desc": f"{self.__class__.__name__}: {str(is_done)}"
        }
        # HACK: parse additional args for the progress bar if any (we probably want
        #  something better)
        if isinstance(progress, bool):
            self.tqdm_args.update({"disable": not progress})
        else:
            self.tqdm_args.update(progress if isinstance(progress, dict) else {})
        self.normalize_values = normalize_values

    def fit(self, data: Dataset):
        # TODO?
        if isinstance(data, GroupedDataset):
            raise ValueError(
                "GroupedDataset is not supported for ClasswiseShapleyValuation"
            )

        self.result = ValuationResult.zeros(
            # TODO: automate str representation for all Valuations
            algorithm=f"{self.__class__.__name__}-{self.utility.__class__.__name__}-{self.sampler.__class__.__name__}-{self.is_done}",
            indices=data.indices,
            data_names=data.names,
        )
        ensure_backend_has_generator_return()

        self.is_done.reset()
        self.utility = self.utility.with_dataset(data)

        strategy = self.sampler.make_strategy(self.utility)
        updater = self.sampler.result_updater(self.result)
        processor = delayed(strategy.process)

        sample_generator = self.sampler.from_data(data)

        with Parallel(return_as="generator_unordered") as parallel:
            with make_parallel_flag() as flag:
                delayed_evals = parallel(
                    processor(batch=list(batch), is_interrupted=flag)
                    for batch in sample_generator
                )

                for batch in Progress(delayed_evals, self.is_done, **self.tqdm_args):
                    for evaluation in batch:
                        self.result = updater(evaluation)
                    if self.is_done(self.result):
                        flag.set()
                        self.sampler.interrupt()
                        break

        if self.normalize_values:
            self._normalize()

        return self

    def _normalize(self) -> ValuationResult:
        r"""
        Normalize a valuation result specific to classwise Shapley.

        Each value $v_i$ associated with the sample $(x_i, y_i)$ is normalized by
        multiplying it with $a_S(D_{y_i})$ and dividing by $\sum_{j \in D_{y_i}} v_j$.
        For more details, see (Schoch et al., 2022) <sup><a
        href="#schoch_csshapley_2022">1</a> </sup>.

        Returns:
            Normalized ValuationResult object.
        """
        if self.result is None:
            raise ValueError("You must call fit before calling _normalize()")

        if self.utility.training_data is None:
            raise ValueError("You should call fit before calling _normalize()")

        logger.info("Normalizing valuation result.")
        x, y = self.utility.training_data.data()
        unique_labels = get_unique_labels(y)
        self.utility.model.fit(x, y)

        for idx_label, label in enumerate(unique_labels):
            active_elements = y == label
            indices_label_set = np.where(active_elements)[0]
            indices_label_set = self.utility.training_data.indices[indices_label_set]

            self.scorer.label = label
            in_class_acc, _ = self.scorer.compute_in_and_out_of_class_scores(
                self.utility.model
            )

            sigma = np.sum(self.result.values[indices_label_set])
            if sigma != 0:
                self.result.scale(in_class_acc / sigma, data_indices=indices_label_set)

        return self.result
