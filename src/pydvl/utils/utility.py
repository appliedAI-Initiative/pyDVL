"""
!!! Warning "Deprecation notice"
    This module is deprecated since v0.10.0. For use with the methods in
    [pydvl.valuation][] please use any of the classes in
    [pydvl.valuation.utility][] instead.

This module contains classes to manage and learn utility functions for the
computation of values. Please see the documentation on
[Computing Data Values][computing-data-values] for more information.

[Utility][pydvl.utils.utility.Utility] holds information about model,
data and scoring function (the latter being what one usually understands
under *utility* in the general definition of Shapley value).
It is automatically cached across machines when the
[cache is configured][getting-started-cache] and it is enabled upon construction.

[DataUtilityLearning][pydvl.utils.utility.DataUtilityLearning] adds support
for learning the scoring function to avoid repeated re-training
of the model to compute the score.

This module also contains derived `Utility` classes for toy games that are used
for testing and for demonstration purposes.

## References

[^1]: <a name="wang_improving_2022"></a>Wang, T., Yang, Y. and Jia, R., 2021.
    [Improving cooperative game theory-based data valuation via data utility
    learning](https://arxiv.org/abs/2107.06336). arXiv preprint arXiv:2107.06336.

"""

import logging
import warnings
from typing import Dict, FrozenSet, Iterable, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.metrics import check_scoring

from pydvl.utils import Dataset
from pydvl.utils.caching import CacheBackend, CachedFuncConfig, CacheStats
from pydvl.utils.score import Scorer
from pydvl.utils.types import SupervisedModel

__all__ = ["Utility", "DataUtilityLearning"]

logger = logging.getLogger(__name__)


class Utility:
    """Convenience wrapper with configurable memoization of the scoring
    function.

    An instance of `Utility` holds the triple of model, dataset and scoring
    function which determines the value of data points. This is used for the
    computation of [all game-theoretic values][game-theoretical-methods] like
    [Shapley values][pydvl.value.shapley] and [the Least
    Core][pydvl.value.least_core].

    The Utility expect the model to fulfill the
    [SupervisedModel][pydvl.utils.types.SupervisedModel] interface i.e.
    to have `fit()`, `predict()`, and `score()` methods.

    When calling the utility, the model will be
    [cloned](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html)
    if it is a Scikit-Learn model, otherwise a copy is created using
    [copy.deepcopy][]

    Since evaluating the scoring function requires retraining the model and that
    can be time-consuming, this class wraps it and caches the results of each
    execution. Caching is available both locally and across nodes, but must
    always be enabled for your project first, see [the documentation][getting-started-cache]
    and the [module documentation][pydvl.utils.caching].

    Attributes:
        model: The supervised model.
        data: An object containing the split data.
        scorer: A scoring function. If None, the `score()` method of the model
            will be used. See [score][pydvl.utils.score] for ways to create
            and compose scorers, in particular how to set default values and
            ranges.

    Args:
        model: Any supervised model. Typical choices can be found in the
            [sci-kit learn documentation](https://scikit-learn.org/stable/supervised_learning.html).
        data: [Dataset][pydvl.utils.dataset.Dataset]
            or [GroupedDataset][pydvl.utils.dataset.GroupedDataset] instance.
        scorer: A scoring object. If None, the `score()` method of the model
            will be used. See [score][pydvl.utils.score] for ways to create
            and compose scorers, in particular how to set default values and
            ranges. For convenience, a string can be passed, which will be used
            to construct a [Scorer][pydvl.utils.score.Scorer].
        default_score: As a convenience when no `scorer` object is passed
            (where a default value can be provided), this argument also allows
            to set the default score for models that have not been fit, e.g.
            when too little data is passed, or errors arise.
        score_range: As with `default_score`, this is a convenience argument for
            when no `scorer` argument is provided, to set the numerical range
            of the score function. Some Monte Carlo methods can use this to
            estimate the number of samples required for a certain quality of
            approximation.
        catch_errors: set to `True` to catch the errors when `fit()` fails. This
            could happen in several steps of the pipeline, e.g. when too little
            training data is passed, which happens often during Shapley value
            calculations. When this happens, the `default_score` is returned as
            a score and computation continues.
        show_warnings: Set to `False` to suppress warnings thrown by `fit()`.
        cache_backend: Optional instance of [CacheBackend][pydvl.utils.caching.base.CacheBackend]
            used to wrap the _utility method of the Utility instance.
            By default, this is set to None and that means that the utility evaluations
            will not be cached.
        cached_func_options: Optional configuration object for cached utility evaluation.
        clone_before_fit: If `True`, the model will be cloned before calling
            `fit()`.

    ??? Example
        ``` pycon
        >>> from pydvl.utils import Utility, DataUtilityLearning, Dataset
        >>> from sklearn.linear_model import LinearRegression, LogisticRegression
        >>> from sklearn.datasets import load_iris
        >>> dataset = Dataset.from_sklearn(load_iris(), random_state=16)
        >>> u = Utility(LogisticRegression(random_state=16), dataset)
        >>> u(dataset.indices)
        0.9
        ```

        With caching enabled:

        ```pycon
        >>> from pydvl.utils import Utility, DataUtilityLearning, Dataset
        >>> from pydvl.utils.caching.memory import InMemoryCacheBackend
        >>> from sklearn.linear_model import LinearRegression, LogisticRegression
        >>> from sklearn.datasets import load_iris
        >>> dataset = Dataset.from_sklearn(load_iris(), random_state=16)
        >>> cache_backend = InMemoryCacheBackend()
        >>> u = Utility(LogisticRegression(random_state=16), dataset, cache_backend=cache_backend)
        >>> u(dataset.indices)
        0.9
        ```

    """

    model: SupervisedModel
    data: Dataset
    scorer: Scorer

    def __init__(
        self,
        model: SupervisedModel,
        data: Dataset,
        scorer: Optional[Union[str, Scorer]] = None,
        *,
        default_score: float = 0.0,
        score_range: Tuple[float, float] = (-np.inf, np.inf),
        catch_errors: bool = True,
        show_warnings: bool = False,
        cache_backend: Optional[CacheBackend] = None,
        cached_func_options: Optional[CachedFuncConfig] = None,
        clone_before_fit: bool = True,
    ):
        self.model = self._clone_model(model)
        self.data = data
        if isinstance(scorer, str):
            scorer = Scorer(scorer, default=default_score, range=score_range)
        self.scorer = check_scoring(self.model, scorer)
        self.default_score = scorer.default if scorer is not None else default_score
        # TODO: auto-fill from known scorers ?
        self.score_range = scorer.range if scorer is not None else np.array(score_range)
        self.clone_before_fit = clone_before_fit
        self.catch_errors = catch_errors
        self.show_warnings = show_warnings
        self.cache = cache_backend
        if cached_func_options is None:
            cached_func_options = CachedFuncConfig()
        # TODO: Find a better way to do this.
        if cached_func_options.hash_prefix is None:
            # FIX: This does not handle reusing the same across runs.
            cached_func_options.hash_prefix = str(hash((model, data, scorer)))
        self.cached_func_options = cached_func_options
        self._initialize_utility_wrapper()

    def _initialize_utility_wrapper(self):
        if self.cache is not None:
            self._utility_wrapper = self.cache.wrap(
                self._utility, config=self.cached_func_options
            )
        else:
            self._utility_wrapper = self._utility

    def __call__(self, indices: Iterable[int]) -> float:
        """
        Args:
            indices: a subset of valid indices for the
                `x_train` attribute of [Dataset][pydvl.utils.dataset.Dataset].
        """
        utility: float = self._utility_wrapper(frozenset(indices))
        return utility

    def _utility(self, indices: FrozenSet) -> float:
        """Clones the model, fits it on a subset of the training data
        and scores it on the test data.

        If an instance of [CacheBackend][pydvl.utils.caching.base.CacheBackend]
        is passed during construction, results are
        memoized to avoid duplicate computation. This is useful in particular
        when computing utilities of permutations of indices or when randomly
        sampling from the powerset of indices.

        Args:
            indices: a subset of valid indices for the
                `x_train` attribute of [Dataset][pydvl.utils.dataset.Dataset].
                The type must be hashable for the caching to work,
                e.g. wrap the argument with [frozenset][]
                (rather than `tuple` since order should not matter)

        Returns:
            0 if no indices are passed, `default_score` if we fail
                to fit the model or the scorer returns [numpy.nan][]. Otherwise, the score
                of the model on the test data.
        """
        if not indices:
            return 0.0

        x_train, y_train = self.data.get_training_data(list(indices))
        x_test, y_test = self.data.get_test_data(list(indices))

        with warnings.catch_warnings():
            if not self.show_warnings:
                warnings.simplefilter("ignore")
            try:
                if self.clone_before_fit:
                    model = self._clone_model(self.model)
                else:
                    model = self.model
                model.fit(x_train, y_train)
                score = float(self.scorer(model, x_test, y_test))
                # Some scorers raise exceptions if they return NaNs, some might not
                if np.isnan(score):
                    warnings.warn("Scorer returned NaN", RuntimeWarning)
                    return self.default_score
                return score
            except Exception as e:
                if self.catch_errors:
                    warnings.warn(str(e), RuntimeWarning)
                    return self.default_score
                raise

    @staticmethod
    def _clone_model(model: SupervisedModel) -> SupervisedModel:
        """Clones the passed model to avoid the possibility
        of reusing a fitted estimator

        Args:
            model: Any supervised model. Typical choices can be found
                on [this page](https://scikit-learn.org/stable/supervised_learning.html)
        """
        try:
            model = clone(model)
        except TypeError:
            # This happens if the passed model is not an sklearn model
            # In this case, we just make a deepcopy of the model.
            model = clone(model, safe=False)
        model = cast(SupervisedModel, model)
        return model

    @property
    def cache_stats(self) -> Optional[CacheStats]:
        """Cache statistics are gathered when cache is enabled.
        See [CacheStats][pydvl.utils.caching.base.CacheStats] for all fields returned.
        """
        cache_stats: Optional[CacheStats] = None
        if self.cache is not None:
            cache_stats = self._utility_wrapper.stats
        return cache_stats

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle _utility_wrapper
        state.pop("_utility_wrapper", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add _utility_wrapper back since it doesn't exist in the pickle
        self._initialize_utility_wrapper()


class DataUtilityLearning:
    """Implementation of Data Utility Learning
    (Wang et al., 2022)<sup><a href="#wang_improving_2022">1</a></sup>.

    This object wraps a [Utility][pydvl.utils.utility.Utility] and delegates
    calls to it, up until a given budget (number of iterations). Every tuple
    of input and output (a so-called *utility sample*) is stored. Once the
    budget is exhausted, `DataUtilityLearning` fits the given model to the
    utility samples. Subsequent calls will use the learned model to predict the
    utility instead of delegating.

    Args:
        u: The [Utility][pydvl.utils.utility.Utility] to learn.
        training_budget: Number of utility samples to collect before fitting
            the given model.
        model: A supervised regression model

    ??? Example
        ``` pycon
        >>> from pydvl.utils import Utility, DataUtilityLearning, Dataset
        >>> from sklearn.linear_model import LinearRegression, LogisticRegression
        >>> from sklearn.datasets import load_iris
        >>> dataset = Dataset.from_sklearn(load_iris())
        >>> u = Utility(LogisticRegression(), dataset)
        >>> wrapped_u = DataUtilityLearning(u, 3, LinearRegression())
        ... # First 3 calls will be computed normally
        >>> for i in range(3):
        ...     _ = wrapped_u((i,))
        >>> wrapped_u((1, 2, 3)) # Subsequent calls will be computed using the fit model for DUL
        0.0
        ```

    """

    def __init__(
        self, u: Utility, training_budget: int, model: SupervisedModel
    ) -> None:
        self.utility = u
        self.training_budget = training_budget
        self.model = model
        self._current_iteration = 0
        self._is_model_fit = False
        self._utility_samples: Dict[FrozenSet, Tuple[NDArray[np.bool_], float]] = {}

    def _convert_indices_to_boolean_vector(self, x: Iterable[int]) -> NDArray[np.bool_]:
        boolean_vector: NDArray[np.bool_] = np.zeros(
            (1, len(self.utility.data)), dtype=bool
        )
        if x is not None:
            boolean_vector[:, tuple(x)] = True
        return boolean_vector

    def __call__(self, indices: Iterable[int]) -> float:
        indices_boolean_vector = self._convert_indices_to_boolean_vector(indices)
        frozen_indices = frozenset(indices)
        if len(self._utility_samples) < self.training_budget:
            utility = self.utility(frozen_indices)
            self._utility_samples[frozen_indices] = (indices_boolean_vector, utility)
        else:
            if not self._is_model_fit:
                X, y = zip(*self._utility_samples.values())
                X = np.vstack(X)
                y = np.asarray(y)
                self.model.fit(X, y)
                self._is_model_fit = True
            if frozen_indices in self._utility_samples:
                utility = self._utility_samples[frozen_indices][1]
            else:
                utility = self.model.predict(indices_boolean_vector).item()
        return utility

    @property
    def data(self) -> Dataset:
        """Returns the wrapped utility's [Dataset][pydvl.utils.dataset.Dataset]."""
        return self.utility.data
