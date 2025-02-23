"""
!!! tip "New in version 0.9.0"

## References

[^1]: <a name="just_lava_2023"></a>Just et al.
[LAVA: Data Valuation without Pre-Specified Learning Algorithms](https://arxiv.org/abs/2305.00054).
In: Published at ICRL 2023
"""

import itertools
from typing import Callable, Literal, Tuple

import numpy as np
import ot
from numpy.typing import NDArray
from ot.bregman import empirical_sinkhorn2
from ot.gaussian import empirical_bures_wasserstein_distance

from pydvl.utils.dataset import Dataset

__all__ = ["LAVA"]


class LAVA:
    """Computes Data values using LAVA.

    This implements the method described in
    (Just et al., 2023)<sup><a href="just_lava_2023">1</a></sup>.

    Args:
        dataset: The dataset containing training and test samples.
        regularization: Regularization parameter for sinkhorn iterations.
        lambda_: Weight parameter for label distances.
        inner_ot_method: Name of method used to compute the inner (instance-wise)
            OT problem. Must be one of 'gaussian' or 'exact'.
            If set to 'gaussian', the label distributions
            are approximated as Gaussians, and thus their distance is computed as
            the Bures-Wasserstein distance. If set to 'exact', no approximation is
            used, and their distance is computed as an exact Wasserstein problem.

    Examples:
        >>> from pydvl.ot.lava import LAVA
        >>> from pydvl.utils.dataset import Dataset
        >>> from sklearn.datasets import load_iris
        >>> dataset = Dataset.from_sklearn(load_iris())
        >>> lava = LAVA(dataset)
        >>> values = lava.compute_values()
        >>> assert values.shape == (len(dataset.x_train),)
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        regularization: float = 0.1,
        lambda_: float = 1.0,
        inner_ot_method: Literal["exact", "gaussian"] = "exact",
    ) -> None:
        self.dataset = dataset
        self.regularization = regularization
        self.lambda_ = lambda_
        self.inner_ot_method = inner_ot_method

    def compute_values(self) -> NDArray:
        """Compute calibrated gradients using Optimal Transport.

        Returns:
            Array of dimensions `(n_train)` that contains the calibrated gradients.
        """
        dual_solution = self._compute_ot_dual()
        calibrated_gradients = self._compute_calibrated_gradients(dual_solution)
        return calibrated_gradients

    def _compute_calibrated_gradients(
        self, dual_solution: Tuple[NDArray, NDArray]
    ) -> NDArray:
        r"""Compute calibrated gradients using the dual solution of Optimal Transport problem.

        $$
        \frac{\partial OT(\mu_t, \mu_v )}{\partial \mu_t(z_i)}
        = f_i^∗ - \sum\limits_{j\in\{1,\dots,N\} \setminus i} \frac{f_j^∗}{N-1}
        $$

        Args:
            dual_solution: Dual solution of Optimal Transport problem.

        Returns:
            Array of dimensions `(n_train)`
        """
        f1k = np.array(dual_solution[0])
        training_size = len(self.dataset.x_train)
        calibrated_gradients = (1 + 1 / (training_size - 1)) * f1k - f1k.sum() / (
            training_size - 1
        )
        return calibrated_gradients

    def _compute_ot_dual(self) -> Tuple[NDArray, NDArray]:
        """Compute the dual solution of the Optimal Transport problem."""
        ground_cost = self._compute_ground_cost()
        a = ot.unif(len(self.dataset.x_train))
        b = ot.unif(len(self.dataset.x_test))
        gamma, log = ot.sinkhorn(
            a,
            b,
            ground_cost,
            self.regularization,
            log=True,
            verbose=False,
            numItermax=10000,
        )
        u, v = log["u"], log["v"]
        return u, v

    def _compute_ground_cost(self) -> NDArray:
        label_cost = self._compute_label_cost()
        feature_cost = self._compute_feature_cost()
        ground_cost = feature_cost + self.lambda_ * label_cost
        return ground_cost

    def _compute_feature_cost(
        self, metric: Literal["euclidean", "sqeuclidean"] = "sqeuclidean"
    ) -> NDArray:
        """Compute distance between the features of the training and test sets.

        The first has dimensions `(n1, d1)` and the second has dimensions `(n2, d2)`.

        Args:
            p: p-norm

        Returns:
             Array with dimensions `(n1, n2)`
        """
        distance = ot.dist(self.dataset.x_train, self.dataset.x_test, metric=metric)
        return distance

    def _compute_label_cost(self) -> NDArray:
        """Compute distances between classes in the training and test sets.

        The number of classes in the first set is `n_classes1` and the second is `n_classes2`.

        Returns:
            An array with dimensions `(n_classes1, n_classes2)`
        """
        if self.inner_ot_method == "exact":
            ot_method = empirical_sinkhorn2
            reg = 0.1
        else:
            ot_method = empirical_bures_wasserstein_distance
            reg = 0.1

        (
            D_train_train,
            D_train_test,
            D_test_test,
        ) = self._compute_label_distances(ot_method, reg=reg)

        label_distances = np.concatenate(
            [
                np.concatenate([D_train_train, D_train_test], axis=1),
                np.concatenate([D_train_test, D_test_test], axis=1),
            ],
            axis=0,
        )

        """
        M = (
            label_distances.shape[1] * self.dataset.y_train[..., np.newaxis]
            + self.dataset.y_test[np.newaxis, ...]
        )
        label_cost = label_distances.ravel()[M.ravel()].reshape(
            len(self.dataset.y_train), len(self.dataset.y_test)
        )
        """
        indexing_array = (
            D_train_test.shape[1] * self.dataset.y_train[..., np.newaxis]
            + self.dataset.y_test[np.newaxis, ...]
        )
        label_cost = D_train_test.ravel()[indexing_array.ravel()].reshape(
            len(self.dataset.y_train), len(self.dataset.y_test)
        )

        return label_cost

    def _compute_label_distances(
        self, ot_method: Callable, reg: float = 0.1
    ) -> Tuple[NDArray, NDArray, NDArray]:
        c_train = np.sort(np.unique(self.dataset.y_train))
        c_test = np.sort(np.unique(self.dataset.y_test))
        n_train, n_test = len(c_train), len(c_test)

        D_train_test = np.zeros((n_train, n_test))
        for i, j in itertools.product(range(n_train), range(n_test)):
            distance = ot_method(
                self.dataset.x_train[self.dataset.y_train == c_train[i]],
                self.dataset.x_test[self.dataset.y_test == c_test[j]],
                reg=reg,
            )
            D_train_test[i, j] = distance

        D_train_train = np.zeros((n_train, n_train))
        for i, j in itertools.combinations(range(n_train), 2):
            distance = empirical_sinkhorn2(
                self.dataset.x_train[self.dataset.y_train == c_train[i]],
                self.dataset.x_train[self.dataset.y_train == c_train[j]],
                reg=reg,
            )
            D_train_train[i, j] = D_train_train[j, i] = distance

        D_test_test = np.zeros((n_test, n_test))
        for i, j in itertools.combinations(range(n_train), 2):
            distance = empirical_sinkhorn2(
                self.dataset.x_test[self.dataset.y_test == c_test[i]],
                self.dataset.x_test[self.dataset.y_test == c_test[j]],
                reg=reg,
            )
            D_test_test[i, j] = D_test_test[j, i] = distance

        return D_train_train, D_train_test, D_test_test
