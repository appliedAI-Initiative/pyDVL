"""
!!! Warning "Deprecation notice"
    This module is deprecated since v0.10.0. For use with the methods in
    [pydvl.valuation][] please use [pydvl.valuation.dataset][] instead.

This module contains convenience classes to handle data and groups thereof.

Shapley and Least Core value computations require evaluation of a scoring function
(the *utility*). This is typically the performance of the model on a test set
(as an approximation to its true expected performance). It is therefore convenient
to keep both the training data and the test data together to be passed around to
methods in [shapley][pydvl.value.shapley] and [least_core][pydvl.value.least_core].
This is done with [Dataset][pydvl.utils.dataset.Dataset].

This abstraction layer also seamlessly grouping data points together if one is
interested in computing their value as a group, see
[GroupedDataset][pydvl.utils.dataset.GroupedDataset].

Objects of both types are used to construct a [Utility][pydvl.utils.utility.Utility]
object.

"""

import logging
from collections import OrderedDict
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch, check_X_y

__all__ = ["Dataset", "GroupedDataset"]

logger = logging.getLogger(__name__)


class Dataset:
    """A convenience class to handle datasets.

    It holds a dataset, split into training and test data, together with several
    labels on feature names, data point names and a description.
    """

    def __init__(
        self,
        x_train: Union[NDArray, pd.DataFrame],
        y_train: Union[NDArray, pd.DataFrame],
        x_test: Union[NDArray, pd.DataFrame],
        y_test: Union[NDArray, pd.DataFrame],
        feature_names: Optional[Sequence[str]] = None,
        target_names: Optional[Sequence[str]] = None,
        data_names: Optional[Sequence[str]] = None,
        description: Optional[str] = None,
        # FIXME: use same parameter name as in check_X_y()
        is_multi_output: bool = False,
    ):
        """Constructs a Dataset from data and labels.

        Args:
            x_train: training data
            y_train: labels for training data
            x_test: test data
            y_test: labels for test data
            feature_names: name of the features of input data
            target_names: names of the features of target data
            data_names: names assigned to data points.
                For example, if the dataset is a time series, each entry can be a
                timestamp which can be referenced directly instead of using a row
                number.
            description: A textual description of the dataset.
            is_multi_output: set to `False` if labels are scalars, or to
                `True` if they are vectors of dimension > 1.
        """
        self.x_train, self.y_train = check_X_y(
            x_train, y_train, multi_output=is_multi_output
        )
        self.x_test, self.y_test = check_X_y(
            x_test, y_test, multi_output=is_multi_output
        )

        if x_train.shape[-1] != x_test.shape[-1]:
            raise ValueError(
                f"Mismatching number of features: "
                f"{x_train.shape[-1]} and {x_test.shape[-1]}"
            )
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"Mismatching number of samples: "
                f"{x_train.shape[-1]} and {x_test.shape[-1]}"
            )
        if x_test.shape[0] != y_test.shape[0]:
            raise ValueError(
                f"Mismatching number of samples: "
                f"{x_test.shape[-1]} and {y_test.shape[-1]}"
            )

        def make_names(s: str, a: np.ndarray) -> List[str]:
            n = a.shape[1] if len(a.shape) > 1 else 1
            return [f"{s}{i:0{1 + int(np.log10(n))}d}" for i in range(1, n + 1)]

        self.feature_names = feature_names
        self.target_names = target_names

        if self.feature_names is None:
            if isinstance(x_train, pd.DataFrame):
                self.feature_names = x_train.columns.tolist()
            else:
                self.feature_names = make_names("x", x_train)

        if self.target_names is None:
            if isinstance(y_train, pd.DataFrame):
                self.target_names = y_train.columns.tolist()
            else:
                self.target_names = make_names("y", y_train)

        if len(self.x_train.shape) > 1:
            if (
                len(self.feature_names) != self.x_train.shape[-1]
                or len(self.feature_names) != self.x_test.shape[-1]
            ):
                raise ValueError("Mismatching number of features and names")
        if len(self.y_train.shape) > 1:
            if (
                len(self.target_names) != self.y_train.shape[-1]
                or len(self.target_names) != self.y_test.shape[-1]
            ):
                raise ValueError("Mismatching number of targets and names")

        self.description = description or "No description"
        self._indices: NDArray[np.int_] = np.arange(len(self.x_train), dtype=np.int_)
        self._data_names: NDArray[np.object_] = (
            np.array(data_names, dtype=object)
            if data_names is not None
            else self._indices.astype(object)
        )

    def __getitem__(self, idx: Union[int, slice, Iterable]) -> Tuple:
        return self.x_train[idx], self.y_train[idx]

    def feature(self, name: str) -> Tuple[slice, int]:
        try:
            return np.index_exp[:, self.feature_names.index(name)]  # type: ignore
        except ValueError:
            raise ValueError(f"Feature {name} is not in {self.feature_names}")

    def get_training_data(
        self, indices: Optional[Iterable[int]] = None
    ) -> Tuple[NDArray, NDArray]:
        """Given a set of indices, returns the training data that refer to those
        indices.

        This is used mainly by [Utility][pydvl.utils.utility.Utility] to retrieve
        subsets of the data from indices. It is typically **not needed in
        algorithms**.

        Args:
            indices: Optional indices that will be used to select points from
                the training data. If `None`, the entire training data will be
                returned.

        Returns:
            If `indices` is not `None`, the selected x and y arrays from the
                training data. Otherwise, the entire dataset.
        """
        if indices is None:
            return self.x_train, self.y_train
        x = self.x_train[indices]
        y = self.y_train[indices]
        return x, y

    def get_test_data(
        self, indices: Optional[Iterable[int]] = None
    ) -> Tuple[NDArray, NDArray]:
        """Returns the entire test set regardless of the passed indices.

        The passed indices will not be used because for data valuation
        we generally want to score the trained model on the entire test data.

        Additionally, the way this method is used in the
        [Utility][pydvl.utils.utility.Utility] class, the passed indices will
        be those of the training data and would not work on the test data.

        There may be cases where it is desired to use parts of the test data.
        In those cases, it is recommended to inherit from
        [Dataset][pydvl.utils.dataset.Dataset] and override
        [get_test_data()][pydvl.utils.dataset.Dataset.get_test_data].

        For example, the following snippet shows how one could go about
        mapping the training data indices into test data indices
        inside [get_test_data()][pydvl.utils.dataset.Dataset.get_test_data]:

        ??? Example
            ```pycon
            >>> from pydvl.utils import Dataset
            >>> import numpy as np
            >>> class DatasetWithTestDataIndices(Dataset):
            ...    def get_test_data(self, indices=None):
            ...        if indices is None:
            ...            return self.x_test, self.y_test
            ...        fraction = len(list(indices)) / len(self)
            ...        mapped_indices = len(self.x_test) / len(self) * np.asarray(indices)
            ...        mapped_indices = np.unique(mapped_indices.astype(int))
            ...        return self.x_test[mapped_indices], self.y_test[mapped_indices]
            ...
            >>> X = np.random.rand(100, 10)
            >>> y = np.random.randint(0, 2, 100)
            >>> dataset = DatasetWithTestDataIndices.from_arrays(X, y)
            >>> indices = np.random.choice(dataset.indices, 30, replace=False)
            >>> _ = dataset.get_training_data(indices)
            >>> _ = dataset.get_test_data(indices)
            ```

        Args:
            indices: Optional indices into the test data. This argument is
                unused left for compatibility with
                [get_training_data()][pydvl.utils.dataset.Dataset.get_training_data].

        Returns:
            The entire test data.
        """
        return self.x_test, self.y_test

    def target(self, name: str) -> Tuple[slice, int]:
        try:
            return np.index_exp[:, self.target_names.index(name)]  # type: ignore
        except ValueError:
            raise ValueError(f"Target {name} is not in {self.target_names}")

    @property
    def indices(self) -> NDArray[np.int_]:
        """Index of positions in data.x_train.

        Contiguous integers from 0 to len(Dataset).
        """
        return self._indices

    @property
    def data_names(self) -> NDArray[np.object_]:
        """Names of each individual datapoint.

        Used for reporting Shapley values.
        """
        return self._data_names

    @property
    def dim(self) -> int:
        """Returns the number of dimensions of a sample."""
        return int(self.x_train.shape[1]) if len(self.x_train.shape) > 1 else 1

    def __str__(self):
        return self.description

    def __len__(self):
        return len(self.x_train)

    @classmethod
    def from_sklearn(
        cls,
        data: Bunch,
        train_size: float = 0.8,
        random_state: Optional[int] = None,
        stratify_by_target: bool = False,
        **kwargs: Any,
    ) -> "Dataset":
        """Constructs a [Dataset][pydvl.utils.Dataset] object from a
        [sklearn.utils.Bunch][], as returned by the `load_*`
        functions in [scikit-learn toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html).

        ??? Example
            ```pycon
            >>> from pydvl.utils import Dataset
            >>> from sklearn.datasets import load_boston
            >>> dataset = Dataset.from_sklearn(load_boston())
            ```

        Args:
            data: scikit-learn Bunch object. The following attributes are supported:

                - `data`: covariates.
                - `target`: target variables (labels).
                - `feature_names` (**optional**): the feature names.
                - `target_names` (**optional**): the target names.
                - `DESCR` (**optional**): a description.
            train_size: size of the training dataset. Used in `train_test_split`
            random_state: seed for train / test split
            stratify_by_target: If `True`, data is split in a stratified
                fashion, using the target variable as labels. Read more in
                [scikit-learn's user guide](https://scikit-learn.org/stable/modules/cross_validation.html#stratification).
            kwargs: Additional keyword arguments to pass to the
                [Dataset][pydvl.utils.Dataset] constructor. Use this to pass e.g. `is_multi_output`.

        Returns:
            Object with the sklearn dataset

        !!! tip "Changed in version 0.6.0"
            Added kwargs to pass to the [Dataset][pydvl.utils.Dataset] constructor.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            data.data,
            data.target,
            train_size=train_size,
            random_state=random_state,
            stratify=data.target if stratify_by_target else None,
        )
        return cls(
            x_train,
            y_train,
            x_test,
            y_test,
            feature_names=data.get("feature_names"),
            target_names=data.get("target_names"),
            description=data.get("DESCR"),
            **kwargs,
        )

    @classmethod
    def from_arrays(
        cls,
        X: NDArray,
        y: NDArray,
        train_size: float = 0.8,
        random_state: Optional[int] = None,
        stratify_by_target: bool = False,
        **kwargs: Any,
    ) -> "Dataset":
        """Constructs a [Dataset][pydvl.utils.Dataset] object from X and y numpy arrays  as
        returned by the `make_*` functions in [sklearn generated datasets](https://scikit-learn.org/stable/datasets/sample_generators.html).

        ??? Example
            ```pycon
            >>> from pydvl.utils import Dataset
            >>> from sklearn.datasets import make_regression
            >>> X, y = make_regression()
            >>> dataset = Dataset.from_arrays(X, y)
            ```

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)
            train_size: size of the training dataset. Used in `train_test_split`
            random_state: seed for train / test split
            stratify_by_target: If `True`, data is split in a stratified fashion,
                using the y variable as labels. Read more in [sklearn's user
                guide](https://scikit-learn.org/stable/modules/cross_validation.html#stratification).
            kwargs: Additional keyword arguments to pass to the
                [Dataset][pydvl.utils.Dataset] constructor. Use this to pass e.g. `feature_names`
                or `target_names`.

        Returns:
            Object with the passed X and y arrays split across training and test sets.

        !!! tip "New in version 0.4.0"

        !!! tip "Changed in version 0.6.0"
            Added kwargs to pass to the [Dataset][pydvl.utils.Dataset] constructor.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_size,
            random_state=random_state,
            stratify=y if stratify_by_target else None,
        )
        return cls(x_train, y_train, x_test, y_test, **kwargs)


class GroupedDataset(Dataset):
    def __init__(
        self,
        x_train: NDArray,
        y_train: NDArray,
        x_test: NDArray,
        y_test: NDArray,
        data_groups: Sequence,
        feature_names: Optional[Sequence[str]] = None,
        target_names: Optional[Sequence[str]] = None,
        group_names: Optional[Sequence[str]] = None,
        description: Optional[str] = None,
        **kwargs: Any,
    ):
        """Class for grouping datasets.

        Used for calculating Shapley values of subsets of the data considered
        as logical units. For instance, one can group by value of a categorical
        feature, by bin into which a continuous feature falls, or by label.

        Args:
            x_train: training data
            y_train: labels of training data
            x_test: test data
            y_test: labels of test data
            data_groups: Iterable of the same length as `x_train` containing
                a group label for each training data point. The label can be of any
                type, e.g. `str` or `int`. Data points with the same label will
                then be grouped by this object and considered as one for effects of
                valuation.
            feature_names: names of the covariates' features.
            target_names: names of the labels or targets y
            group_names: names of the groups. If not provided, the labels
                from `data_groups` will be used.
            description: A textual description of the dataset
            kwargs: Additional keyword arguments to pass to the
                [Dataset][pydvl.utils.Dataset] constructor.

        !!! tip "Changed in version 0.6.0"
        Added `group_names` and forwarding of `kwargs`
        """
        super().__init__(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            feature_names=feature_names,
            target_names=target_names,
            description=description,
            **kwargs,
        )

        if len(data_groups) != len(x_train):
            raise ValueError(
                f"data_groups and x_train must have the same length."
                f"Instead got {len(data_groups)=} and {len(x_train)=}"
            )

        self.groups: OrderedDict[Any, List[int]] = OrderedDict(
            {k: [] for k in set(data_groups)}
        )
        for idx, group in enumerate(data_groups):
            self.groups[group].append(idx)
        self.group_items = list(self.groups.items())
        self._indices = np.arange(len(self.groups.keys()))
        self._data_names = (
            np.array(group_names, dtype=object)
            if group_names is not None
            else np.array(list(self.groups.keys()), dtype=object)
        )

    def __len__(self):
        return len(self.groups)

    @property
    def indices(self):
        """Indices of the groups."""
        return self._indices

    # FIXME this is a misnomer, should be `names` in `Dataset` so that here it
    #  makes sense
    @property
    def data_names(self):
        """Names of the groups."""
        return self._data_names

    def get_training_data(
        self, indices: Optional[Iterable[int]] = None
    ) -> Tuple[NDArray, NDArray]:
        """Returns the data and labels of all samples in the given groups.

        Args:
            indices: group indices whose elements to return. If `None`,
                all data from all groups are returned.

        Returns:
            Tuple of training data x and labels y.
        """
        if indices is None:
            indices = self.indices
        data_indices = [
            idx for group_id in indices for idx in self.group_items[group_id][1]
        ]
        return super().get_training_data(data_indices)

    @classmethod
    def from_sklearn(
        cls,
        data: Bunch,
        train_size: float = 0.8,
        random_state: Optional[int] = None,
        stratify_by_target: bool = False,
        data_groups: Optional[Sequence] = None,
        **kwargs: Any,
    ) -> "GroupedDataset":
        """Constructs a [GroupedDataset][pydvl.utils.GroupedDataset] object from a
        [sklearn.utils.Bunch][sklearn.utils.Bunch] as returned by the `load_*` functions in
        [scikit-learn toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html) and groups
        it.

        ??? Example
            ```pycon
            >>> from sklearn.datasets import load_iris
            >>> from pydvl.utils import GroupedDataset
            >>> iris = load_iris()
            >>> data_groups = iris.data[:, 0] // 0.5
            >>> dataset = GroupedDataset.from_sklearn(iris, data_groups=data_groups)
            ```

        Args:
            data: scikit-learn Bunch object. The following attributes are supported:

                - `data`: covariates.
                - `target`: target variables (labels).
                - `feature_names` (**optional**): the feature names.
                - `target_names` (**optional**): the target names.
                - `DESCR` (**optional**): a description.
            train_size: size of the training dataset. Used in `train_test_split`.
            random_state: seed for train / test split.
            stratify_by_target: If `True`, data is split in a stratified
                fashion, using the target variable as labels. Read more in
                [sklearn's user guide](https://scikit-learn.org/stable/modules/cross_validation.html#stratification).
            data_groups: an array holding the group index or name for each
                data point. The length of this array must be equal to the number of
                data points in the dataset.
            kwargs: Additional keyword arguments to pass to the
                [Dataset][pydvl.utils.Dataset] constructor.

        Returns:
            Dataset with the selected sklearn data
        """
        if data_groups is None:
            raise ValueError(
                "data_groups must be provided when constructing a GroupedDataset"
            )

        x_train, x_test, y_train, y_test, data_groups_train, _ = train_test_split(
            data.data,
            data.target,
            data_groups,
            train_size=train_size,
            random_state=random_state,
            stratify=data.target if stratify_by_target else None,
        )

        dataset = Dataset(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, **kwargs
        )
        return cls.from_dataset(dataset, data_groups_train)  # type: ignore

    @classmethod
    def from_arrays(
        cls,
        X: NDArray,
        y: NDArray,
        train_size: float = 0.8,
        random_state: Optional[int] = None,
        stratify_by_target: bool = False,
        data_groups: Optional[Sequence] = None,
        **kwargs: Any,
    ) -> "Dataset":
        """Constructs a [GroupedDataset][pydvl.utils.GroupedDataset] object from X and y numpy arrays
        as returned by the `make_*` functions in
        [scikit-learn generated datasets](https://scikit-learn.org/stable/datasets/sample_generators.html).

        ??? Example
            ```pycon
            >>> from sklearn.datasets import make_classification
            >>> from pydvl.utils import GroupedDataset
            >>> X, y = make_classification(
            ...     n_samples=100,
            ...     n_features=4,
            ...     n_informative=2,
            ...     n_redundant=0,
            ...     random_state=0,
            ...     shuffle=False
            ... )
            >>> data_groups = X[:, 0] // 0.5
            >>> dataset = GroupedDataset.from_arrays(X, y, data_groups=data_groups)
            ```

        Args:
            X: array of shape (n_samples, n_features)
            y: array of shape (n_samples,)
            train_size: size of the training dataset. Used in `train_test_split`.
            random_state: seed for train / test split.
            stratify_by_target: If `True`, data is split in a stratified
                fashion, using the y variable as labels. Read more in
                [sklearn's user guide](https://scikit-learn.org/stable/modules/cross_validation.html#stratification).
            data_groups: an array holding the group index or name for each data
                point. The length of this array must be equal to the number of
                data points in the dataset.
            kwargs: Additional keyword arguments that will be passed to the
                [Dataset][pydvl.utils.Dataset] constructor.

        Returns:
            Dataset with the passed X and y arrays split across training and
                test sets.

        !!! tip "New in version 0.4.0"

        !!! tip "Changed in version 0.6.0"
            Added kwargs to pass to the [Dataset][pydvl.utils.Dataset] constructor.
        """
        if data_groups is None:
            raise ValueError(
                "data_groups must be provided when constructing a GroupedDataset"
            )
        x_train, x_test, y_train, y_test, data_groups_train, _ = train_test_split(
            X,
            y,
            data_groups,
            train_size=train_size,
            random_state=random_state,
            stratify=y if stratify_by_target else None,
        )
        dataset = Dataset(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, **kwargs
        )
        return cls.from_dataset(dataset, data_groups_train)

    @classmethod
    def from_dataset(
        cls, dataset: Dataset, data_groups: Sequence[Any]
    ) -> "GroupedDataset":
        """Creates a [GroupedDataset][pydvl.utils.GroupedDataset] object from the data a
        [Dataset][pydvl.utils.Dataset] object and a mapping of data groups.

        ??? Example
            ```pycon
            >>> import numpy as np
            >>> from pydvl.utils import Dataset, GroupedDataset
            >>> dataset = Dataset.from_arrays(
            ...     X=np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]]),
            ...     y=np.asarray([0, 1, 0, 1]),
            ... )
            >>> dataset = GroupedDataset.from_dataset(dataset, data_groups=[0, 0, 1, 1])
            ```

        Args:
            dataset: The original data.
            data_groups: An array holding the group index or name for each data
                point. The length of this array must be equal to the number of
                data points in the dataset.

        Returns:
            A [GroupedDataset][pydvl.utils.GroupedDataset] with the initial
                [Dataset][pydvl.utils.Dataset] grouped by data_groups.
        """
        return cls(
            x_train=dataset.x_train,
            y_train=dataset.y_train,
            x_test=dataset.x_test,
            y_test=dataset.y_test,
            data_groups=data_groups,
            feature_names=dataset.feature_names,
            target_names=dataset.target_names,
            description=dataset.description,
        )
