import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import make_classification

from pydvl.ot.lava import LAVA
from pydvl.utils.dataset import Dataset


@pytest.fixture
def synthetic_dataset():
    x, y = make_classification(
        n_samples=8,
        n_features=3,
        n_informative=2,
        n_repeated=0,
        n_redundant=1,
        n_classes=2,
        random_state=16,
        flip_y=0,
    )
    dataset = Dataset.from_arrays(
        x, y, train_size=0.5, stratify_by_target=True, random_state=16
    )
    return dataset


@pytest.fixture
def synthetic_dataset_same_train_test(synthetic_dataset):
    synthetic_dataset.x_test = synthetic_dataset.x_train
    synthetic_dataset.y_test = synthetic_dataset.y_train
    return synthetic_dataset


@pytest.fixture
def synthetic_dataset_flipped_labels(synthetic_dataset):
    rng = np.random.default_rng(16)
    flip_mask = rng.uniform(size=len(synthetic_dataset.y_train)) < 0.1
    assert flip_mask.sum() > 0
    synthetic_dataset.y_train[flip_mask] = np.invert(
        synthetic_dataset.y_train[flip_mask].astype(bool)
    ).astype(int)
    return synthetic_dataset, flip_mask


def test_lava_exact_and_gaussian(synthetic_dataset):
    lava = LAVA(synthetic_dataset, inner_ot_method="gaussian")
    gaussian_values = lava.compute_values()
    lava = LAVA(synthetic_dataset, inner_ot_method="exact")
    exact_values = lava.compute_values()
    # We make sure that values are not all the same
    assert_array_equal(gaussian_values, exact_values)


def test_lava_not_all_same_values(synthetic_dataset):
    lava = LAVA(synthetic_dataset, inner_ot_method="gaussian")
    values = lava.compute_values()
    # We make sure that values are not all the same
    assert np.any(~np.isclose(values, values[0]))


def test_lava_same_train_and_test(synthetic_dataset_same_train_test):
    lava = LAVA(synthetic_dataset_same_train_test, inner_ot_method="gaussian")
    values = lava.compute_values()
    # We make sure that all values are zero
    assert_array_equal(values, np.zeros_like(values))


def test_lava_flipped_labels(synthetic_dataset_flipped_labels):
    dataset, flip_mask = synthetic_dataset_flipped_labels
    lava = LAVA(dataset, inner_ot_method="gaussian")
    values = lava.compute_values()
    # We make sure that values are not all the same
    assert np.any(~np.isclose(values, values[0]))
