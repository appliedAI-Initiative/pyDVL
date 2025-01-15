"""
This module implements a trivial random valuation method.

"""

from pydvl.utils import Seed
from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.result import ValuationResult


class RandomValuation(Valuation):
    """
    A trivial valuation method that assigns random values to each data point.

    Values are in the range [0, 1), as generated by
    [ValuationResult.from_random][pydvl.valuation.result.ValuationResult.from_random].
    """

    def __init__(self, random_state: Seed):
        super().__init__()
        self.random_state = random_state

    def fit(self, train: Dataset):
        self.result = ValuationResult.from_random(
            size=len(train), seed=self.random_state
        )
