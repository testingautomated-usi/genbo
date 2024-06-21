import copy
from typing import List

import numpy as np

from global_log import GlobalLog
from test.config import CTE_FITNESS_NAME
from test.fitness import Fitness


class CTEFitness(Fitness):

    def __init__(
        self,
        ctes: List[float],
        max_abs_value: float,
        min_cte: float = None,
        mock: bool = False,
    ):
        super(CTEFitness, self).__init__()
        self.ctes = ctes
        self.min_cte = min_cte
        self.mock = mock
        self.max_abs_value = max_abs_value
        self.name = CTE_FITNESS_NAME
        self.logger = GlobalLog("CTEFitness")
        self.mock_value = self.get_random_value()

    def clone(self) -> "Fitness":
        return CTEFitness(
            ctes=copy.deepcopy(self.ctes),
            min_cte=self.min_cte,
            mock=self.mock,
            max_abs_value=self.max_abs_value,
        )

    def get_value(self) -> float:
        if not self.mock:
            if self.min_cte is not None:
                return self.min_cte
            assert len(self.ctes) > 0, "List of ctes cannot be empty"
            return round(max([abs(cte_value) for cte_value in self.ctes]), 4)
        return self.mock_value

    def get_min_value(self) -> float:
        if self.max_abs_value is not None:
            return -self.max_abs_value
        if len(self.ctes) > 0:
            self.logger.warn(
                "Computing min cte using the available lists of CTEs: {}".format(
                    min(self.ctes)
                )
            )
            return min(self.ctes)
        raise RuntimeError("Not possible to get min value")

    def get_max_value(self) -> float:
        if self.max_abs_value is not None:
            return self.max_abs_value
        if len(self.ctes) > 0:
            self.logger.warn(
                "Computing max cte using the available lists of CTEs: {}".format(
                    min(self.ctes)
                )
            )
        raise RuntimeError("Not possible to get max value")

    def get_random_value(self) -> float:
        if self.max_abs_value is not None and self.max_abs_value == 3.0:
            # this way the failures are rare
            return round(float(np.random.normal(loc=0, scale=1)), 4)
        self.logger.warn("Getting a random CTE value from -5 to 5")
        return round(float(np.random.uniform(low=-5, high=5)), 4)
