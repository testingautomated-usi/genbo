import numpy as np

from test.config import FITNESS_PAIR_NAME
from test.fitness import Fitness


class FitnessPairFitness(Fitness):

    def __init__(
        self,
        fitness_m1: Fitness,
        fitness_m2: Fitness,
        maximize: bool = True,
        mock: bool = False,
    ):
        super(FitnessPairFitness, self).__init__()
        self.fitness_m1 = fitness_m1
        self.fitness_m2 = fitness_m2
        self.mock = mock
        self.maximize = maximize
        self.mock_value = self.get_random_value()
        self.name = FITNESS_PAIR_NAME

    def clone(self) -> "Fitness":
        return FitnessPairFitness(
            fitness_m1=self.fitness_m1.clone(),
            fitness_m2=self.fitness_m2.clone(),
            mock=self.mock,
            maximize=self.maximize,
        )

    def get_value(self) -> float:
        if not self.mock:
            if self.maximize:
                return max(self.fitness_m1.get_value(), self.fitness_m2.get_value())
            return min(self.fitness_m1.get_value(), self.fitness_m2.get_value())
        return self.mock_value

    def get_min_value(self) -> float:
        return min(self.fitness_m1.get_min_value(), self.fitness_m2.get_min_value())

    def get_max_value(self) -> float:
        return max(self.fitness_m1.get_max_value(), self.fitness_m2.get_max_value())

    def get_random_value(self) -> float:
        return round(np.random.uniform(low=-3, high=3), 4)
