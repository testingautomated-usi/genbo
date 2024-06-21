from typing import List

from test.config import FITNESS_NAMES, LATERAL_POSITION_FITNESS_NAME, CTE_FITNESS_NAME
from test.cte_fitness import CTEFitness
from test.fitness import Fitness
from test.lateral_position_fitness import LateralPositionFitness


def make_fitness(
    fitness_name: str,
    lateral_positions: List[float],
    ctes: List[float],
    max_abs_value: float = None,
    mock: bool = False,
) -> Fitness:
    assert fitness_name in FITNESS_NAMES, "Fitness name should be in {}".format(
        FITNESS_NAMES
    )
    if fitness_name == LATERAL_POSITION_FITNESS_NAME:
        return LateralPositionFitness(lateral_positions=lateral_positions, mock=mock)
    if fitness_name == CTE_FITNESS_NAME:
        return CTEFitness(ctes=ctes, max_abs_value=max_abs_value, mock=mock)

    raise RuntimeError("Unknown fitness name: {}".format(fitness_name))
