import time

import numpy as np

from code_pipeline.evaluator import Evaluator
from config import MOCK_SIM_NAME
from envs.env_utils import get_max_cte, make_env
from global_log import GlobalLog
from self_driving.state_utils import get_state_condition_function
from test.config import CTE_FITNESS_NAME
from test.fitness_utils import make_fitness
from test.individual import Individual
from test.state_individual import StateIndividual
from test_generators.random_seed_state_test_generator import (
    RandomSeedStateTestGenerator,
)
from utils.randomness import set_random_seed


class MockEvaluator(Evaluator):

    def __init__(self, fitness_threshold: float = None):
        super(MockEvaluator, self).__init__()
        self.logger = GlobalLog("mock_evaluator")
        self.fitness_threshold = fitness_threshold

    def run_sim(self, individual: Individual) -> None:

        self.logger.info("Executing individual with id {}".format(individual.id))
        self.logger.debug(
            "Individual implementation: {}".format(individual.get_implementation())
        )

        # TODO: possibly add a sleep

        nums = np.random.randint(low=5, high=100)
        ctes = list(np.random.normal(loc=0, scale=1.0, size=(nums,)))
        lateral_positions = list(np.random.normal(loc=0, scale=1.0, size=(nums,)))

        fitness = make_fitness(
            fitness_name=CTE_FITNESS_NAME,
            lateral_positions=lateral_positions,
            ctes=ctes,
            max_abs_value=get_max_cte(env_name=MOCK_SIM_NAME),
        )
        individual.set_fitness(fitness=fitness)
        if self.fitness_threshold is None:
            individual.set_success(is_success=np.random.choice([True, False]))
        else:
            individual.set_success(
                is_success=(
                    True if fitness.get_value() < self.fitness_threshold else False
                )
            )

        # set fitness and all possible features
        individual.set_behavioural_metrics(
            speeds=list(np.random.uniform(low=0.0, high=35.0, size=(nums,))),
            steering_angles=list(np.random.uniform(low=-1.0, high=1.0, size=(nums,))),
            lateral_positions=lateral_positions,
            ctes=ctes,
        )

    def close(self) -> None:
        pass


if __name__ == "__main__":
    env_name = MOCK_SIM_NAME
    seed = 0

    set_random_seed(seed=seed)

    env = make_env(
        simulator_name=env_name,
        seed=seed,
        port=-1,
        collect_trace=False,
    )

    path_to_csv_file = "../logs/donkey/sandbox_lab/reference_trace.csv"

    test_generator = RandomSeedStateTestGenerator(
        env_name=env_name,
        road_points=[],
        control_points=[],
        road_width=1,
        constant_road=True,
        state_condition_fn=get_state_condition_function(
            env_name=env_name, constant_road=True
        ),
        path_to_csv_file=path_to_csv_file,
    )

    evaluator = MockEvaluator()

    for i in range(5):
        state = test_generator.generate()
        individual = StateIndividual(state=state)
        print("Individual before mutation: {}".format(individual.get_representation()))
        individual = individual.mutate()
        print("Individual after mutation: {}".format(individual.get_representation()))
        evaluator.run_sim(individual=individual)
        print("Fitness: {}.".format(individual.get_fitness().get_value()))
        # Simulating computation time for computing next individual
        time.sleep(2)

    evaluator.close()
