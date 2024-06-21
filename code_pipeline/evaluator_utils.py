from code_pipeline.evaluator import Evaluator
from code_pipeline.mock_evaluator import MockEvaluator
from code_pipeline.real_evaluator import RealEvaluator
from config import SIMULATOR_NAMES
from custom_types import GymEnv
from self_driving.agent import Agent
from test.config import (
    EVALUATOR_NAMES,
    MOCK_EVALUATOR_NAME,
    REAL_EVALUATOR_NAME,
    FITNESS_NAMES,
)


def make_evaluator(
    evaluator_name: str,
    env_name: str,
    env: GymEnv,
    agent: Agent,
    fitness_name: str,
    collect_images: bool = False,
    max_abs_value_fitness: float = None,
    fitness_threshold: int = None,
) -> Evaluator:
    assert env_name in SIMULATOR_NAMES, "Env {} not found. Choose between {}".format(
        env_name, SIMULATOR_NAMES
    )
    assert evaluator_name in EVALUATOR_NAMES, "Evaluator name not in {}.".format(
        evaluator_name, EVALUATOR_NAMES
    )
    assert (
        fitness_name in FITNESS_NAMES
    ), "Fitness name {} not found. Choose between {}".format(
        fitness_name, FITNESS_NAMES
    )
    if evaluator_name == MOCK_EVALUATOR_NAME:
        assert fitness_threshold is not None, "Specify a fitness threshold"
        return MockEvaluator(fitness_threshold=fitness_threshold)
    if evaluator_name == REAL_EVALUATOR_NAME:
        return RealEvaluator(
            env_name=env_name,
            env=env,
            agent=agent,
            fitness_name=fitness_name,
            max_abs_value_fitness=max_abs_value_fitness,
            collect_images=collect_images,
        )
