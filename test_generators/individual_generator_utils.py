from typing import Union

from code_pipeline.evaluator import Evaluator
from config import SIMULATOR_NAMES
from test.config import (
    INDIVIDUAL_GENERATOR_NAMES,
    GENERATOR_TYPES,
    ONE_PLUS_LAMBDA_GENERATOR_NAME,
    SEQUENCE_GENERATOR_NAME,
)
from test_generators.individual_generator import IndividualGenerator
from test_generators.one_plus_lambda_individual_generator import (
    OnePlusLambdaIndividualGenerator,
)
from test_generators.road_test_generator import RoadTestGenerator
from test_generators.sequence_individual_generator import SequenceIndividualGenerator
from test_generators.state_test_generator import SeedStateTestGenerator


def make_individual_generator(
    generator_name: str,
    env_name: str,
    individual_name: str,
    generator_type: str,
    evaluator: Evaluator,
    archive_logdir: str,
    archive_filename: str,
    num_restarts: int,
    length_exponential_factor: Union[float, int] = 1.1,
    lam: int = 1,
    start_id: int = 1,
    num_runs_failure: int = 1,
    maximize: bool = True,
    seed_state_test_generator: SeedStateTestGenerator = None,
    road_test_generator: RoadTestGenerator = None,
    bias: bool = False,
    mutate_both_members: bool = False,
) -> IndividualGenerator:
    assert env_name in SIMULATOR_NAMES, "Env {} not found. Choose between {}".format(
        env_name, SIMULATOR_NAMES
    )
    assert (
        generator_name in INDIVIDUAL_GENERATOR_NAMES
    ), "Test generator {} not found. Choose between {}".format(
        generator_name, INDIVIDUAL_GENERATOR_NAMES
    )
    assert generator_type in GENERATOR_TYPES, "Generator type must be in {}".format(
        GENERATOR_TYPES
    )
    if generator_name == SEQUENCE_GENERATOR_NAME:
        return SequenceIndividualGenerator(
            start_id=start_id,
            evaluator=evaluator,
            individual_name=individual_name,
            num_restarts=num_restarts,
            length_exponential_factor=length_exponential_factor,
            generator_type=generator_type,
            archive_logdir=archive_logdir,
            archive_filename=archive_filename,
            num_runs_failure=num_runs_failure,
            maximize=maximize,
            seed_state_test_generator=seed_state_test_generator,
            road_test_generator=road_test_generator,
            bias=bias,
            mutate_both_members=mutate_both_members,
        )
    if generator_name == ONE_PLUS_LAMBDA_GENERATOR_NAME:
        return OnePlusLambdaIndividualGenerator(
            start_id=start_id,
            evaluator=evaluator,
            individual_name=individual_name,
            num_restarts=num_restarts,
            generator_type=generator_type,
            archive_logdir=archive_logdir,
            archive_filename=archive_filename,
            num_runs_failure=num_runs_failure,
            maximize=maximize,
            lam=lam,
            seed_state_test_generator=seed_state_test_generator,
            road_test_generator=road_test_generator,
            bias=bias,
            mutate_both_members=mutate_both_members,
        )

    raise NotImplementedError("{} not supported yet".format(generator_name))
