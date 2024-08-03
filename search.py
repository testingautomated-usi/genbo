import argparse
import datetime
import logging
import os

import numpy as np

from code_pipeline.evaluator_utils import make_evaluator
from config import (
    SIMULATOR_NAMES,
    AGENT_TYPES,
    ROAD_TEST_GENERATOR_NAMES,
    DONKEY_SIM_NAME,
)
from envs.donkey.scenes.simulator_scenes import (
    SIMULATOR_SCENE_NAMES,
    GENERATED_TRACK_NAME,
)
from envs.env_utils import make_env, make_agent, get_max_cte
from global_log import GlobalLog
from test.config import (
    INDIVIDUAL_NAMES,
    STATE_PAIR_INDIVIDUAL_NAME,
    SEED_STATE_GENERATOR_NAMES,
    RANDOM_SEED_STATE_GENERATOR_NAME,
    INDIVIDUAL_GENERATOR_NAMES,
    EVALUATOR_NAMES,
    FITNESS_NAMES,
    CTE_FITNESS_NAME,
    SEED_STATE_GENERATOR_TYPE,
    MOCK_EVALUATOR_NAME,
    SEQUENCE_GENERATOR_NAME,
)
from test_generators.individual_generator_utils import make_individual_generator
from test_generators.replay_individual_generator import ReplayIndividualGenerator
from test_generators.road_generator_utils import make_road_test_generator
from test_generators.seed_state_generator_utils import make_seed_state_generator
from utils.randomness import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs")
parser.add_argument(
    "--env-name", help="Env name", type=str, choices=SIMULATOR_NAMES, required=True
)
parser.add_argument(
    "--evaluator-name",
    help="Evaluator name",
    type=str,
    choices=EVALUATOR_NAMES,
    required=True,
)
parser.add_argument(
    "--donkey-exe-path",
    help="Path to the donkey simulator executor",
    type=str,
    default=None,
)
parser.add_argument(
    "--donkey-scene-name",
    help="Scene name for the donkey simulator",
    choices=SIMULATOR_SCENE_NAMES,
    type=str,
    default=GENERATED_TRACK_NAME,
)
parser.add_argument("--seed", help="Random seed", type=int, default=-1)
parser.add_argument(
    "--add-to-port", help="Modify default simulator port", type=int, default=-1
)
parser.add_argument(
    "--headless", help="Headless simulation", action="store_true", default=False
)
parser.add_argument(
    "--bias",
    help="Bias the search towards higher (in absolute value) values",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--collect-images",
    help="Collect images during the search process",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--agent-type", help="Agent type", type=str, choices=AGENT_TYPES, default="random"
)
parser.add_argument(
    "--road-test-generator-name",
    help="Which road test generator to use",
    type=str,
    choices=ROAD_TEST_GENERATOR_NAMES,
    default="constant",
)
parser.add_argument(
    "--model-path",
    help="Path to agent model with extension (only if agent_type == 'supervised')",
    type=str,
    default=None,
)
parser.add_argument(
    "--predict-throttle",
    help="Predict steering and throttle. Model to load must have been trained using an output dimension of 2",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--track-num",
    help="Track number when simulator_name is Donkey and simulator_scene is GeneratedTrack",
    type=int,
    default=0,
)
parser.add_argument(
    "--max-steps",
    help="Max steps of the environment (only for DONKEY at the moment)",
    type=int,
    default=None,
)
parser.add_argument(
    "--individual-name",
    help="Name of the individual to instantiate for the search process",
    choices=INDIVIDUAL_NAMES,
    type=str,
    default=STATE_PAIR_INDIVIDUAL_NAME,
)
parser.add_argument(
    "--fitness-name",
    help="Fitness name",
    choices=FITNESS_NAMES,
    type=str,
    default=CTE_FITNESS_NAME,
)
parser.add_argument(
    "--fitness-threshold",
    help="Fitness threshold (only when evaluator_name == 'mock')",
    type=float,
    default=None,
)
parser.add_argument(
    "--seed-state-generator-name",
    help="Name of the seed state generator",
    choices=SEED_STATE_GENERATOR_NAMES,
    type=str,
    default=RANDOM_SEED_STATE_GENERATOR_NAME,
)
parser.add_argument(
    "--individual-generator-name",
    help="Name of the individual generator",
    choices=INDIVIDUAL_GENERATOR_NAMES,
    type=str,
    default=SEQUENCE_GENERATOR_NAME,
)
parser.add_argument(
    "--num-iterations", help="Num iterations of the search", type=int, default=25
)
parser.add_argument(
    "--num-restarts", help="Num restarts of the search", type=int, default=25
)
parser.add_argument(
    "--lam",
    help="Lambda parameter (only when individual_name_generator == 'one_plus_lambda')",
    type=int,
    default=1,
)
parser.add_argument(
    "--length-exponential-factor",
    help="Length exponential factor (only when individual_name_generator == 'sequence')",
    default=1.1,
)
parser.add_argument(
    "--simulation-multiplier",
    help="Accelerate simulation (only Donkey)",
    choices=[1, 2, 3, 4, 5],
    type=int,
    default=1,
)
parser.add_argument(
    "--do-not-replay",
    help="Do not replay the individuals in the archive",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--num-runs",
    help="Num of runs when replaying each individual to take the randomness of the simulator into account",
    type=int,
    default=5,
)
parser.add_argument(
    "--num-runs-failure",
    help="Num runs to confirm that an execution of an individual is a failure during the search (to take the randomness of the simulator into account)",
    type=int,
    default=1,
)
parser.add_argument(
    "--mutate-both-members",
    help="Mutate both members of the individual (only with individual-name == 'state_pair_individual')",
    action="store_true",
    default=False,
)

args, _ = parser.parse_known_args()


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    folder = args.folder
    logger = GlobalLog("search")

    if args.seed == -1:
        try:
            args.seed = np.random.randint(2**32 - 1)
        except ValueError as e:
            args.seed = np.random.randint(2**30 - 1)

    generator_type = SEED_STATE_GENERATOR_TYPE

    archive_logdir = os.path.join(
        args.folder,
        "test_generation",
        args.individual_generator_name,
        generator_type,
        args.env_name,
    )
    if args.env_name == DONKEY_SIM_NAME:
        if args.donkey_scene_name == GENERATED_TRACK_NAME:
            archive_logdir = os.path.join(
                archive_logdir, "{}_{}".format(args.donkey_scene_name, args.track_num)
            )
        else:
            archive_logdir = os.path.join(
                archive_logdir, "{}".format(args.donkey_scene_name)
            )

    os.makedirs(name=archive_logdir, exist_ok=True)

    datetime_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    archive_filename = "{}_seed_{}_agent_{}".format(
        datetime_str, args.seed, args.agent_type
    )

    if args.agent_type == "supervised":
        model_path = args.model_path.replace(args.env_name + "-", "")
        last_slash_index = model_path.rindex(os.path.sep)
        last_dot_index = model_path.rindex(".")
        archive_filename += "_{}".format(
            model_path[last_slash_index + 1 : last_dot_index]
        )

    archive_filename += "_iterations_{}".format(args.num_iterations)
    archive_filename += "_restarts_{}".format(args.num_restarts)

    if args.individual_generator_name == SEQUENCE_GENERATOR_NAME:
        archive_filename += "_len_exp_factor_{}".format(args.length_exponential_factor)

    logging.basicConfig(
        filename=os.path.join(
            archive_logdir, "{}_seed_{}_log.txt".format(datetime_str, args.seed)
        ),
        filemode="w",
    )
    logger.info("================= ARGS: {} =================".format(args))

    args.length_exponential_factor = float(args.length_exponential_factor)
    if args.individual_generator_name == SEQUENCE_GENERATOR_NAME:
        assert (
            args.length_exponential_factor > 0
        ), "Length exponential factor must be > 0"
        args.length_exponential_factor = int(args.length_exponential_factor)
        logger.info("Linear sequence: {}".format(args.length_exponential_factor))

    set_random_seed(seed=args.seed)

    road_test_generator = make_road_test_generator(
        generator_name=args.road_test_generator_name,
        map_size=250,
        simulator_name=args.env_name,
        donkey_scene_name=args.donkey_scene_name,
        track_num=args.track_num,
    )

    if args.evaluator_name == MOCK_EVALUATOR_NAME:
        env = None
    else:
        env = make_env(
            simulator_name=args.env_name,
            seed=args.seed,
            donkey_exe_path=args.donkey_exe_path,
            donkey_scene_name=args.donkey_scene_name,
            port=args.add_to_port,
            sim_mul=args.simulation_multiplier,
            collect_trace=False,
            headless=args.headless,
            track_num=args.track_num,
            max_steps=args.max_steps,
            road_test_generator=road_test_generator,
        )

    seed_test_generator = make_seed_state_generator(
        generator_name=args.seed_state_generator_name,
        env_name=args.env_name,
        constant_road=args.road_test_generator_name == "constant",
        donkey_scene_name=args.donkey_scene_name,
        track_num=args.track_num,
        folder=folder,
    )

    agent = make_agent(
        env_name=args.env_name,
        donkey_scene_name=args.donkey_scene_name,
        env=env,
        model_path=args.model_path,
        agent_type=args.agent_type,
        predict_throttle=args.predict_throttle,
    )

    max_abs_value_fitness = (
        get_max_cte(env_name=args.env_name, donkey_scene_name=args.donkey_scene_name)
        if args.fitness_name == CTE_FITNESS_NAME
        else None
    )

    evaluator = make_evaluator(
        env=env,
        evaluator_name=args.evaluator_name,
        env_name=args.env_name,
        agent=agent,
        fitness_name=args.fitness_name,
        collect_images=args.collect_images,
        max_abs_value_fitness=max_abs_value_fitness,
        fitness_threshold=args.fitness_threshold,
    )

    individual_generator = make_individual_generator(
        generator_name=args.individual_generator_name,
        env_name=args.env_name,
        individual_name=args.individual_name,
        generator_type=generator_type,
        evaluator=evaluator,
        archive_logdir=archive_logdir,
        archive_filename=archive_filename,
        num_restarts=args.num_restarts,
        num_runs_failure=args.num_runs_failure,
        length_exponential_factor=args.length_exponential_factor,
        lam=args.lam,
        maximize=True,
        seed_state_test_generator=seed_test_generator,
        bias=args.bias,
        mutate_both_members=args.mutate_both_members,
    )

    archive = individual_generator.evolve(
        num_iterations=args.num_iterations, close_at_last=args.do_not_replay
    )

    if not args.do_not_replay:
        replicate_individual_generator = ReplayIndividualGenerator(
            evaluator=evaluator,
            individual_name=args.individual_name,
            num_runs=args.num_runs,
        )
        individual_runs = replicate_individual_generator.replicate(
            individuals=archive.get_individuals()
        )
        archive.set_individual_properties(individual_runs=individual_runs)
        archive.save(filepath=archive_logdir, filename_no_ext=archive_filename)
