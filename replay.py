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
from envs.vec_env.dummy_vec_env import DummyVecEnv
from envs.vec_env.vec_video_recorder import VecVideoRecorder
from global_log import GlobalLog
from test.archive import Archive
from test.config import (
    INDIVIDUAL_NAMES,
    STATE_PAIR_INDIVIDUAL_NAME,
    EVALUATOR_NAMES,
    FITNESS_NAMES,
    CTE_FITNESS_NAME,
    SEED_STATE_GENERATOR_TYPE,
    MOCK_EVALUATOR_NAME,
    INDIVIDUAL_GENERATOR_NAMES,
    SEQUENCE_GENERATOR_NAME,
)
from test_generators.replay_individual_generator import ReplayIndividualGenerator
from test_generators.road_generator_utils import make_road_test_generator
from utils.dataset_utils import save_archive
from utils.randomness import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs")
parser.add_argument(
    "--env-name", help="Env name", type=str, choices=SIMULATOR_NAMES, required=True
)
parser.add_argument(
    "--archive-filenames",
    nargs="+",
    help="Name of the archive file to load, without extension",
    type=str,
    default=None,
    required=True,
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
    "--collect-images",
    help="Collect images for further fine-tuning",
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
    "--individual-generator-name",
    help="Name of the individual generator",
    choices=INDIVIDUAL_GENERATOR_NAMES,
    type=str,
    default=SEQUENCE_GENERATOR_NAME,
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
    "--simulation-multiplier",
    help="Accelerate simulation (only Donkey)",
    choices=[1, 2, 3, 4, 5],
    type=int,
    default=1,
)
parser.add_argument(
    "--mode",
    help="Replay mode",
    type=str,
    choices=["replicate", "recovery"],
    default="replicate",
)
parser.add_argument(
    "--num-runs",
    help="Num runs for replaying to take the randomness of the simulator into account",
    type=int,
    default=5,
)
parser.add_argument(
    "--video",
    help="Collect video of the simulation",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--execute-failure-member",
    help="Execute the failed member of the pair (if individual_name = 'state_pair_individual')",
    action="store_true",
    default=False,
)

args, _ = parser.parse_known_args()


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    folder = args.folder
    logger = GlobalLog("replay")

    if args.seed == -1:
        try:
            args.seed = np.random.randint(2**32 - 1)
        except ValueError as e:
            args.seed = np.random.randint(2**30 - 1)

    generator_type = SEED_STATE_GENERATOR_TYPE

    set_random_seed(seed=args.seed)

    road_test_generator = make_road_test_generator(
        generator_name=args.road_test_generator_name,
        map_size=250,
        simulator_name=args.env_name,
        donkey_scene_name=args.donkey_scene_name,
        track_num=args.track_num,
    )

    datetime_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    filename_no_ext = "{}_{}_execute_{}".format(
        datetime_str, args.mode, "fail" if args.execute_failure_member else "success"
    )
    if args.agent_type == "supervised":
        filename_no_ext += "_{}_on_".format(
            args.model_path[args.model_path.rindex("/") + 1 :].split(".")[0]
        )
    else:
        filename_no_ext += "_autopilot_on_"

    example_filename = args.archive_filenames[0]
    on_autopilot = example_filename.find("autopilot") != -1
    on_dave2 = example_filename.find("dave2") != -1

    if on_autopilot:
        filename_no_ext += "autopilot"
    else:
        assert (
            on_dave2
        ), "Not possible to determine the model that produced the boundary states. Archive filename: {}".format(
            args.archive_filename[0]
        )
        filename_no_ext += "{}".format(
            example_filename[
                example_filename.find("dave2") : example_filename.find("_iterations")
            ]
        )

    if args.collect_images:
        filename_no_ext += "_collect_images"

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

    logging.basicConfig(
        filename=os.path.join(archive_logdir, "{}.txt".format(filename_no_ext)),
        filemode="w",
    )
    logger.info("================= ARGS: {} =================".format(args))

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
        if args.video and not args.collect_images:
            env = DummyVecEnv([lambda: env])

            # Record the video starting at the first step
            env = VecVideoRecorder(
                venv=env,
                video_folder=archive_logdir,
                record_video_trigger=lambda x: x == 0,
                name_prefix=filename_no_ext,
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

    archive = Archive(replay=True)
    replay_individual_generator = ReplayIndividualGenerator(
        evaluator=evaluator,
        individual_name=args.individual_name,
        num_runs=args.num_runs,
    )

    if args.mode == "replicate":

        for archive_num, archive_filename in enumerate(args.archive_filenames):
            archive_filepath = os.path.join(
                archive_logdir, "{}.json".format(archive_filename)
            )
            assert os.path.exists(
                archive_filepath
            ), "Archive file {} does not exist".format(archive_filepath)

            logger.info(
                f"Replicating individuals in the archive {archive_filename}: archive num {archive_num}"
            )

            archive = Archive(replay=True)
            individuals = archive.load(
                filepath=archive_logdir,
                filename_no_ext=archive_filename,
                individual_name=args.individual_name,
                check_null_state=False,
            )

            individual_runs = replay_individual_generator.replicate(
                individuals=individuals, close=False
            )
            archive.set_individual_properties(individual_runs=individual_runs)
            archive.save(filepath=archive_logdir, filename_no_ext=archive_filename)

        replay_individual_generator.evaluator.close()

    elif args.mode == "recovery":

        if args.agent_type == "supervised":
            logging.info("Model being executed: {}".format(args.model_path))

        individual_dicts = dict()

        observations = {i: [] for i in range(len(args.archive_filenames))}
        actions = {i: [] for i in range(len(args.archive_filenames))}
        is_success_flags = {i: [] for i in range(len(args.archive_filenames))}
        episode_lengths = {i: [] for i in range(len(args.archive_filenames))}
        for i in range(len(args.archive_filenames)):
            archive_filename = args.archive_filenames[i]
            archive_filepath = os.path.join(
                archive_logdir, "{}.json".format(archive_filename)
            )
            assert os.path.exists(
                archive_filepath
            ), "Archive file {} does not exist".format(archive_filepath)

            logger.info(
                "Executing individuals in {} for recovery analysis using {} agent".format(
                    archive_filename, args.agent_type
                )
            )

            individuals = archive.load(
                filepath=archive_logdir,
                filename_no_ext=archive_filename,
                individual_name=args.individual_name,
                check_null_state=False,
            )

            # filter individuals by taking only those with replicable_percentage > 0%
            individuals = list(
                filter(lambda ind: ind.replicable_percentage > 0, individuals)
            )
            if len(individuals) > 0:
                logger.info(
                    "Found {} individuals with a replicable percentage > 0".format(
                        len(individuals)
                    )
                )
                individual_successful_runs, state_individuals = (
                    replay_individual_generator.replay_for_recovery(
                        individuals=individuals,
                        execute_success=not args.execute_failure_member,
                    )
                )
                individual_dicts[args.archive_filenames[i]] = individual_successful_runs
                if args.collect_images:
                    observations[i].extend(
                        [
                            individual.get_observations()
                            for individual in state_individuals
                        ]
                    )
                    actions[i].extend(
                        [individual.get_actions() for individual in state_individuals]
                    )
                    is_success_flags[i].extend(
                        [individual.is_success() for individual in state_individuals]
                    )
                    episode_lengths[i].extend(
                        [
                            len(individual.get_observations())
                            for individual in state_individuals
                        ]
                    )
            else:
                individual_dicts[args.archive_filenames[i]] = {
                    "replicable_percentage": {
                        individual.id: individual.replicable_percentage
                    }
                    for individual in individuals
                }
                logger.info("No individuals with a replicable percentage > 0")

        if args.collect_images:
            if args.road_test_generator_name == "constant":
                model_names = []
                for archive_name in args.archive_filenames:
                    str_to_match = "agent_supervised"
                    idx_start = archive_name.find(str_to_match)
                    idx_end = archive_name.find("_iterations")
                    model_name = archive_name[
                        idx_start + len(str_to_match) + 1 : idx_end
                    ]
                    model_names.append(model_name)

                assert (
                    len(model_names) > 0 and len(set(model_names)) == 1
                ), f"There is some error when setting the model name: {set(model_names)}"

                archive_name = "{}-{}-finetuning-agent-{}".format(
                    args.env_name,
                    datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
                    model_names[0],
                )
                if (
                    args.env_name == DONKEY_SIM_NAME
                    and args.donkey_scene_name == GENERATED_TRACK_NAME
                ):
                    archive_name += "-{}".format(args.donkey_scene_name)
                    if args.donkey_scene_name == GENERATED_TRACK_NAME:
                        archive_name += "-{}".format(args.track_num)
            else:
                archive_name = "{}-{}-archive-agent-{}-seed-{}-episodes-{}-max-angle-{}-length-{}".format(
                    args.env_name,
                    datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
                    args.agent_type,
                    args.seed,
                    args.num_episodes,
                    args.max_angle,
                    args.num_control_nodes,
                )

            if args.simulation_multiplier > 1:
                archive_name += "-{}x".format(args.simulation_multiplier)

            # I am not concatenating now, as without concatenation it is easier to filter
            # such data by means of the is_success_flags; concatenation is done when loading the dataset
            # after filtering has occured
            # observations = np.concatenate(observations)
            # actions = np.concatenate(actions)

            for i in observations.keys():
                save_archive(
                    actions=actions[i],
                    observations=observations[i],
                    is_success_flags=is_success_flags[i],
                    episode_lengths=episode_lengths[i],
                    archive_path=folder,
                    archive_name=archive_name + f"-run-{i + 1}",
                )

        replay_individual_generator.evaluator.close()
        if not args.collect_images:
            replay_individual_generator.save_recovery_file(
                filepath=archive_logdir,
                filename_no_ext=filename_no_ext,
                individual_dicts=individual_dicts,
                agent_type=args.agent_type,
                model_path=args.model_path,
            )

    else:
        raise RuntimeError("Unknown mode: {}".format(args.mode))
