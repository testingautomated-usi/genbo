import argparse
import datetime
import os
import time

import gym
import numpy as np
from config import (
    SIMULATOR_NAMES,
    AGENT_TYPES,
    DONKEY_SIM_NAME,
    ROAD_TEST_GENERATOR_NAMES,
)
from envs.donkey.scenes.simulator_scenes import (
    SIMULATOR_SCENE_NAMES,
    GENERATED_TRACK_NAME,
)
from envs.env_utils import make_env, make_agent
from global_log import GlobalLog
from test_generators.road_generator_utils import make_road_test_generator
from utils.dataset_utils import save_archive
from utils.randomness import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs")
parser.add_argument(
    "--env-name", help="Env name", type=str, choices=SIMULATOR_NAMES, required=True
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
parser.add_argument(
    "--collect-trace",
    help="Collect driving trace in a csv file called REFERENCE_TRACE_FILENAME in the logs (only for DONKEY for now)",
    action="store_true",
    default=False,
)
parser.add_argument("--seed", help="Random seed", type=int, default=-1)
parser.add_argument(
    "--add-to-port", help="Modify default simulator port", type=int, default=-1
)
parser.add_argument(
    "--num-episodes", help="Number of tracks to generate", type=int, default=3
)
parser.add_argument(
    "--headless", help="Headless simulation", action="store_true", default=False
)
parser.add_argument(
    "--collect-images-with-supervised",
    help="Collect images despite agent is supervised",
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
    "--simulation-multiplier",
    help="Accelerate simulation (only Donkey)",
    choices=[1, 2, 3, 4, 5],
    type=int,
    default=1,
)
parser.add_argument(
    "--no-save-archive",
    help="Do not save the archive at the end of the episodes",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--regression",
    help="Stops the evaluation when the model fails more than 5 times",
    action="store_true",
    default=False,
)
args = parser.parse_args()


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    folder = args.folder
    logger = GlobalLog("evaluate")

    if args.seed == -1:
        try:
            args.seed = np.random.randint(2**32 - 1)
        except ValueError as e:
            args.seed = np.random.randint(2**30 - 1)

    set_random_seed(seed=args.seed)

    road_test_generator = make_road_test_generator(
        generator_name=args.road_test_generator_name,
        map_size=250,
        simulator_name=args.env_name,
        donkey_scene_name=args.donkey_scene_name,
        track_num=args.track_num,
    )

    env = make_env(
        simulator_name=args.env_name,
        seed=args.seed,
        port=args.add_to_port,
        sim_mul=args.simulation_multiplier,
        road_test_generator=road_test_generator,
        donkey_exe_path=args.donkey_exe_path,
        donkey_scene_name=args.donkey_scene_name,
        headless=args.headless,
        collect_trace=args.collect_trace,
        track_num=args.track_num,
        max_steps=args.max_steps,
    )
    agent = make_agent(
        env_name=args.env_name,
        env=env,
        model_path=args.model_path,
        agent_type=args.agent_type,
        predict_throttle=args.predict_throttle,
        donkey_scene_name=args.donkey_scene_name,
    )

    actions = []
    observations = []
    tracks = []
    times_elapsed = []
    is_success_flags = []
    car_position_x_episodes = []
    car_position_y_episodes = []
    episode_lengths = []
    lateral_positions_episodes = []
    ctes_episodes = []
    speeds_episodes = []
    steering_angles_episodes = []

    success_sum = 0

    episode_count = 0
    state_dict = dict()

    max_cte = -np.inf

    start_time = time.perf_counter()

    num_failures = 0

    while episode_count < args.num_episodes:
        done, state = False, None
        episode_length = 0
        car_positions_x = []
        car_positions_y = []
        lateral_positions = []
        ctes = []
        steering_angles = []
        speeds = []

        obs = env.reset()

        start_time = time.perf_counter()

        while not done:
            action = agent.predict(obs=obs, state=state_dict)
            # Clip Action to avoid out of bound errors
            if isinstance(env.action_space, gym.spaces.Box):
                action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, _, done, info = env.step(action)

            car_positions_x.append(info["pos"][0])
            car_positions_y.append(info["pos"][1])

            state_dict["cte"] = info.get("cte", None)
            state_dict["cte_pid"] = info.get("cte_pid", None)
            state_dict["speed"] = info.get("speed", None)

            state_dict["steering"] = info.get("steering", None)
            state_dict["throttle"] = info.get("throttle", None)

            if info.get("lateral_position", None) is not None:
                lateral_positions.append(float(info["lateral_position"]))

            if info.get("cte", None) is not None:
                ctes.append(float(info["cte"]))

            steering_angles.append(float(action[0]))

            if info.get("speed", None) is not None:
                speeds.append(float(info["speed"]))

            if state_dict["cte"] is not None:
                max_cte = max(abs(state_dict["cte"]), max_cte)

            # FIXME: first action is random for autopilots
            if episode_length > 0 and args.agent_type == "autopilot":
                actions.append(action)
                observations.append(obs)
            elif (
                args.agent_type != "autopilot"
                and args.agent_type == "supervised"
                and args.collect_images_with_supervised
            ):
                actions.append(action)
                observations.append(obs)
            elif args.agent_type == "supervised":
                actions.append(action)

            episode_length += 1

            if done:

                times_elapsed.append(time.perf_counter() - start_time)
                car_position_x_episodes.append(car_positions_x)
                car_position_y_episodes.append(car_positions_y)

                if info.get("track", None) is not None:
                    tracks.append(info["track"])

                if info.get("is_success", None) is not None:
                    success_sum += info["is_success"]
                    is_success_flags.append(info["is_success"])

                logger.debug("Episode #{}".format(episode_count + 1))
                logger.debug("Episode Length: {}".format(episode_length))
                logger.debug(
                    "Time elapsed: {}s".format(time.perf_counter() - start_time)
                )
                logger.debug("Is success: {}".format(info["is_success"]))

                if info["is_success"] == 0:
                    num_failures += 1

                logger.debug(f"Num failures: {num_failures}")

                if state_dict["cte"] is not None:
                    logger.debug("Max abs CTE: {}".format(max_cte))

                if len(lateral_positions) > 0:
                    lateral_positions_episodes.append(
                        (
                            np.mean(lateral_positions),
                            np.min(lateral_positions),
                            np.max(lateral_positions),
                            np.std(lateral_positions),
                        )
                    )

                if len(speeds) > 0:
                    speeds_episodes.append(
                        (
                            np.mean(speeds),
                            np.min(speeds),
                            np.max(speeds),
                            np.std(speeds),
                        )
                    )

                if len(steering_angles) > 0:
                    steering_angles_episodes.append(
                        (
                            np.mean(steering_angles),
                            np.min(steering_angles),
                            np.max(steering_angles),
                            np.std(steering_angles),
                        )
                    )

                if len(ctes) > 0:
                    ctes_episodes.append(
                        (np.mean(ctes), np.min(ctes), np.max(ctes), np.std(ctes))
                    )

                start_time = time.perf_counter()

                if episode_length <= 5:
                    logger.warn("Removing short episode")
                    if args.agent_type == "autopilot":
                        original_length_actions = len(actions)
                        original_length_observations = len(observations)
                        items_to_remove = (
                            episode_length - 1
                            if args.agent_type == "autopilot"
                            else episode_length
                        )
                        # first random action of each episode is not included
                        condition = (
                            episode_length > 1
                            if args.agent_type == "autopilot"
                            else episode_length > 0
                        )
                        while condition:
                            actions.pop()
                            observations.pop()
                            episode_length -= 1
                            condition = (
                                episode_length > 1
                                if args.agent_type == "autopilot"
                                else episode_length > 0
                            )

                        assert (
                            len(actions) + items_to_remove == original_length_actions
                        ), "Error when removing actions. To remove: {}, Original: {}, New: {}".format(
                            items_to_remove, original_length_actions, len(actions)
                        )
                        assert (
                            len(observations) + items_to_remove
                            == original_length_observations
                        ), "Error when removing observations. To remove: {}, Original: {}, New: {}".format(
                            items_to_remove,
                            original_length_observations,
                            len(observations),
                        )
                    elif args.agent_type == "supervised":
                        original_length_actions = len(actions)
                        items_to_remove = episode_length
                        while episode_length > 0:
                            actions.pop()
                            observations.pop()
                            episode_length -= 1
                            condition = (
                                episode_length > 1
                                if args.agent_type == "autopilot"
                                else episode_length > 0
                            )

                        assert (
                            len(actions) + items_to_remove == original_length_actions
                        ), "Error when removing actions. To remove: {}, Original: {}, New: {}".format(
                            items_to_remove, original_length_actions, len(actions)
                        )

                    if len(tracks) > 0:
                        track_to_repeat = tracks.pop()
                        road_test_generator.set_road_to_generate(road=track_to_repeat)

                else:
                    episode_lengths.append(episode_length)
                    episode_count += 1

                state_dict = {}

        if num_failures == 5 and args.regression:
            logger.debug("Break!")
            break

    if not args.regression:
        logger.debug("Success rate: {:.2f}".format(success_sum / episode_count))
        logger.debug(f"Is success flags: {is_success_flags}")
        if len(lateral_positions_episodes) > 0:
            means = np.mean([lp[0] for lp in lateral_positions_episodes])
            mins = np.mean([lp[1] for lp in lateral_positions_episodes])
            maxs = np.mean([lp[2] for lp in lateral_positions_episodes])
            stds = np.mean([lp[3] for lp in lateral_positions_episodes])
            logger.debug(
                f"Lateral positions. Mean: {means}, Min: {mins}, Max: {maxs}, Std: {stds}"
            )
            logger.debug(f"Lateral positions: {lateral_positions_episodes}")

        if len(ctes_episodes) > 0:
            means = np.mean([cte[0] for cte in ctes_episodes])
            mins = np.mean([cte[1] for cte in ctes_episodes])
            maxs = np.mean([cte[2] for cte in ctes_episodes])
            stds = np.mean([cte[3] for cte in ctes_episodes])
            logger.debug(f"Ctes. Mean: {means}, Min: {mins}, Max: {maxs}, Std: {stds}")
            logger.debug(f"Ctes: {ctes_episodes}")

        if len(steering_angles_episodes) > 0:
            means = np.mean([ste[0] for ste in steering_angles_episodes])
            mins = np.mean([ste[1] for ste in steering_angles_episodes])
            maxs = np.mean([ste[2] for ste in steering_angles_episodes])
            stds = np.mean([ste[3] for ste in steering_angles_episodes])
            logger.debug(
                f"Steering angles. Mean: {means}, Min: {mins}, Max: {maxs}, Std: {stds}"
            )
            logger.debug(f"Steering angles: {steering_angles_episodes}")

        if len(speeds_episodes) > 0:
            means = np.mean([speed[0] for speed in speeds_episodes])
            mins = np.mean([speed[1] for speed in speeds_episodes])
            maxs = np.mean([speed[2] for speed in speeds_episodes])
            stds = np.mean([speed[3] for speed in speeds_episodes])
            logger.debug(
                f"Speeds. Mean: {means}, Min: {mins}, Max: {maxs}, Std: {stds}"
            )
            logger.debug(f"Speeds: {speeds_episodes}")

        logger.debug("Mean time elapsed: {:.2f}s".format(np.mean(times_elapsed)))

        if args.road_test_generator_name == "constant":
            archive_name = "{}-{}-archive-agent-{}-episodes-{}".format(
                args.env_name,
                datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
                args.agent_type,
                args.num_episodes,
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

        if not args.no_save_archive:
            save_archive(
                actions=actions,
                observations=observations,
                is_success_flags=is_success_flags,
                tracks=tracks,
                car_positions_x_episodes=car_position_x_episodes,
                car_positions_y_episodes=car_position_y_episodes,
                episode_lengths=episode_lengths,
                archive_path=folder,
                archive_name=archive_name,
            )

    env.reset(skip_generation=True)

    if args.env_name == DONKEY_SIM_NAME:
        time.sleep(2)
        env.exit_scene()
        env.close_connection()

    time.sleep(5)
    env.close()
