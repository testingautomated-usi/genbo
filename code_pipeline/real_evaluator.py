import os.path
import time
from typing import Tuple, List

import gym
import numpy as np

from code_pipeline.evaluator import Evaluator
from config import DONKEY_SIM_NAME
from custom_types import GymEnv
from envs.donkey.donkey_env_utils import make_simulator_scene
from envs.donkey.scenes.simulator_scenes import GENERATED_TRACK_NAME
from envs.env_utils import make_env, make_agent, get_max_cte
from envs.vec_env.base_vec_env import VecEnv
from global_log import GlobalLog
from self_driving.agent import Agent
from self_driving.state_utils import get_state_condition_function
from test.config import CTE_FITNESS_NAME
from test.fitness_utils import make_fitness
from test.individual import Individual
from test.state_individual import StateIndividual

import platform

from test_generators.replicate_test_generator import ReplicateStateTestGenerator
from utils.randomness import set_random_seed


class RealEvaluator(Evaluator):

    def __init__(
        self,
        env_name: str,
        env: GymEnv,
        agent: Agent,
        fitness_name: str,
        collect_images: bool = False,
        max_abs_value_fitness: float = None,
    ):
        super(RealEvaluator, self).__init__()
        self.env_name = env_name
        self.env = env
        self.agent = agent
        self.logger = GlobalLog("real_evaluator")
        self.collect_images = collect_images
        self.fitness_name = fitness_name
        self.max_abs_value_fitness = max_abs_value_fitness

    def _run(self, individual: Individual) -> Tuple[
        bool,
        int,
        List[float],
        List[float],
        List[float],
        List[float],
        List[np.ndarray],
        List[np.array],
    ]:
        done = False
        start_time = time.perf_counter()
        obs = self.env.reset(individual=individual)

        steering_angles = []
        lateral_positions = []
        ctes = []
        episode_length = 0
        state_dict = {}
        speeds = []
        observations = []
        actions = []
        is_success = False

        while not done:
            if episode_length == 0:
                action = np.asarray([0.0, 0.0])
            else:
                action = self.agent.predict(obs=obs, state=state_dict)
            # Clip Action to avoid out of bound errors
            if isinstance(self.env.action_space, gym.spaces.Box):
                action = np.clip(
                    action, self.env.action_space.low, self.env.action_space.high
                )
            obs, _, done, info = self.env.step(action)

            if isinstance(self.env, VecEnv):
                obs = obs[0]
                done = done[0]
                info = info[0]

            state_dict["cte"] = info.get("cte", None)
            state_dict["cte_pid"] = info.get("cte_pid", None)
            state_dict["speed"] = info.get("speed", None)

            steering_angles.append(action[0])
            lateral_position = info.get("lateral_position", None)
            assert lateral_position is not None, "Lateral position needs to be present"
            lateral_positions.append(lateral_position)
            ctes.append(state_dict["cte"])
            speeds.append(info.get("speed", None))

            episode_length += 1
            # removing logging of first image
            # if self.collect_images or (LOGGING_LEVEL == "DEBUG" and episode_length == 1):
            if self.collect_images:
                observations.append(obs)
                actions.append(action)
            if done:
                is_success = bool(info["is_success"])
                self.logger.debug("Episode length: {}".format(episode_length))
                self.logger.debug("Is success: {}".format(is_success))

        self.logger.debug(
            "Episode executed in {:.2f}s".format(time.perf_counter() - start_time)
        )
        return (
            is_success,
            episode_length,
            steering_angles,
            lateral_positions,
            ctes,
            speeds,
            observations,
            actions,
        )

    def run_sim(self, individual: Individual) -> None:

        (
            is_success,
            episode_length,
            steering_angles,
            lateral_positions,
            ctes,
            speeds,
            observations,
            actions,
        ) = self._run(individual=individual)
        max_repetitions = 0
        while episode_length < 5 and max_repetitions < 10:
            self.logger.debug(
                "{} Repeat short episode: {}".format(max_repetitions, episode_length)
            )
            (
                is_success,
                episode_length,
                steering_angles,
                lateral_positions,
                ctes,
                speeds,
                observations,
                actions,
            ) = self._run(individual=individual)
            max_repetitions += 1

        assert (
            max_repetitions < 10
        ), "Unable to successfully execute individual with id {}".format(individual.id)

        individual.set_success(is_success=is_success)

        # set fitness and all possible features
        individual.set_fitness(
            make_fitness(
                fitness_name=self.fitness_name,
                lateral_positions=lateral_positions,
                ctes=ctes,
                max_abs_value=self.max_abs_value_fitness,
            )
        )

        individual.set_behavioural_metrics(
            speeds=speeds,
            steering_angles=steering_angles,
            lateral_positions=lateral_positions,
            ctes=ctes,
        )

        individual.set_observations(observations=observations)
        individual.set_actions(actions=actions)

    def close(self) -> None:
        self.env.reset(skip_generation=True)

        if self.env_name == DONKEY_SIM_NAME:
            time.sleep(2)
            self.env.exit_scene()
            self.env.close_connection()

        time.sleep(5)
        self.env.close()


if __name__ == "__main__":

    env_name = DONKEY_SIM_NAME
    seed = 0
    platform_ = platform.system()
    donkey_exe_path = "../../../Downloads/DonkeySimMacBostage/donkey_sim.app"
    donkey_scene_name = GENERATED_TRACK_NAME
    track_num = 0
    headless = False

    set_random_seed(seed=seed)

    assert platform_.lower() == "darwin", "Only on MacOS for now"
    assert os.path.exists(donkey_exe_path), "Donkey executor file not found: {}".format(
        donkey_exe_path
    )

    env = make_env(
        simulator_name=env_name,
        seed=seed,
        donkey_exe_path=donkey_exe_path,
        donkey_scene_name=donkey_scene_name,
        port=-1,
        collect_trace=False,
        headless=headless,
        track_num=track_num,
    )

    state_dict = {
        "pos_x": 124.9955,
        "pos_y": 0.60000001,
        "pos_z": 36.50955,
        "rot_x": 0.007688666,
        "rot_y": 0.006867982,
        "rot_z": 0.007399417,
        "rot_w": -0.9999195,
        "rotation_angle": 359.2195,
        "vel_x": 0.13958722342549928,
        "vel_z": 13.023627270163642,
    }

    test_generator = ReplicateStateTestGenerator(
        env_name=env_name,
        road_points=[],
        control_points=[],
        road_width=1,
        constant_road=True,
        state_condition_fn=get_state_condition_function(
            env_name=env_name, constant_road=True
        ),
        path_to_csv_file=None,
        state_dict=state_dict,
        simulator_scene=make_simulator_scene(
            scene_name=donkey_scene_name, track_num=track_num
        ),
    )

    # path_to_csv_file = "../logs/donkey/sandbox_lab/reference_trace.csv"
    # path_to_csv_file = "../logs/donkey/generated_track/reference_trace_0.csv"
    #
    # test_generator = RandomStateTestGenerator(
    #     env_name=env_name,
    #     road_points=[],
    #     control_points=[],
    #     road_width=1,
    #     constant_road=True,
    #     state_condition_fn=get_state_condition_function(env_name=env_name, constant_road=True),
    #     path_to_csv_file=path_to_csv_file,
    #     simulator_scene=make_simulator_scene(scene_name=donkey_scene_name, track_num=track_num)
    # )

    # model_path = '../logs/models/mixed-dave2-2022_06_04_14_03_27.h5'  # robust model
    # model_path = '../logs/models/mixed-dave2-2022_06_07_15_51_20.h5'  # weak model
    # assert os.path.exists(model_path), "Model path not found: {}".format(model_path)

    model_path = None
    agent_type = "autopilot"

    agent = make_agent(
        env_name=env_name,
        donkey_scene_name=donkey_scene_name,
        env=env,
        model_path=model_path,
        agent_type=agent_type,
        predict_throttle=False,
    )

    evaluator = RealEvaluator(
        env_name=env_name,
        env=env,
        agent=agent,
        fitness_name=CTE_FITNESS_NAME,
        max_abs_value_fitness=get_max_cte(
            env_name=env_name, donkey_scene_name=donkey_scene_name
        ),
        collect_images=False,
    )

    for i in range(2):
        state = test_generator.generate()
        individual = StateIndividual(state=state)
        # print("Individual before mutation: {}".format(individual.get_representation()))
        # individual = individual.mutate()
        # print("Individual after mutation: {}".format(individual.get_representation()))
        evaluator.run_sim(individual=individual)
        print("Fitness: {}.".format(individual.get_fitness().get_value()))

    evaluator.close()
