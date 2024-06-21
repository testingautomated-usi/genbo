import os.path
from typing import Tuple, Type, Optional

import gym

from config import (
    SIMULATOR_NAMES,
    DONKEY_SIM_NAME,
    AGENT_TYPES,
    INPUT_SHAPE,
    MOCK_SIM_NAME,
)
from custom_types import GymEnv
from envs.donkey.config import MAX_SPEED_DONKEY, MIN_SPEED_DONKEY, MAX_CTE_ERROR_DONKEY
from envs.donkey.donkey_env_utils import make_simulator_scene
from envs.donkey.donkey_gym_env import DonkeyGymEnv
from envs.donkey.scenes.simulator_scenes import GENERATED_TRACK_NAME, SANDBOX_LAB_NAME
from envs.mock_gym_env import MockGymEnv
from self_driving.agent import Agent
from self_driving.autopilot_agent import AutopilotAgent
from self_driving.random_agent import RandomAgent
from self_driving.supervised_agent import SupervisedAgent
from test_generators.road_test_generator import RoadTestGenerator


def unwrap_wrapper(
    env: gym.Env, wrapper_class: Type[gym.Wrapper]
) -> Optional[gym.Wrapper]:
    """
    The MIT License

    Copyright (c) 2019 Antonin Raffin

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.

    Retrieve a ``VecEnvWrapper`` object by recursively searching.
    :param env: Environment to unwrap
    :param wrapper_class: Wrapper to look for
    :return: Environment unwrapped till ``wrapper_class`` if it has been wrapped with it
    """
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    return None


def is_wrapped(env: Type[gym.Env], wrapper_class: Type[gym.Wrapper]) -> bool:
    """
    The MIT License

    Copyright (c) 2019 Antonin Raffin

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.

    Check if a given environment has been wrapped with a given wrapper.
    :param env: Environment to check
    :param wrapper_class: Wrapper class to look for
    :return: True if environment has been wrapped with ``wrapper_class``.
    """
    return unwrap_wrapper(env, wrapper_class) is not None


def make_env(
    simulator_name: str,
    seed: int,
    port: int,
    sim_mul: int = 1,
    road_test_generator: RoadTestGenerator = None,
    donkey_exe_path: str = None,
    donkey_scene_name: str = None,
    headless: bool = False,
    collect_trace: bool = False,
    track_num: int = None,
    max_steps: int = None,
) -> GymEnv:
    assert (
        simulator_name in SIMULATOR_NAMES
    ), "Unknown simulator name {}. Choose among {}".format(
        simulator_name, SIMULATOR_NAMES
    )

    if simulator_name == DONKEY_SIM_NAME:
        return DonkeyGymEnv(
            seed=seed,
            add_to_port=port,
            road_test_generator=road_test_generator,
            simulator_scene=make_simulator_scene(
                scene_name=donkey_scene_name, track_num=track_num
            ),
            sim_mul=sim_mul,
            headless=headless,
            exe_path=donkey_exe_path,
            collect_trace=collect_trace,
            max_steps=max_steps,
        )

    if simulator_name == MOCK_SIM_NAME:
        return MockGymEnv()

    raise RuntimeError("Unknown simulator name: {}".format(simulator_name))


def get_max_min_speed(env_name: str) -> Tuple[int, int]:
    assert (
        env_name in SIMULATOR_NAMES
    ), "Unknown simulator name {}. Choose among {}".format(env_name, SIMULATOR_NAMES)

    if env_name == DONKEY_SIM_NAME:
        return MAX_SPEED_DONKEY, MIN_SPEED_DONKEY

    if env_name == MOCK_SIM_NAME:
        return 30, 10  # completely random

    raise RuntimeError("Unknown simulator name: {}".format(env_name))


def get_max_cte(env_name: str, donkey_scene_name: str = None) -> float:
    assert (
        env_name in SIMULATOR_NAMES
    ), "Unknown simulator name {}. Choose among {}".format(env_name, SIMULATOR_NAMES)

    if env_name == DONKEY_SIM_NAME:
        if donkey_scene_name == GENERATED_TRACK_NAME:
            return MAX_CTE_ERROR_DONKEY
        elif donkey_scene_name == SANDBOX_LAB_NAME:
            return MAX_CTE_ERROR_DONKEY + 3

    elif env_name == MOCK_SIM_NAME:
        # no particular reason to use one w.r.t. another for now
        return MAX_CTE_ERROR_DONKEY

    raise RuntimeError("Unknown simulator name {}".format(env_name))


def make_agent(
    env_name: str,
    env: GymEnv,
    agent_type: str,
    model_path: str,
    predict_throttle: bool = False,
    fake_images: bool = False,
    donkey_scene_name: str = None,
) -> Agent:
    assert agent_type in AGENT_TYPES, "Unknown agent type {}. Choose among {}".format(
        agent_type, AGENT_TYPES
    )
    assert (
        env_name in SIMULATOR_NAMES
    ), "Unknown simulator name {}. Choose among {}".format(env_name, SIMULATOR_NAMES)

    max_speed, min_speed = get_max_min_speed(env_name=env_name)

    if agent_type == "supervised":
        assert os.path.exists(model_path), "Model path {} does not exist".format(
            model_path
        )
        return SupervisedAgent(
            env=env,
            env_name=env_name,
            max_speed=max_speed,
            min_speed=min_speed,
            model_path=model_path,
            input_shape=INPUT_SHAPE,
            predict_throttle=predict_throttle,
            fake_images=fake_images,
        )

    if agent_type == "autopilot":
        return AutopilotAgent(
            env=env,
            env_name=env_name,
            max_speed=max_speed,
            min_speed=min_speed,
            donkey_scene_name=donkey_scene_name,
        )

    if agent_type == "random":
        return RandomAgent(env=env, env_name=env_name)

    raise RuntimeError("Unknown agent type: {}".format(agent_type))
