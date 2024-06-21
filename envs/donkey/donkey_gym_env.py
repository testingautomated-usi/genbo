# Original author: Roma Sokolkov
# Edited by Antonin Raffin
"""
MIT License

Copyright (c) 2018 Roma Sokolkov
Copyright (c) 2018 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os

import gym
import numpy as np
from gym import spaces

from config import DONKEY_SIM_NAME
from envs.donkey.config import (
    BASE_PORT,
    BASE_SOCKET_LOCAL_ADDRESS,
    MAX_STEERING,
    INPUT_DIM,
)
from envs.donkey.core.donkey_sim import DonkeyUnitySimController
from envs.donkey.scenes.simulator_scenes import SimulatorScene
from envs.unity_proc import UnityProcess
from global_log import GlobalLog
from test.individual import Individual
from test_generators.road_test_generator import RoadTestGenerator
from custom_types import ObserveData


class DonkeyGymEnv(gym.Env):
    """
    Gym interface for DonkeyCar with support for using
    a VAE encoded observation instead of raw pixels if needed.
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        seed: int,
        add_to_port: int,
        simulator_scene: SimulatorScene,
        sim_mul: int = 1,
        headless: bool = False,
        exe_path: str = None,
        road_test_generator: RoadTestGenerator = None,
        collect_trace: bool = False,
        max_steps: int = None,
    ):

        self.exe_path = exe_path
        self.logger = GlobalLog("DonkeyGymEnv")
        self.test_generator = road_test_generator

        # TCP port for communicating with simulation
        if add_to_port == -1:
            port = int(os.environ.get("DONKEY_SIM_PORT", 9091))
            socket_local_address = int(
                os.environ.get("BASE_SOCKET_LOCAL_ADDRESS", 52804)
            )
        else:
            port = BASE_PORT + add_to_port
            socket_local_address = BASE_SOCKET_LOCAL_ADDRESS + port

        self.logger.debug("Simulator port: {}".format(port))

        self.unity_process = None
        if self.exe_path is not None:
            self.logger.info("Starting DonkeyGym env")
            assert os.path.exists(self.exe_path), "Path {} does not exist".format(
                self.exe_path
            )
            # Start Unity simulation subprocess if needed
            self.unity_process = UnityProcess(sim_name=DONKEY_SIM_NAME)
            self.unity_process.start(
                sim_path=self.exe_path, sim_mul=sim_mul, headless=headless, port=port
            )

        # start simulation com
        self.viewer = DonkeyUnitySimController(
            socket_local_address=socket_local_address,
            port=port,
            seed=seed,
            road_test_generator=road_test_generator,
            simulator_scene=simulator_scene,
            collect_trace=collect_trace,
            max_steps=max_steps,
        )

        # steering + throttle, action space must be symmetric
        self.action_space = spaces.Box(
            low=np.array([-MAX_STEERING, -1]),
            high=np.array([MAX_STEERING, 1]),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=0, high=255, shape=INPUT_DIM, dtype=np.uint8
        )
        self.seed(seed)
        # wait until loaded
        self.viewer.wait_until_loaded()

    def close_connection(self):
        return self.viewer.close_connection()

    def exit_scene(self):
        self.viewer.handler.send_exit_scene()

    def stop_simulation(self):
        self.viewer.handler.send_pause_simulation()

    def restart_simulation(self):
        self.viewer.handler.send_restart_simulation()

    def step(self, action: np.ndarray) -> ObserveData:
        """
        :param action: (np.ndarray)
        :return: (np.ndarray, float, bool, dict)
        """
        # action[0] is the steering angle
        # action[1] is the throttle
        self.viewer.take_action(action)
        observation, r, done, info = self.observe()

        return observation, r, done, info

    def reset(
        self, skip_generation: bool = False, individual: Individual = None
    ) -> np.ndarray:

        self.viewer.reset(skip_generation=skip_generation, individual=individual)
        observation, _, done, info = self.observe()

        return observation

    def render(self, mode="human"):
        """
        :param mode: (str)
        """
        if mode == "rgb_array":
            return self.viewer.handler.original_image
        return None

    def observe(self) -> ObserveData:
        """
        Encode the observation using VAE if needed.

        :return: (np.ndarray, float, bool, dict)
        """
        observation, _, done, info = self.viewer.observe()
        return observation, _, done, info

    def close(self):
        if self.unity_process is not None:
            self.unity_process.quit()
        self.viewer.quit()

    def pause_simulation(self):
        self.viewer.handler.send_pause_simulation()

    def restart_simulation(self):
        self.viewer.handler.send_restart_simulation()

    def seed(self, seed=None):
        self.viewer.seed(seed)
