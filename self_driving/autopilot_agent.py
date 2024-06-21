import time
from typing import Dict

from config import DONKEY_SIM_NAME
from custom_types import GymEnv

import numpy as np

from envs.donkey.config import (
    KP_SANDBOX_DONKEY,
    KD_SANDBOX_DONKEY,
    KI_SANDBOX_DONKEY,
    KP_GENERATED_TRACK_DONKEY,
    KD_GENERATED_TRACK_DONKEY,
    KI_GENERATED_TRACK_DONKEY,
)
from envs.donkey.scenes.simulator_scenes import SANDBOX_LAB_NAME, GENERATED_TRACK_NAME
from self_driving.agent import Agent


class AutopilotAgent(Agent):

    def __init__(
        self,
        env: GymEnv,
        env_name: str,
        max_speed: int,
        min_speed: int,
        donkey_scene_name: str = None,
    ):
        super().__init__(env=env, env_name=env_name)

        self.previous_time = 0.0
        self.previous_cte = 0.0
        self.total_error = 0.0

        self.max_speed = max_speed
        self.min_speed = min_speed

        self.donkey_scene_name = donkey_scene_name

    def predict(self, obs: np.ndarray, state: Dict) -> np.ndarray:
        """
        The PID code is licenced under:

        Copyright (c) 2017, Tawn Kramer
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:
            * Redistributions of source code must retain the above copyright
              notice, this list of conditions and the following disclaimer.
            * Redistributions in binary form must reproduce the above copyright
              notice, this list of conditions and the following disclaimer in the
              documentation and/or other materials provided with the distribution.
            * Neither the name of the <organization> nor the
              names of its contributors may be used to endorse or promote products
              derived from this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
        ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
        WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
        DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
        (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
        LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
        ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
        (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
        SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
        """

        if self.env_name == DONKEY_SIM_NAME and len(state) > 0:

            delta = time.perf_counter() - self.previous_time

            diff_cte = (state["cte_pid"] - self.previous_cte) / delta
            self.previous_cte = state["cte_pid"]
            self.previous_time = time.perf_counter()

            self.total_error += state["cte_pid"]

            if self.env_name == DONKEY_SIM_NAME:
                if self.donkey_scene_name == SANDBOX_LAB_NAME:
                    steering = (
                        (-KP_SANDBOX_DONKEY * state["cte_pid"])
                        - (KD_SANDBOX_DONKEY * diff_cte)
                        - (KI_SANDBOX_DONKEY * self.total_error)
                    )
                elif self.donkey_scene_name == GENERATED_TRACK_NAME:
                    steering = (
                        (-KP_GENERATED_TRACK_DONKEY * state["cte_pid"])
                        - (KD_GENERATED_TRACK_DONKEY * diff_cte)
                        - (KI_GENERATED_TRACK_DONKEY * self.total_error)
                    )
                else:
                    raise RuntimeError(
                        "Unknown donkey scene name: {}".format(self.donkey_scene_name)
                    )
            else:
                raise RuntimeError("Unknown env name: {}".format(self.env_name))

            steering = np.clip(a=steering, a_min=-1.0, a_max=1.0)
            speed = state["speed"]
            if speed > self.max_speed:
                speed_limit = self.min_speed  # slow down
            else:
                speed_limit = self.max_speed

            throttle = np.clip(
                a=1.0 - steering**2 - (speed / speed_limit) ** 2, a_min=0.1, a_max=1.0
            )
            action = np.asarray([steering, throttle])

            return action

        return self.env.action_space.sample()
