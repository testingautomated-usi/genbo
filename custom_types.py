from typing import Tuple, Dict, Union

import gym
import numpy as np

from envs.vec_env.base_vec_env import VecEnv

ObserveData = Tuple[np.ndarray, float, bool, Dict]
GymEnv = Union[gym.Env, VecEnv]

Tuple4F = Tuple[float, float, float, float]
Tuple2F = Tuple[float, float]
