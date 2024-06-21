from typing import List, Callable

import numpy as np
from shapely.geometry import Point

from config import DONKEY_SIM_NAME
from envs.donkey.scenes.simulator_scenes import SimulatorScene
from self_driving.state import State
from self_driving.state_utils import get_state_condition_function
from test_generators.state_test_generator import SeedStateTestGenerator
from utils.randomness import set_random_seed


class RandomSeedStateTestGenerator(SeedStateTestGenerator):

    def __init__(
        self,
        env_name: str,
        road_points: List[Point],
        control_points: List[Point],
        road_width: int,
        constant_road: bool,
        state_condition_fn: Callable[[int], bool] = None,
        path_to_csv_file: str = None,
        simulator_scene: SimulatorScene = None,
    ):
        super().__init__(
            env_name=env_name,
            road_points=road_points,
            control_points=control_points,
            road_width=road_width,
            constant_road=constant_road,
            state_condition_fn=state_condition_fn,
            path_to_csv_file=path_to_csv_file,
            simulator_scene=simulator_scene,
        )

        self.all_states = []

    def generate(self) -> State:
        assert self.reference_trace is not None, "Reference trace cannot be None"
        if len(self.all_states) == 0:
            self.all_states = self.reference_trace.get_states(
                road=self.road,
                state_condition_fn=self.state_condition_fn,
                other_bounding_box=self.other_bounding_box,
                donkey_simulator_scene=self.simulator_scene,
            )
        state = np.random.choice(a=self.all_states, size=1)[0]
        return state


if __name__ == "__main__":

    set_random_seed(seed=1)

    env_name = DONKEY_SIM_NAME

    random_state_test_generator = RandomSeedStateTestGenerator(
        env_name=env_name,
        road_points=[],
        control_points=[],
        road_width=1,
        constant_road=True,
        state_condition_fn=get_state_condition_function(
            env_name=env_name, constant_road=True
        ),
        path_to_csv_file="../logs/donkey/sandbox_lab/reference_trace.csv",
    )
    print(
        "After mutation: {}".format(
            random_state_test_generator.generate().get_representation()
        )
    )
