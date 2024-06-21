from typing import List, Callable

import numpy as np
from shapely.geometry import Point

from config import DONKEY_SIM_NAME
from envs.donkey.donkey_env_utils import make_simulator_scene
from envs.donkey.scenes.simulator_scenes import SimulatorScene, GENERATED_TRACK_NAME
from self_driving.state import State
from self_driving.state_utils import get_state_condition_function
from test_generators.state_test_generator import SeedStateTestGenerator
from utils.randomness import set_random_seed


class PerformanceSeedStateTestGenerator(SeedStateTestGenerator):

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
        self.indices = []
        self.chosen_indices = set()
        self.probabilities = []

    def generate(self) -> State:
        assert self.reference_trace is not None, "Reference trace cannot be None"
        if len(self.all_states) == 0:
            self.all_states = self.reference_trace.get_states(
                road=self.road,
                state_condition_fn=self.state_condition_fn,
                other_bounding_box=self.other_bounding_box,
                donkey_simulator_scene=self.simulator_scene,
            )
            self.indices = np.arange(
                start=0, stop=len(self.all_states), step=1, dtype=np.int32
            )
            performances = np.asarray(
                sorted([state.get_performance() for state in self.all_states])
            )
            unnormalized_probabilities = np.exp(
                performances - performances[0]
            )  # to avoid numerical problem
            self.probabilities = unnormalized_probabilities / np.sum(
                unnormalized_probabilities
            )

        state_index = np.random.choice(a=self.indices, size=1, p=self.probabilities)[0]
        while state_index in self.chosen_indices:
            state_index = np.random.choice(
                a=self.indices, size=1, p=self.probabilities
            )[0]

        self.chosen_indices.add(state_index)
        if len(self.chosen_indices) == len(self.all_states):
            # reset indices
            self.chosen_indices.clear()

        return self.all_states[state_index]


if __name__ == "__main__":

    set_random_seed(seed=1)

    env_name = DONKEY_SIM_NAME

    performance_state_test_generator = PerformanceSeedStateTestGenerator(
        env_name=env_name,
        road_points=[],
        control_points=[],
        road_width=1,
        constant_road=True,
        state_condition_fn=get_state_condition_function(
            env_name=env_name, constant_road=True
        ),
        path_to_csv_file="../logs/donkey/generated_track/reference_trace_0.csv",
        simulator_scene=make_simulator_scene(
            scene_name=GENERATED_TRACK_NAME, track_num=0
        ),
    )
    for i in range(100):
        print("Sampled state: {}".format(performance_state_test_generator.generate()))
