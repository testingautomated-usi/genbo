from typing import List, Callable, Dict

from shapely.geometry import Point

from envs.donkey.scenes.simulator_scenes import SimulatorScene
from self_driving.state import State
from self_driving.state_utils import get_state
from test_generators.state_test_generator import SeedStateTestGenerator


class ReplicateStateTestGenerator(SeedStateTestGenerator):

    def __init__(
        self,
        env_name: str,
        road_points: List[Point],
        control_points: List[Point],
        road_width: int,
        constant_road: bool,
        state_dict: Dict,
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

        self.state_dict = state_dict

    def generate(self) -> State:

        state = get_state(
            road=self.road,
            env_name=self.env_name,
            bounding_box=self.bounding_box,
            donkey_simulator_scene=self.simulator_scene,
        )

        current_waypoint = self.road.get_closest_control_point_index(
            point=Point(self.state_dict["pos_x"], 0.0, self.state_dict["pos_z"])
        )
        if (
            self.state_condition_fn is not None
            and self.state_condition_fn(current_waypoint)
            and self.other_bounding_box is not None
        ):
            state.set_bounding_box(bounding_box=self.other_bounding_box)
        state.update_state(**self.state_dict)

        return state
