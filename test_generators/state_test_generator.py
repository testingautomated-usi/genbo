from abc import ABC, abstractmethod
from typing import List, Callable

from shapely.geometry import Point

from envs.donkey.scenes.simulator_scenes import SimulatorScene
from self_driving.bounding_box_utils import get_bounding_box, get_other_bounding_box
from self_driving.road_utils import get_road
from self_driving.state import State
from self_driving.state_utils import ReferenceTrace


class SeedStateTestGenerator(ABC):

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
        self.state_to_generate = None
        self.env_name = env_name

        self.road = get_road(
            simulator_name=env_name,
            road_points=road_points,
            control_points=control_points,
            road_width=road_width,
            constant_road=constant_road,
            simulator_scene=simulator_scene,
        )
        self.simulator_scene = simulator_scene
        self.bounding_box = get_bounding_box(
            env_name=env_name,
            waypoints=self.road.get_waypoints(),
            road_width=self.road.road_width,
            donkey_simulator_scene=simulator_scene,
        )
        self.other_bounding_box = get_other_bounding_box(
            env_name=env_name,
            waypoints=self.road.get_waypoints(),
            road_width=self.road.road_width,
            donkey_simulator_scene=simulator_scene,
        )

        self.reference_trace = None
        if path_to_csv_file is not None:
            self.reference_trace = ReferenceTrace(
                path_to_csv_file=path_to_csv_file,
                env_name=env_name,
                bounding_box=self.bounding_box,
            )

        self.state_condition_fn = state_condition_fn

    @abstractmethod
    def generate(self) -> State:
        raise NotImplemented("Not implemented")

    def set_state_to_generate(self, state: State) -> None:
        self.state_to_generate = state
