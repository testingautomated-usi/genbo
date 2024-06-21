from envs.donkey.scenes.simulator_scenes import SimulatorScene, GENERATED_TRACK_NAME
from self_driving.road import Road
from self_driving.road_utils import get_road, get_constant_road
from test_generators.road_test_generator import RoadTestGenerator


class ConstantRoadTestGenerator(RoadTestGenerator):

    def __init__(
        self, map_size: int, simulator_name: str, simulator_scene: SimulatorScene = None
    ):
        super().__init__(map_size=map_size)
        self.map_size = map_size
        self.simulator_name = simulator_name
        self.simulator_scene = simulator_scene

    def generate(self) -> Road:
        if self.simulator_scene.get_scene_name() == GENERATED_TRACK_NAME:
            return get_constant_road(
                simulator_name=self.simulator_name,
                simulator_scene=self.simulator_scene,
            )

        return get_road(
            simulator_name=self.simulator_name,
            road_points=[],
            control_points=[],
            road_width=0,
            constant_road=True,
        )

    def set_max_angle(self, max_angle: int) -> None:
        raise NotImplemented("Not implemented with constant test generator")
