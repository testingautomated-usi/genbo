from config import ROAD_TEST_GENERATOR_NAMES
from envs.donkey.donkey_env_utils import make_simulator_scene
from test_generators.constant_road_test_generator import ConstantRoadTestGenerator
from test_generators.road_test_generator import RoadTestGenerator


def make_road_test_generator(
    generator_name: str,
    map_size: int,
    simulator_name: str,
    donkey_scene_name: str = None,
    track_num: int = None,
) -> RoadTestGenerator:
    assert (
        generator_name in ROAD_TEST_GENERATOR_NAMES
    ), "Test generator {} not found. Choose between {}".format(
        generator_name, ROAD_TEST_GENERATOR_NAMES
    )
    if generator_name == "constant":
        return ConstantRoadTestGenerator(
            map_size=map_size,
            simulator_name=simulator_name,
            simulator_scene=make_simulator_scene(
                scene_name=donkey_scene_name, track_num=track_num
            ),
        )

    raise NotImplemented("Test generator {} not found".format(generator_name))
