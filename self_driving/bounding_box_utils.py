from typing import List, Optional

from config import SIMULATOR_NAMES, DONKEY_SIM_NAME, MOCK_SIM_NAME
from envs.donkey.scenes.simulator_scenes import SimulatorScene
from self_driving.bounding_box import BoundingBox
from self_driving.donkey_bounding_box import DonkeyBoundingBox
from self_driving.waypoint import Waypoint


def get_bounding_box(
    env_name: str,
    waypoints: List[Waypoint],
    road_width: float,
    donkey_simulator_scene: SimulatorScene = None,
) -> BoundingBox:
    assert (
        env_name in SIMULATOR_NAMES
    ), "Env name {} not supported. Choose between: {}".format(env_name, SIMULATOR_NAMES)

    if env_name == DONKEY_SIM_NAME or env_name == MOCK_SIM_NAME:
        return DonkeyBoundingBox(
            waypoints=waypoints,
            road_width=road_width,
            simulator_scene=donkey_simulator_scene,
        )

    raise RuntimeError("Unknown env name: {}".format(env_name))


def get_other_bounding_box(
    env_name: str,
    waypoints: List[Waypoint],
    road_width: float,
    donkey_simulator_scene: SimulatorScene = None,
) -> Optional[BoundingBox]:
    assert (
        env_name in SIMULATOR_NAMES
    ), "Env name {} not supported. Choose between: {}".format(env_name, SIMULATOR_NAMES)
    bounding_box = get_bounding_box(
        env_name=env_name,
        waypoints=waypoints,
        road_width=road_width,
        donkey_simulator_scene=donkey_simulator_scene,
    )

    if env_name == DONKEY_SIM_NAME or env_name == MOCK_SIM_NAME:
        bounding_box.road_width += 3
        return bounding_box

    return None
