import math
from typing import Callable, Tuple

import numpy as np
from shapely.geometry import Point

from config import SIMULATOR_NAMES, DONKEY_SIM_NAME, MOCK_SIM_NAME
from envs.donkey.config import MAX_SPEED_DONKEY


def get_velocity_checker(env_name: str) -> Callable[[Point], bool]:
    assert (
        env_name in SIMULATOR_NAMES
    ), "Env name {} not supported. Choose between: {}".format(env_name, SIMULATOR_NAMES)

    if env_name == DONKEY_SIM_NAME or env_name == MOCK_SIM_NAME:
        max_speed = MAX_SPEED_DONKEY
    else:
        raise RuntimeError("Unknown env name: {}".format(env_name))

    def _check_velocity(point: Point) -> bool:
        assert point.has_z, "Point has no z dimension"
        magnitude = get_velocity_magnitude(point=point, convert_to_kmh=True)
        if math.isclose(magnitude, max_speed, rel_tol=0.1):
            return True
        return magnitude < max_speed

    return _check_velocity


def get_velocity_magnitude(
    point: Point, convert_to_kmh: bool = False, consider_y_component: bool = False
) -> float:
    assert point.has_z, "Point has no z dimension"
    if consider_y_component:
        velocity_ms = np.sqrt(point.x * point.x + point.y * point.y + point.z * point.z)
    else:
        velocity_ms = np.sqrt(point.x * point.x + point.z * point.z)
    if convert_to_kmh:
        return round(velocity_ms * 3.6, 7)
    return round(velocity_ms, 7)


def get_velocity_components(
    env_name: str,
    velocity_magnitude: float,
    orientation_change: float = None,
    convert_to_ms: bool = False,
) -> Tuple[float, float, float]:
    assert (
        env_name in SIMULATOR_NAMES
    ), "Env name {} not supported. Choose between: {}".format(env_name, SIMULATOR_NAMES)

    if env_name == DONKEY_SIM_NAME or env_name == MOCK_SIM_NAME:
        if convert_to_ms:
            if orientation_change is not None:
                vel_x = (
                    velocity_magnitude / 3.6 * np.sin(np.radians(orientation_change))
                )
                vel_z = (
                    velocity_magnitude / 3.6 * np.cos(np.radians(orientation_change))
                )
            else:
                vel_x = 0.0
                vel_z = velocity_magnitude / 3.6
        else:
            if orientation_change is not None:
                vel_x = velocity_magnitude * np.sin(np.radians(orientation_change))
                vel_z = velocity_magnitude * np.cos(np.radians(orientation_change))
            else:
                vel_x = 0.0
                vel_z = velocity_magnitude
        vel_y = 0.0
        assert (
            vel_x is not None and vel_z is not None
        ), "Error in computing velocity components. Magnitude velocity: {}, Orientation change: {}".format(
            velocity_magnitude, orientation_change
        )
        return vel_x, vel_y, vel_z
    raise RuntimeError("Unknown env name: {}".format(env_name))


def get_velocity_orientation(env_name: str, velocity: Point) -> float:
    assert (
        env_name in SIMULATOR_NAMES
    ), "Env name {} not supported. Choose between: {}".format(env_name, SIMULATOR_NAMES)

    if env_name == DONKEY_SIM_NAME or env_name == MOCK_SIM_NAME:
        assert velocity.has_z, "Velocity has no z component"

        velocity_magnitude = get_velocity_magnitude(
            point=velocity, convert_to_kmh=False
        )

        if math.isclose(velocity_magnitude, 0.0, abs_tol=1e-1):
            return 0.0

        orientation = float(np.degrees(np.arcsin(velocity.x / velocity_magnitude)))
        assert (
            orientation is not None
        ), "Error in computing orientation. Velocity_x: {}, Magnitude velocity: {}".format(
            velocity.x, velocity_magnitude
        )

        return orientation

    raise RuntimeError("Unknown env name: {}".format(env_name))
