import copy
import math
import time
from typing import List, Union, cast, Tuple, Dict, Callable

import numpy as np
from shapely.geometry import Point

from config import DONKEY_SIM_NAME
from envs.donkey.config import (
    DONKEY_REFERENCE_TRACE_HEADER,
    EPS_VELOCITY_DONKEY,
    EPS_ORIENTATION_DONKEY,
    DONKEY_REFERENCE_TRACE_USED_KEYS,
    EPS_POSITION_DONKEY_SANDBOX,
    EPS_POSITION_DONKEY_GENERATED,
    MAX_CTE_ERROR_DONKEY,
    MAX_ORIENTATION_CHANGE_DONKEY,
    MAX_SPEED_DONKEY,
)
from envs.donkey.scenes.simulator_scenes import (
    SimulatorScene,
    SANDBOX_LAB_NAME,
    GENERATED_TRACK_NAME,
)
from global_log import GlobalLog
from self_driving.bounding_box import BoundingBox
from self_driving.orientation_utils import (
    Quaternion,
    quaternion_to_euler,
    get_angle_difference,
    intersect,
    get_higher_orientation,
)
from self_driving.position_utils import compute_cte, get_higher_cte
from self_driving.road import Road
from self_driving.state import State
from self_driving.velocity_utils import (
    get_velocity_components,
    get_velocity_orientation,
    get_velocity_magnitude,
)
from utils.stats import Stats


class DonkeyCarState(State):

    def __init__(
        self,
        road: Road,
        bounding_box: BoundingBox,
        velocity_checker: Callable[[Point], bool],
        orientation_checker: Callable[[Quaternion, Quaternion], bool],
        donkey_simulator_scene: SimulatorScene = None,
    ):
        super().__init__(
            road=road,
            bounding_box=bounding_box,
            velocity_checker=velocity_checker,
            orientation_checker=orientation_checker,
        )
        self.impl = dict()
        self.keys = DONKEY_REFERENCE_TRACE_HEADER.split(",")
        for key in self.keys:
            self.impl[key] = 0.0

        self.donkey_simulator_scene = donkey_simulator_scene

        self.relative_orientation = 0.0
        self.velocity_magnitude = 0.0
        self.cte = 0.0

        self.stats = Stats.get_instance()

        # self.logger = GlobalLog(logger_prefix="DonkeyCarState", verbose=False)
        self.logger = GlobalLog(logger_prefix="DonkeyCarState", verbose=True)

    def get_performance(self, normalize: bool = False) -> float:
        if normalize:
            return (
                compute_cte(
                    position=Point(
                        self.impl["pos_x"], self.impl["pos_y"], self.impl["pos_z"]
                    ),
                    road=self.road,
                )
                / MAX_CTE_ERROR_DONKEY
            )
        return compute_cte(
            position=Point(self.impl["pos_x"], self.impl["pos_y"], self.impl["pos_z"]),
            road=self.road,
        )

    def update_state(self, check_orientation: bool = True, **kwargs) -> None:
        for key, value in kwargs.items():
            assert key in self.keys, "Key {} not in keys {}".format(key, self.keys)
            self.impl[key] = value

        position = Point(self.impl["pos_x"], self.impl["pos_y"], self.impl["pos_z"])
        idx = self.road.get_closest_control_point_index(point=position)
        closest_waypoint = self.road.get_waypoints()[idx]
        orientation = Quaternion(
            v=np.asarray(
                [
                    self.impl["rot_x"],
                    self.impl["rot_y"],
                    self.impl["rot_z"],
                    self.impl["rot_w"],
                ]
            )
        )
        orientation_closest_waypoint = closest_waypoint.get_rotation()

        _, yaw_current, _ = quaternion_to_euler(
            rot_x=orientation.x,
            rot_y=orientation.y,
            rot_z=orientation.z,
            rot_w=orientation.w,
        )
        _, yaw_closest, _ = quaternion_to_euler(
            rot_x=orientation_closest_waypoint.x,
            rot_y=orientation_closest_waypoint.y,
            rot_z=orientation_closest_waypoint.z,
            rot_w=orientation_closest_waypoint.w,
        )

        self.relative_orientation = get_angle_difference(
            source=orientation, target=orientation_closest_waypoint
        )

        if check_orientation:
            assert abs(
                self.relative_orientation
            ) <= MAX_ORIENTATION_CHANGE_DONKEY or math.isclose(
                abs(self.relative_orientation),
                MAX_ORIENTATION_CHANGE_DONKEY,
                rel_tol=0.01,
            ), (
                f"(Update State) Error when setting orientation. Current orientation {yaw_current} "
                f"vs orientation closest waypoint {yaw_closest}. "
                f"Difference: {abs(self.relative_orientation)} > {MAX_ORIENTATION_CHANGE_DONKEY}"
            )

        self.velocity_magnitude = get_velocity_magnitude(
            point=Point(self.impl["vel_x"], self.impl["vel_y"], self.impl["vel_z"]),
            convert_to_kmh=True,
        )
        assert self.velocity_magnitude <= MAX_SPEED_DONKEY or math.isclose(
            self.velocity_magnitude, MAX_SPEED_DONKEY, rel_tol=0.01
        ), f"Error in setting velocity: {self.velocity_magnitude} > {MAX_SPEED_DONKEY}"

        self.cte = compute_cte(
            position=Point(self.impl["pos_x"], self.impl["pos_y"], self.impl["pos_z"]),
            road=self.road,
        )
        assert abs(self.cte) <= MAX_CTE_ERROR_DONKEY or math.isclose(
            abs(self.cte), MAX_CTE_ERROR_DONKEY, rel_tol=0.01
        ), f"Current cte {abs(self.cte)} is >= than the maximum {MAX_CTE_ERROR_DONKEY}"

    def parse(self, **kwargs) -> None:
        self.update_state(**kwargs)

    def export(self) -> Dict:
        result = dict()
        state_dict = copy.deepcopy(self.impl)
        result["implementation"] = state_dict
        if (
            get_velocity_magnitude(
                point=Point(self.impl["vel_x"], self.impl["vel_y"], self.impl["vel_z"])
            )
            > 0.0
        ):
            assert self.velocity_magnitude > 0.0, "Velocity magnitude cannot be <= 0.0"

        result["additional_metrics"] = {
            "relative_orientation": self.relative_orientation,
            "velocity_magnitude": self.velocity_magnitude,
            "distance_to_center": self.cte,
        }
        result["env_name"] = DONKEY_SIM_NAME
        result["road"] = self.road.export()
        return result

    def clone(self) -> "State":
        new_state = DonkeyCarState(
            road=self.road.clone(),
            bounding_box=self.bounding_box.clone(),
            velocity_checker=self.velocity_checker,
            orientation_checker=self.orientation_checker,
            donkey_simulator_scene=self.donkey_simulator_scene,
        )
        new_state.update_state(**self.impl)

        return new_state

    @staticmethod
    def get_eps_position(donkey_simulator_scene: SimulatorScene) -> float:
        if donkey_simulator_scene.get_scene_name() == SANDBOX_LAB_NAME:
            eps_position_donkey = EPS_POSITION_DONKEY_SANDBOX
        elif donkey_simulator_scene.get_scene_name() == GENERATED_TRACK_NAME:
            eps_position_donkey = EPS_POSITION_DONKEY_GENERATED
        else:
            raise RuntimeError(
                "Unknown simulator scene: {}".format(
                    donkey_simulator_scene.get_scene_name()
                )
            )
        return eps_position_donkey

    def _mutate_position(
        self,
        state: "DonkeyCarState",
        other_state: "DonkeyCarState" = None,
        is_close_constraint: bool = False,
        bias: bool = False,
    ) -> bool:

        changed = True
        self.logger.debug(
            "Mutate position: ({}, {})".format(state.impl["pos_x"], state.impl["pos_z"])
        )

        eps_position_donkey = self.get_eps_position(
            donkey_simulator_scene=self.donkey_simulator_scene
        )

        if other_state is not None and is_close_constraint:

            pos_x = state.impl["pos_x"]
            pos_y = state.impl["pos_y"]
            pos_z = state.impl["pos_z"]
            a = other_state.impl["pos_x"]
            b = other_state.impl["pos_z"]
            c = eps_position_donkey

            k = np.sqrt(-(a**2) + 2 * a * pos_x + c**2 - pos_x**2)
            self.logger.debug(f"Range x: {[a - c, a + c]}")
            self.logger.debug(f"Range z: {[b - k, b + k]}")

            if np.random.random() < 0.5:
                self.logger.debug("Change pos_x")
                if bias:
                    current_cte = abs(self.cte)

                    # compute margins
                    new_pos_x = a - c + 0.05  # to avoid that k goes to zero
                    k = np.sqrt(-(a**2) + 2 * a * new_pos_x + c**2 - new_pos_x**2)
                    new_pos_z = b - k
                    cte_1 = abs(
                        compute_cte(
                            position=Point(new_pos_x, pos_y, new_pos_z), road=state.road
                        )
                    )
                    new_pos_z = k + b
                    cte_2 = abs(
                        compute_cte(
                            position=Point(new_pos_x, pos_y, new_pos_z), road=state.road
                        )
                    )

                    new_pos_x = a + c - 0.05  # to avoid that k goes to zero
                    k = np.sqrt(-(a**2) + 2 * a * new_pos_x + c**2 - new_pos_x**2)
                    new_pos_z = b - k
                    cte_3 = abs(
                        compute_cte(
                            position=Point(new_pos_x, pos_y, new_pos_z), road=state.road
                        )
                    )
                    new_pos_z = k + b
                    cte_4 = abs(
                        compute_cte(
                            position=Point(new_pos_x, pos_y, new_pos_z), road=state.road
                        )
                    )

                    ranges = [cte_1, cte_2, cte_3, cte_4]
                    self.logger.debug(f"Current cte {current_cte}, ranges: {ranges}")

                    if current_cte > max(ranges) or math.isclose(
                        current_cte, max(ranges), rel_tol=0.001
                    ):
                        self.logger.debug(
                            f"Not possible to increase the cte w.r.t. the current one {current_cte} "
                            f"as the cte values obtained by reaching the limits of x and z are smaller: {ranges}"
                        )
                        return False

                    max_range_idx = int(np.argmax(ranges))

                    new_pos_x, new_pos_z = get_higher_cte(
                        curr_pos_x=pos_x,
                        curr_pos_y=pos_y,
                        curr_pos_z=pos_z,
                        curr_cte=current_cte,
                        val_a=a,
                        val_b=b,
                        val_c=c,
                        index=max_range_idx,
                        change_x=True,
                        road=state.road,
                    )

                    if new_pos_x == -1.0 and new_pos_z == -1.0:
                        self.logger.debug(
                            f"Not possible to increase the cte from its current one {abs(current_cte)}"
                        )
                        return False

                    new_cte = abs(
                        compute_cte(
                            position=Point(new_pos_x, pos_y, new_pos_z), road=state.road
                        )
                    )
                    self.logger.debug(
                        f"Previous cte: {abs(current_cte)} < new cte: {abs(new_cte)}"
                    )
                    assert (
                        new_cte > current_cte
                    ), f"New cte {abs(new_cte)} is <= current cte {abs(current_cte)}"
                else:
                    # https://www.wolframalpha.com/input?i=sqrt%28%28x+-+a%29+%5E+2+%2B+%28z+-+b%29+%5E+2%29+%3C+c
                    new_pos_x = np.random.uniform(low=a - c, high=a + c)
                    k = np.sqrt(-(a**2) + 2 * a * new_pos_x + c**2 - new_pos_x**2)
                    new_pos_z = np.random.uniform(low=b - k, high=k + b)
            else:
                self.logger.debug("Change pos_z")
                if bias:
                    current_cte = abs(self.cte)

                    # compute margins
                    new_pos_z = b - c + 0.05  # to avoid that k goes to zero
                    k = np.sqrt(-(b**2) + 2 * b * new_pos_z + c**2 - new_pos_z**2)
                    new_pos_x = a - k
                    cte_1 = abs(
                        compute_cte(
                            position=Point(new_pos_x, pos_y, new_pos_z), road=state.road
                        )
                    )

                    new_pos_z = b - c + 0.05  # to avoid that k goes to zero
                    k = np.sqrt(-(b**2) + 2 * b * new_pos_z + c**2 - new_pos_z**2)
                    new_pos_x = k + a
                    cte_2 = abs(
                        compute_cte(
                            position=Point(new_pos_x, pos_y, new_pos_z), road=state.road
                        )
                    )

                    new_pos_z = b + c - 0.05  # to avoid that k goes to zero
                    k = np.sqrt(-(b**2) + 2 * b * new_pos_z + c**2 - new_pos_z**2)
                    new_pos_x = a - k
                    cte_3 = abs(
                        compute_cte(
                            position=Point(new_pos_x, pos_y, new_pos_z), road=state.road
                        )
                    )

                    new_pos_z = b + c - 0.05  # to avoid that k goes to zero
                    k = np.sqrt(-(b**2) + 2 * b * new_pos_z + c**2 - new_pos_z**2)
                    new_pos_x = k + a
                    cte_4 = abs(
                        compute_cte(
                            position=Point(new_pos_x, pos_y, new_pos_z), road=state.road
                        )
                    )

                    ranges = [cte_1, cte_2, cte_3, cte_4]
                    self.logger.debug(f"Current cte {current_cte}, ranges: {ranges}")

                    if current_cte > max(ranges) or math.isclose(
                        current_cte, max(ranges), rel_tol=0.001
                    ):
                        self.logger.debug(
                            f"Not possible to increase the cte w.r.t. the current one {current_cte} "
                            f"as the cte values obtained by reaching the limits of x and z are smaller: {ranges}"
                        )
                        return False

                    max_range_idx = int(np.argmax(ranges))

                    new_pos_x, new_pos_z = get_higher_cte(
                        curr_pos_x=pos_x,
                        curr_pos_y=pos_y,
                        curr_pos_z=pos_z,
                        curr_cte=current_cte,
                        val_a=a,
                        val_b=b,
                        val_c=c,
                        index=max_range_idx,
                        change_x=False,
                        road=state.road,
                    )

                    if new_pos_x == -1.0 and new_pos_z == -1.0:
                        self.logger.debug(
                            f"Not possible to increase the cte from its current one {abs(current_cte)}"
                        )
                        return False

                    new_cte = abs(
                        compute_cte(
                            position=Point(new_pos_x, pos_y, new_pos_z), road=state.road
                        )
                    )
                    self.logger.debug(
                        f"Previous cte: {current_cte} < new cte: {new_cte}"
                    )
                    assert (
                        new_cte > current_cte
                    ), f"New cte {new_cte} is <= current cte {current_cte}"
                else:
                    # https://www.wolframalpha.com/input?i=sqrt%28%28x+-+b%29+%5E+2+%2B+%28z+-+a%29+%5E+2%29+%3C+c
                    new_pos_z = np.random.uniform(low=b - c, high=b + c)
                    k = np.sqrt(-(b**2) + 2 * b * new_pos_z + c**2 - new_pos_z**2)
                    new_pos_x = np.random.uniform(low=a - k, high=k + a)

            self.logger.debug(f"NEW_POS_X: {new_pos_x}, NEW_POS_Z: {new_pos_z}")

            distance = np.sqrt((new_pos_x - a) ** 2 + (new_pos_z - b) ** 2)
            assert distance < eps_position_donkey or math.isclose(
                distance, eps_position_donkey, rel_tol=0.01
            ), (
                f"Something went wrong when changing position components. "
                f"Distance {distance} >= epsilon {eps_position_donkey}"
            )

            pos_x = new_pos_x
            pos_z = new_pos_z
            self.logger.debug("POS_X after mutation: {}".format(new_pos_x))
            self.logger.debug("POS_Z after mutation: {}".format(new_pos_z))

            distance = np.sqrt(
                (pos_x - other_state.impl["pos_x"]) ** 2
                + (pos_z - other_state.impl["pos_z"]) ** 2
            )
            assert distance < eps_position_donkey or math.isclose(
                distance, eps_position_donkey, rel_tol=0.01
            ), (
                f"Something went wrong when changing position components. "
                f"Distance {distance} >= epsilon {eps_position_donkey}"
            )

            state.impl["pos_x"] = pos_x
            state.impl["pos_z"] = pos_z

        else:
            raise RuntimeError("Provide 'other_state' and 'is_close_constraint' = True")

        self.logger.debug(
            "Mutate position: ({}, {}), {}".format(
                state.impl["pos_x"], state.impl["pos_z"], changed
            )
        )
        return changed

    def _mutate_velocity(
        self,
        state: "DonkeyCarState",
        other_state: "DonkeyCarState" = None,
        is_close_constraint: bool = False,
        orientation_change: float = None,
        bias: bool = False,
    ) -> Tuple[bool, float]:
        magnitude = get_velocity_magnitude(
            point=Point(state.impl["vel_x"], state.impl["vel_y"], state.impl["vel_z"]),
            convert_to_kmh=True,
        )
        self.logger.debug("Mutate velocity: {}".format(magnitude))

        if other_state is not None and is_close_constraint:
            velocity_magnitude_other = get_velocity_magnitude(
                point=Point(
                    other_state.impl["vel_x"],
                    other_state.impl["vel_y"],
                    other_state.impl["vel_z"],
                ),
                convert_to_kmh=True,
            )
            if orientation_change is not None:

                a = velocity_magnitude_other
                b = EPS_VELOCITY_DONKEY
                c = MAX_SPEED_DONKEY

                x = [max(a - b, -c), min(a + b, c)]
                if bias:
                    current_magnitude = magnitude
                    max_magnitude = min(max(abs(x[0]), abs(x[-1])), MAX_SPEED_DONKEY)
                    if max_magnitude <= current_magnitude:
                        self.logger.debug(
                            f"Not possible to increase the magnitude w.r.t. the current one "
                            f"{current_magnitude} as the max velocity magnitude is smaller "
                            f"{x}, {max_magnitude}"
                        )
                        return False, magnitude

                    candidates_magnitude = abs(
                        np.random.uniform(
                            low=current_magnitude + 0.01, high=max_magnitude, size=50
                        )
                    )
                    new_magnitude = np.random.choice(
                        [
                            candidate_magnitude
                            for candidate_magnitude in candidates_magnitude
                            if abs(candidate_magnitude - current_magnitude)
                            <= EPS_VELOCITY_DONKEY
                        ]
                    )
                    self.logger.debug(
                        f"Previous magnitude: {current_magnitude} < new magnitude: {new_magnitude}"
                    )
                else:
                    new_magnitude = abs(np.random.uniform(low=x[0], high=x[-1]))

                assert abs(
                    new_magnitude - velocity_magnitude_other
                ) < EPS_VELOCITY_DONKEY or math.isclose(
                    abs(new_magnitude - velocity_magnitude_other),
                    EPS_VELOCITY_DONKEY,
                    rel_tol=0.01,
                ), "The new magnitude {} is not close to the magnitude of the other state {}. Diff {} > {}".format(
                    new_magnitude,
                    velocity_magnitude_other,
                    abs(new_magnitude - velocity_magnitude_other),
                    EPS_VELOCITY_DONKEY,
                )

                vel_x, vel_y, vel_z = get_velocity_components(
                    env_name=DONKEY_SIM_NAME,
                    velocity_magnitude=new_magnitude,
                    orientation_change=orientation_change,
                    convert_to_ms=True,
                )
                assert self.velocity_checker(
                    Point(vel_x, vel_y, vel_z)
                ), "The new magnitude {} > max speed {}".format(new_magnitude, c)

                state.impl["vel_x"] = vel_x
                state.impl["vel_y"] = vel_y
                state.impl["vel_z"] = vel_z

            else:
                raise RuntimeError(
                    "Provide 'other_state', 'is_close_constraint' = True and 'orientation_change'"
                )

            magnitude = new_magnitude
            assert new_magnitude > 0.0, "Velocity magnitude must be > 0.0"
        else:
            raise RuntimeError("Provide 'other_state' and 'is_close_constraint' = True")

        self.logger.debug("Mutate velocity: {}, {}".format(magnitude, True))
        return True, magnitude

    def _mutate_orientation(
        self,
        state: "DonkeyCarState",
        other_state: "DonkeyCarState" = None,
        is_close_constraint: bool = False,
        position: Point = None,
        bias: bool = False,
    ) -> bool:

        pitch, yaw, roll = quaternion_to_euler(
            rot_x=state.impl["rot_x"],
            rot_y=state.impl["rot_y"],
            rot_z=state.impl["rot_z"],
            rot_w=state.impl["rot_w"],
        )

        self.logger.debug("Mutate orientation: {}".format(yaw))

        if other_state is not None and is_close_constraint:
            other_pitch, other_yaw, other_roll = quaternion_to_euler(
                rot_x=other_state.impl["rot_x"],
                rot_y=other_state.impl["rot_y"],
                rot_z=other_state.impl["rot_z"],
                rot_w=other_state.impl["rot_w"],
            )
            yaw_source = None
            if position is not None:
                # assumes position is valid
                idx = self.road.get_closest_control_point_index(point=position)
                closest_waypoint = self.road.get_waypoints()[idx]
                self.logger.debug(
                    "Closest waypoint position: ({}, {})".format(
                        closest_waypoint.position.x, closest_waypoint.position.z
                    )
                )
                _, yaw_source, _ = closest_waypoint.get_rotation().get_euler()
                self.logger.debug("Orientation closest waypoint: {}".format(yaw_source))

            if yaw_source is not None:

                a = other_yaw
                b = EPS_ORIENTATION_DONKEY
                c = yaw_source
                d = MAX_ORIENTATION_CHANGE_DONKEY

                # by using modules there should not be negatives
                x = [(a - b) % 360, (a + b) % 360]
                y = [(c - d) % 360, (c + d) % 360]

                intersection = [max(x[0], y[0]), min(x[-1], y[-1])]
                self.logger.debug(
                    "x: {}, y: {}, range orientation: {}".format(x, y, intersection)
                )
                if intersection[0] > intersection[-1]:

                    # https://stackoverflow.com/questions/11775473/check-if-two-segments-on-the-same-circle-overlap-intersect
                    # https://creativecommons.org/licenses/by-sa/3.0/, until line 481

                    b1_norm = y[0]
                    a1_norm = x[0]
                    a2_norm = x[-1]
                    b2_norm = y[-1]

                    c1 = intersect(angle=b1_norm, min_angle=a1_norm, max_angle=a2_norm)
                    c2 = intersect(angle=b2_norm, min_angle=a1_norm, max_angle=a2_norm)
                    c3 = intersect(angle=a1_norm, min_angle=b1_norm, max_angle=b2_norm)
                    c4 = intersect(angle=a2_norm, min_angle=b1_norm, max_angle=b2_norm)

                    overlap = c1 or c2 or c3 or c4

                    if overlap:
                        # c1, c2 or c1, c3 or c1, c4
                        # c2, c3 or c2, c4
                        # c3, c4
                        if c1:
                            low = b1_norm
                            if c2:
                                high = b2_norm
                            elif c3:
                                high = a1_norm
                            elif c4:
                                high = a2_norm
                            else:
                                raise RuntimeError(
                                    "At least two conditions must be true. Found c1 to be True"
                                )
                        elif c2:
                            low = b2_norm
                            if c3:
                                high = a1_norm
                            elif c4:
                                high = a2_norm
                            else:
                                raise RuntimeError(
                                    "At least two conditions must be true. Found c2 to be True"
                                )
                        elif c3:
                            # low = b2_norm
                            low = a1_norm
                            if c4:
                                high = a2_norm
                            else:
                                raise RuntimeError(
                                    "At least two conditions must be true. Found c2 to be True"
                                )
                        else:
                            raise RuntimeError(
                                "At least two conditions must be true. Found none to be True"
                            )

                        if low > high:
                            if low > 270 and high < 90:
                                high = low + high
                            else:
                                tmp = high
                                high = low
                                low = tmp
                        elif low < 90 and high > 270:
                            low, high = intersection[0], (
                                intersection[0] + intersection[-1]
                            )

                        self.logger.debug("There is overlap: {}".format([low, high]))
                        if bias:
                            # recompute relative orientation because if the position changed the stored relative
                            # orientation is no longer valid
                            current_orientation = abs(
                                get_angle_difference(source=c, target=yaw)
                            )
                            if current_orientation != abs(self.relative_orientation):
                                self.logger.debug(
                                    f"Current orientation {current_orientation} != "
                                    f"stored relative orientation {abs(self.relative_orientation)}. "
                                    f"Position might have changed"
                                )

                            abs_angle_difference = abs(
                                get_angle_difference(source=low, target=high)
                            )
                            if abs_angle_difference <= current_orientation:
                                self.logger.debug(
                                    f"Not possible to increase the relative orientation w.r.t. the current one "
                                    f"{current_orientation} as the orientation range is smaller {(low, high)}, "
                                    f"{abs_angle_difference}"
                                )
                                return False

                            right_relative_orientation = abs(
                                get_angle_difference(source=c, target=high)
                            )
                            left_relative_orientation = abs(
                                get_angle_difference(source=c, target=low)
                            )
                            self.logger.debug(
                                f"Right orientation {right_relative_orientation}, Left orientation: {left_relative_orientation}"
                            )

                            if (
                                right_relative_orientation > current_orientation
                                and left_relative_orientation > current_orientation
                            ):
                                if (
                                    right_relative_orientation
                                    >= left_relative_orientation
                                ):
                                    # new_yaw = np.random.uniform(low=max(low, yaw), high=high)
                                    # new_yaw = high
                                    new_yaw = get_higher_orientation(
                                        current_relative_orientation=current_orientation,
                                        orientation_source=c,
                                        orientation_value=yaw,
                                        range_orientation_values=(low, high),
                                        direction="plus",
                                    )
                                else:
                                    # new_yaw = np.random.uniform(low=low, high=min(yaw, high))
                                    # new_yaw = low
                                    new_yaw = get_higher_orientation(
                                        current_relative_orientation=current_orientation,
                                        orientation_source=c,
                                        orientation_value=yaw,
                                        range_orientation_values=(low, high),
                                        direction="minus",
                                    )
                            elif (
                                right_relative_orientation
                                > current_orientation
                                >= left_relative_orientation
                            ):
                                # new_yaw = np.random.uniform(low=max(low, yaw), high=high)
                                # new_yaw = high
                                new_yaw = get_higher_orientation(
                                    current_relative_orientation=current_orientation,
                                    orientation_source=c,
                                    orientation_value=yaw,
                                    range_orientation_values=(low, high),
                                    direction="plus",
                                )
                            elif (
                                right_relative_orientation
                                <= current_orientation
                                < left_relative_orientation
                            ):
                                # new_yaw = np.random.uniform(low=low, high=min(yaw, high))
                                # new_yaw = low
                                new_yaw = get_higher_orientation(
                                    current_relative_orientation=current_orientation,
                                    orientation_source=c,
                                    orientation_value=yaw,
                                    range_orientation_values=(low, high),
                                    direction="minus",
                                )
                            else:
                                self.logger.debug(
                                    f"Not possible to get a higher relative orientation w.r.t. the current one "
                                    f"{current_orientation} given that the allowed range is smaller "
                                    f"{(left_relative_orientation, right_relative_orientation)}"
                                )
                                return False

                            if new_yaw == -1.0:
                                self.logger.debug(
                                    f"Not possible to increase the relative orientation from its current one {abs(current_orientation)}"
                                )
                                return False

                            self.logger.debug(f"New yaw: {new_yaw}")
                            new_orientation = abs(
                                get_angle_difference(source=c, target=new_yaw)
                            )

                            assert new_orientation > current_orientation, (
                                f"New relative orientation {abs(get_angle_difference(source=c, target=new_yaw))} "
                                f"<= current orientation {current_orientation}"
                            )

                            self.logger.debug(
                                f"Previous orientation: {current_orientation} < new orientation: {new_orientation}"
                            )
                        else:
                            new_yaw = np.random.uniform(low=low + 0.01, high=high)
                    else:
                        self.logger.debug(
                            "There is no overlap: {}, {}, {}".format(x, y, intersection)
                        )
                        return False
                else:
                    self.logger.debug(
                        "There is overlap: {}".format(
                            [intersection[0], intersection[-1]]
                        )
                    )
                    if bias:
                        # recompute relative orientation because if the position changed the stored relative
                        # orientation is no longer valid
                        current_orientation = abs(
                            get_angle_difference(source=c, target=yaw)
                        )
                        if current_orientation != abs(self.relative_orientation):
                            self.logger.debug(
                                f"Current orientation {current_orientation} != "
                                f"stored relative orientation {abs(self.relative_orientation)}. "
                                f"Position might have changed"
                            )

                        abs_angle_difference = abs(
                            get_angle_difference(
                                source=intersection[0], target=intersection[-1]
                            )
                        )
                        if abs_angle_difference <= current_orientation:
                            self.logger.debug(
                                f"Not possible to increase the relative orientation w.r.t. the current one "
                                f"{current_orientation} as the orientation range is smaller {(intersection[0], intersection[-1])}, "
                                f"{abs_angle_difference}"
                            )
                            return False

                        right_relative_orientation = abs(
                            get_angle_difference(source=c, target=intersection[-1])
                        )
                        left_relative_orientation = abs(
                            get_angle_difference(source=c, target=intersection[0])
                        )
                        self.logger.debug(
                            f"Right orientation {right_relative_orientation}, Left orientation: {left_relative_orientation}"
                        )

                        if (
                            right_relative_orientation > current_orientation
                            and left_relative_orientation > current_orientation
                        ):
                            if right_relative_orientation >= left_relative_orientation:
                                # new_yaw = np.random.uniform(low=max(intersection[0], yaw), high=intersection[-1])
                                # new_yaw = intersection[-1]
                                new_yaw = get_higher_orientation(
                                    current_relative_orientation=current_orientation,
                                    orientation_source=c,
                                    orientation_value=yaw,
                                    range_orientation_values=(
                                        intersection[0],
                                        intersection[-1],
                                    ),
                                    direction="plus",
                                )
                            else:
                                # new_yaw = np.random.uniform(low=intersection[0], high=min(yaw, intersection[-1]))
                                # new_yaw = intersection[0]
                                new_yaw = get_higher_orientation(
                                    current_relative_orientation=current_orientation,
                                    orientation_source=c,
                                    orientation_value=yaw,
                                    range_orientation_values=(
                                        intersection[0],
                                        intersection[-1],
                                    ),
                                    direction="minus",
                                )
                        elif (
                            right_relative_orientation
                            > current_orientation
                            >= left_relative_orientation
                        ):
                            # new_yaw = np.random.uniform(low=max(intersection[0], yaw), high=intersection[-1])
                            # new_yaw = intersection[-1]
                            new_yaw = get_higher_orientation(
                                current_relative_orientation=current_orientation,
                                orientation_source=c,
                                orientation_value=yaw,
                                range_orientation_values=(
                                    intersection[0],
                                    intersection[-1],
                                ),
                                direction="plus",
                            )
                        elif (
                            right_relative_orientation
                            <= current_orientation
                            < left_relative_orientation
                        ):
                            # new_yaw = np.random.uniform(low=intersection[0], high=min(yaw, intersection[-1]))
                            # new_yaw = intersection[0]
                            new_yaw = get_higher_orientation(
                                current_relative_orientation=current_orientation,
                                orientation_source=c,
                                orientation_value=yaw,
                                range_orientation_values=(
                                    intersection[0],
                                    intersection[-1],
                                ),
                                direction="minus",
                            )
                        else:
                            self.logger.debug(
                                f"Not possible to get a higher relative orientation w.r.t. the current one "
                                f"{current_orientation} given that the allowed range is smaller "
                                f"{(left_relative_orientation, right_relative_orientation)}"
                            )
                            return False

                        if new_yaw == -1.0:
                            self.logger.debug(
                                f"Not possible to increase the relative orientation from its current one {abs(current_orientation)}"
                            )
                            return False

                        self.logger.debug(f"New yaw: {new_yaw}")
                        new_orientation = abs(
                            get_angle_difference(source=c, target=new_yaw)
                        )

                        assert new_orientation > current_orientation, (
                            f"New relative orientation {abs(get_angle_difference(source=c, target=new_yaw))} "
                            f"<= current orientation {current_orientation}"
                        )

                        self.logger.debug(
                            f"Previous orientation: {current_orientation} < new orientation: {new_orientation}"
                        )
                    else:
                        new_yaw = np.random.uniform(
                            low=intersection[0] + 0.01, high=intersection[-1]
                        )

                new_yaw %= 360

                diff = abs(get_angle_difference(source=new_yaw, target=other_yaw))
                assert diff < EPS_ORIENTATION_DONKEY or math.isclose(
                    diff, EPS_ORIENTATION_DONKEY, rel_tol=0.01
                ), "The new orientation {} is not close to the orientation of the other state {}. Eps: {}".format(
                    new_yaw, other_yaw, EPS_ORIENTATION_DONKEY
                )

                orientation = Quaternion.from_euler(pitch=pitch, yaw=new_yaw, roll=roll)
                assert self.orientation_checker(
                    closest_waypoint.get_rotation(), orientation
                ), "The new orientation {} changes too much w.r.t. the orientation of the closest waypoint {}".format(
                    new_yaw, yaw_source
                )

            else:
                raise RuntimeError(
                    "Provide 'other_state', 'is_close_constraint' = True and 'yaw_source'"
                )

            assert new_yaw > 0.0, "Rotation angle must be > 0.0"
            yaw = new_yaw
        else:
            raise RuntimeError("Provide 'other_state' and 'is_close_constraint' = True")

        q = Quaternion.from_euler(pitch=pitch, yaw=yaw, roll=roll)
        state.impl["rot_x"] = q.x
        state.impl["rot_y"] = q.y
        state.impl["rot_z"] = q.z
        state.impl["rot_w"] = q.w

        self.logger.debug("Mutate orientation: {}".format(yaw))
        return True

    def is_close_in_position(self, other: "State") -> bool:
        other_state = cast(DonkeyCarState, other)
        eps_position = self.get_eps_position(
            donkey_simulator_scene=self.donkey_simulator_scene
        )
        distance = np.sqrt(
            (self.impl["pos_x"] - other_state.impl["pos_x"]) ** 2
            + (self.impl["pos_z"] - other_state.impl["pos_z"]) ** 2
        )
        return distance < eps_position or math.isclose(
            distance, eps_position, rel_tol=0.01
        )

    def is_close_in_velocity(self, other: "State") -> bool:
        other_state = cast(DonkeyCarState, other)
        # eps velocity is in km/h
        eps_velocity = EPS_VELOCITY_DONKEY
        velocity_magnitude = get_velocity_magnitude(
            point=Point(self.impl["vel_x"], self.impl["vel_y"], self.impl["vel_z"]),
            convert_to_kmh=True,
        )
        velocity_magnitude_other = get_velocity_magnitude(
            point=Point(
                other_state.impl["vel_x"],
                other_state.impl["vel_y"],
                other_state.impl["vel_z"],
            ),
            convert_to_kmh=True,
        )
        difference = abs(velocity_magnitude - velocity_magnitude_other)
        self.logger.debug(
            "Velocity: {}, Velocity other: {}. Abs difference: {}, Eps: {}".format(
                velocity_magnitude, velocity_magnitude_other, difference, eps_velocity
            )
        )
        return difference < eps_velocity or math.isclose(
            difference, eps_velocity, rel_tol=0.01
        )

    def is_close_in_orientation(self, other: "State") -> bool:
        other_state = cast(DonkeyCarState, other)
        eps_orientation = EPS_ORIENTATION_DONKEY

        _, yaw, _ = quaternion_to_euler(
            rot_x=self.impl["rot_x"],
            rot_y=self.impl["rot_y"],
            rot_z=self.impl["rot_z"],
            rot_w=self.impl["rot_w"],
        )

        _, yaw_other, _ = quaternion_to_euler(
            rot_x=other_state.impl["rot_x"],
            rot_y=other_state.impl["rot_y"],
            rot_z=other_state.impl["rot_z"],
            rot_w=other_state.impl["rot_w"],
        )

        difference = abs(
            Quaternion.compute_angle_difference(first_angle=yaw, second_angle=yaw_other)
        )
        self.logger.debug(
            "Yaw angle: {}, Yaw other angle: {}. Abs difference: {}, Eps: {}".format(
                yaw, yaw_other, difference, eps_orientation
            )
        )
        return difference < eps_orientation or math.isclose(
            difference, eps_orientation, rel_tol=0.01
        )

    def is_valid(self) -> bool:
        position = Point(self.impl["pos_x"], self.impl["pos_y"], self.impl["pos_z"])
        valid_position = self.bounding_box.is_in(point=position)

        orientation = Quaternion(
            v=np.asarray(
                [
                    self.impl["rot_x"],
                    self.impl["rot_y"],
                    self.impl["rot_z"],
                    self.impl["rot_w"],
                ]
            )
        )
        idx = self.road.get_closest_control_point_index(point=position)
        closest_waypoint = self.road.get_waypoints()[idx]
        orientation_closest_waypoint = closest_waypoint.get_rotation()
        valid_orientation = self.orientation_checker(
            orientation_closest_waypoint, orientation
        )

        velocity_magnitude = get_velocity_magnitude(
            point=Point(self.impl["vel_x"], self.impl["vel_y"], self.impl["vel_z"]),
            convert_to_kmh=True,
        )
        orientation_change = get_velocity_orientation(
            env_name=DONKEY_SIM_NAME,
            velocity=Point(self.impl["vel_x"], self.impl["vel_y"], self.impl["vel_z"]),
        )
        vel_x, vel_y, vel_z = get_velocity_components(
            env_name=DONKEY_SIM_NAME,
            velocity_magnitude=velocity_magnitude,
            orientation_change=orientation_change,
            convert_to_ms=True,
        )
        valid_velocity = self.velocity_checker(Point(vel_x, vel_y, vel_z))

        valid = valid_position and valid_orientation and valid_velocity
        if not valid:
            self.logger.debug(
                "State not valid! Position: {}, Orientation: {}, Velocity: {}".format(
                    valid_position, valid_orientation, valid_velocity
                )
            )

        return valid

    def is_close_to(self, other: "State") -> bool:

        is_close_position = self.is_close_in_position(other=other)
        is_close_velocity = self.is_close_in_velocity(other=other)
        is_close_orientation = self.is_close_in_orientation(other=other)

        self.logger.debug(
            "IS CLOSE: position {}, orientation {}, velocity {}".format(
                is_close_position, is_close_orientation, is_close_velocity
            )
        )

        return is_close_position and is_close_velocity and is_close_orientation

    def mutate(
        self,
        other_state: "State" = None,
        is_close_constraint: bool = False,
        bias: bool = False,
        previous_state: "State" = None,
    ) -> Tuple["State", bool]:

        if previous_state is not None and bias:
            # compute deltas w.r.t. the previous state (other_state is the new state)
            other_state = cast(DonkeyCarState, other_state)
            previous_state = cast(DonkeyCarState, previous_state)
            cte_delta = abs(other_state.cte) - abs(previous_state.cte)
            relative_orientation_delta = abs(other_state.relative_orientation) - abs(
                previous_state.relative_orientation
            )
            velocity_magnitude_delta = abs(other_state.velocity_magnitude) - abs(
                previous_state.velocity_magnitude
            )

            self.logger.debug(f"New other state: {other_state}")
            self.logger.debug(f"Previous other state: {previous_state}")
            self.logger.debug(f"Current state: {self}")
            self.logger.debug(f"Cte delta: {cte_delta}")
            self.logger.debug(
                f"Relative orientation delta: {relative_orientation_delta}"
            )
            self.logger.debug(f"Velocity magnitude delta: {velocity_magnitude_delta}")

            pos_x_delta = other_state.impl["pos_x"] - previous_state.impl["pos_x"]
            pos_z_delta = other_state.impl["pos_z"] - previous_state.impl["pos_z"]

            vel_x_delta = other_state.impl["vel_x"] - previous_state.impl["vel_x"]
            vel_z_delta = other_state.impl["vel_z"] - previous_state.impl["vel_z"]

            rotation_angle_delta = (
                other_state.impl["rotation_angle"]
                - previous_state.impl["rotation_angle"]
            )

            self.logger.debug(
                f"Deltas. Pos: {pos_x_delta}, {pos_z_delta}; Vel: {vel_x_delta}, {vel_z_delta}; "
                f"Rotation angle: {rotation_angle_delta}"
            )

            new_state = cast(DonkeyCarState, self.clone())

            new_state.impl["pos_x"] += pos_x_delta
            new_state.impl["pos_z"] += pos_z_delta

            new_state.impl["vel_x"] += vel_x_delta
            new_state.impl["vel_z"] += vel_z_delta

            new_state.impl["rotation_angle"] += rotation_angle_delta

            # FIXME: assuming pitch = roll = 0.0
            pitch, yaw, roll = 0.0, new_state.impl["rotation_angle"], 0.0
            q = Quaternion.from_euler(pitch=pitch, yaw=yaw, roll=roll)
            new_state.impl["rot_x"] = q.x
            new_state.impl["rot_y"] = q.y
            new_state.impl["rot_z"] = q.z
            new_state.impl["rot_w"] = q.w

            new_orientation = Quaternion(
                v=np.asarray(
                    [
                        new_state.impl["rot_x"],
                        new_state.impl["rot_y"],
                        new_state.impl["rot_z"],
                        new_state.impl["rot_w"],
                    ]
                )
            )

            angle_difference = abs(
                get_angle_difference(
                    source=new_orientation.get_euler()[1],
                    target=new_state.impl["rotation_angle"],
                )
            )
            # don't know why here abs_tol is needed instead of rel_tol
            assert math.isclose(angle_difference, 0.0, abs_tol=0.01), (
                f"New orientation {new_orientation.get_euler()[1]} != {new_state.impl['rotation_angle']} "
                f"rotation angle when mutating the state using deltas"
            )

            position = Point(
                new_state.impl["pos_x"],
                new_state.impl["pos_y"],
                new_state.impl["pos_z"],
            )
            valid_position = new_state.bounding_box.is_in(point=position)

            if not valid_position:
                self.logger.debug(
                    f"Position {position} not valid when setting it using deltas"
                )
                return self, False

            self.logger.debug("State after mutation: {}".format(new_state))

            idx = new_state.road.get_closest_control_point_index(point=position)
            orientation_closest_waypoint = new_state.road.get_waypoints()[
                idx
            ].get_rotation()

            new_state.relative_orientation = get_angle_difference(
                source=new_orientation, target=orientation_closest_waypoint
            )

            yaw_current = new_orientation.get_euler()[1]
            yaw_closest = orientation_closest_waypoint.get_euler()[1]

            if abs(new_state.relative_orientation) > MAX_ORIENTATION_CHANGE_DONKEY:
                self.logger.debu(
                    f"(Deltas) Error when setting orientation. Current orientation {yaw_current} vs "
                    f"orientation closest waypoint {yaw_closest}. "
                    f"Difference: {abs(new_state.relative_orientation)} > {MAX_ORIENTATION_CHANGE_DONKEY}"
                )
                return self, False

            new_state.velocity_magnitude = get_velocity_magnitude(
                point=Point(
                    new_state.impl["vel_x"],
                    new_state.impl["vel_y"],
                    new_state.impl["vel_z"],
                ),
                convert_to_kmh=True,
            )

            assert abs(
                new_state.velocity_magnitude
            ) <= MAX_SPEED_DONKEY or math.isclose(
                abs(new_state.velocity_magnitude), MAX_SPEED_DONKEY, rel_tol=0.01
            ), f"(Deltas) Error in setting velocity: {abs(new_state.velocity_magnitude)} > {MAX_SPEED_DONKEY}"

            new_state.cte = compute_cte(
                position=Point(
                    new_state.impl["pos_x"],
                    new_state.impl["pos_y"],
                    new_state.impl["pos_z"],
                ),
                road=new_state.road,
            )

            if abs(new_state.cte) > MAX_CTE_ERROR_DONKEY:
                self.logger.debug(
                    f"(Deltas) Current cte {abs(new_state.cte)} is >= than the maximum {MAX_CTE_ERROR_DONKEY}"
                )
                return self, False

            # FIXME Moving velocity all to z component
            vel_x, vel_y, vel_z = get_velocity_components(
                env_name=DONKEY_SIM_NAME,
                velocity_magnitude=new_state.velocity_magnitude,
                convert_to_ms=True,
            )
            new_state.impl["vel_x"] = vel_x
            new_state.impl["vel_y"] = vel_y
            new_state.impl["vel_z"] = vel_z

            return new_state, True

        start_time = time.perf_counter()

        self.logger.debug("State before mutation: {}".format(self.get_representation()))
        new_state = cast(DonkeyCarState, self.clone())

        changed = False
        changed_position, changed_orientation, changed_velocity = False, False, False

        position = Point(self.impl["pos_x"], self.impl["pos_y"], self.impl["pos_z"])
        valid_position = True
        orientation_change = get_velocity_orientation(
            env_name=DONKEY_SIM_NAME,
            velocity=Point(self.impl["vel_x"], self.impl["vel_y"], self.impl["vel_z"]),
        )

        self.logger.debug(
            "Init mutation time: {:.2f}s".format(time.perf_counter() - start_time)
        )
        start_time_while = time.perf_counter()

        max_iterations = 10

        choice = np.random.choice([0, 1, 2], size=1)[0]

        while not changed and max_iterations > 0:

            random_position, random_orientation, random_velocity = False, False, False
            # on average only one out of the three mutations are applied

            if np.random.random() < 0.3 or choice == 0:
                self.stats.num_mutate_position += 1
                random_position = True
                start_time = time.perf_counter()
                changed_position = self._mutate_position(
                    state=new_state,
                    other_state=other_state,
                    is_close_constraint=is_close_constraint,
                    bias=bias,
                )
                if changed_position:
                    new_position = Point(
                        new_state.impl["pos_x"],
                        new_state.impl["pos_y"],
                        new_state.impl["pos_z"],
                    )
                    valid_position = new_state.bounding_box.is_in(point=new_position)
                    if valid_position:
                        position = new_position
                self.logger.debug(
                    "Change position time: {:.2f}s".format(
                        time.perf_counter() - start_time
                    )
                )
                choice = -1

            if np.random.random() < 0.3 or choice == 1:
                self.stats.num_mutate_orientation += 1
                random_orientation = True
                start_time = time.perf_counter()
                changed_orientation = self._mutate_orientation(
                    state=new_state,
                    other_state=other_state,
                    is_close_constraint=is_close_constraint,
                    position=position,
                    bias=bias,
                )
                if changed_orientation:
                    orientation = Quaternion(
                        v=np.asarray(
                            [
                                new_state.impl["rot_x"],
                                new_state.impl["rot_y"],
                                new_state.impl["rot_z"],
                                new_state.impl["rot_w"],
                            ]
                        )
                    )
                    idx = self.road.get_closest_control_point_index(point=position)
                    orientation_closest_waypoint = self.road.get_waypoints()[
                        idx
                    ].get_rotation()
                    orientation_change = get_angle_difference(
                        source=orientation_closest_waypoint, target=orientation
                    )
                    self.logger.debug(
                        "Change orientation time: {:.2f}s".format(
                            time.perf_counter() - start_time
                        )
                    )
                choice = -1

            if np.random.random() < 0.3 or choice == 2:
                self.stats.num_mutate_velocity += 1
                random_velocity = True
                start_time = time.perf_counter()
                changed_velocity, velocity_magnitude = self._mutate_velocity(
                    state=new_state,
                    other_state=other_state,
                    is_close_constraint=is_close_constraint,
                    orientation_change=orientation_change,
                    bias=bias,
                )
                self.logger.debug(
                    "Change velocity time: {:.2f}s".format(
                        time.perf_counter() - start_time
                    )
                )
                choice = -1

            if random_position and random_orientation:
                self.stats.num_mutate_position_orientation += 1

            if random_position and random_velocity:
                self.stats.num_mutate_position_velocity += 1

            if random_velocity and random_orientation:
                self.stats.num_mutate_velocity_orientation += 1

            if random_position and random_velocity and random_orientation:
                self.stats.num_mutate_all += 1

            changed = changed_position or changed_velocity or changed_orientation

            if random_position or random_orientation or random_velocity:
                max_iterations -= 1

        self.logger.debug(
            f".......... Num mutations: {self.stats.get_num_mutations()} .........."
        )

        if max_iterations == 0 and not changed:
            self.logger.debug(f"Not possible to mutate state: {self}")
            return self, False

        self.logger.debug(
            "While time: {:.2f}s".format(time.perf_counter() - start_time_while)
        )

        if valid_position and changed_position:
            self.logger.debug("Position valid but checking the orientation")
            orientation = Quaternion(
                v=np.asarray(
                    [
                        new_state.impl["rot_x"],
                        new_state.impl["rot_y"],
                        new_state.impl["rot_z"],
                        new_state.impl["rot_w"],
                    ]
                )
            )
            position = Point(
                new_state.impl["pos_x"],
                new_state.impl["pos_y"],
                new_state.impl["pos_z"],
            )
            idx = self.road.get_closest_control_point_index(point=position)
            orientation_closest_waypoint = self.road.get_waypoints()[idx].get_rotation()
            angle_difference = get_angle_difference(
                source=orientation, target=orientation_closest_waypoint
            )
            if abs(angle_difference) > MAX_ORIENTATION_CHANGE_DONKEY:
                self.logger.debug(
                    f"By changing the position the relative orientation {abs(angle_difference)} "
                    f"changes too much w.r.t. the current maximum {MAX_ORIENTATION_CHANGE_DONKEY}"
                )
                self.logger.debug(
                    "Invalidating position because of the resulting orientation w.r.t. the closest waypoint"
                )
                valid_position = False

            if bias and abs(angle_difference) < abs(self.relative_orientation):
                self.logger.debug(
                    f"By changing the position the relative orientation {abs(angle_difference)} "
                    f"turns out to be smaller than the current one {abs(self.relative_orientation)}. "
                    f"This is because the bias flag is true which constraints the mutations to be increasing."
                )
                valid_position = False

        start_time = time.perf_counter()
        if not valid_position:
            self.logger.debug("Position not valid")
            position = Point(self.impl["pos_x"], self.impl["pos_y"], self.impl["pos_z"])
            orientation = Quaternion(
                v=np.asarray(
                    [
                        self.impl["rot_x"],
                        self.impl["rot_y"],
                        self.impl["rot_z"],
                        self.impl["rot_w"],
                    ]
                )
            )
            idx = self.road.get_closest_control_point_index(point=position)
            orientation_closest_waypoint = self.road.get_waypoints()[idx].get_rotation()

            _, yaw_current, _ = quaternion_to_euler(
                rot_x=orientation.x,
                rot_y=orientation.y,
                rot_z=orientation.z,
                rot_w=orientation.w,
            )
            _, yaw_closest, _ = quaternion_to_euler(
                rot_x=orientation_closest_waypoint.x,
                rot_y=orientation_closest_waypoint.y,
                rot_z=orientation_closest_waypoint.z,
                rot_w=orientation_closest_waypoint.w,
            )

            relative_orientation = get_angle_difference(
                source=orientation, target=orientation_closest_waypoint
            )
            assert abs(
                relative_orientation
            ) <= MAX_ORIENTATION_CHANGE_DONKEY or math.isclose(
                abs(relative_orientation), MAX_ORIENTATION_CHANGE_DONKEY, rel_tol=0.01
            ), (
                f"(Invalid position) Error when setting orientation. Current orientation {yaw_current} "
                f"vs orientation closest waypoint {yaw_closest}. "
                f"Difference: {abs(relative_orientation)} > {MAX_ORIENTATION_CHANGE_DONKEY}"
            )

            assert (
                self.relative_orientation == relative_orientation
            ), f"Problem when computing the relative orientation. New {relative_orientation} != old {self.relative_orientation}"

            velocity_magnitude = get_velocity_magnitude(
                point=Point(self.impl["vel_x"], self.impl["vel_y"], self.impl["vel_z"]),
                convert_to_kmh=True,
            )
            assert velocity_magnitude <= MAX_SPEED_DONKEY or math.isclose(
                velocity_magnitude, MAX_SPEED_DONKEY, rel_tol=0.01
            ), f"Error in setting velocity: {velocity_magnitude} > {MAX_SPEED_DONKEY}"

            assert (
                self.velocity_magnitude == velocity_magnitude
            ), f"Problem when computing the velocity magnitude. New {velocity_magnitude} != old {self.velocity_magnitude}"

            cte = compute_cte(
                position=Point(
                    self.impl["pos_x"], self.impl["pos_y"], self.impl["pos_z"]
                ),
                road=self.road,
            )
            assert abs(cte) <= MAX_CTE_ERROR_DONKEY or math.isclose(
                abs(cte), MAX_CTE_ERROR_DONKEY, rel_tol=0.01
            ), f"Current cte {cte} is >= than the maximum {MAX_CTE_ERROR_DONKEY}"

            assert (
                self.cte == cte
            ), f"Problem when computing the cross track error. New {cte} != old {self.cte}"

            self.logger.debug(
                "Position not valid time: {:.2f}s".format(
                    time.perf_counter() - start_time
                )
            )
            return self, False

        start_time = time.perf_counter()
        self.logger.debug("Position valid (orientation checked)")
        new_orientation = Quaternion(
            v=np.asarray(
                [
                    new_state.impl["rot_x"],
                    new_state.impl["rot_y"],
                    new_state.impl["rot_z"],
                    new_state.impl["rot_w"],
                ]
            )
        )

        position = Point(
            new_state.impl["pos_x"], new_state.impl["pos_y"], new_state.impl["pos_z"]
        )
        new_state.impl["rotation_angle"] = new_orientation.get_euler()[1]
        idx = self.road.get_closest_control_point_index(point=position)
        orientation_closest_waypoint = self.road.get_waypoints()[idx].get_rotation()

        self.logger.debug(
            "Valid mutation! State after mutation: {}".format(
                new_state.get_representation()
            )
        )
        new_state.relative_orientation = get_angle_difference(
            source=new_orientation, target=orientation_closest_waypoint
        )

        _, yaw_current, _ = quaternion_to_euler(
            rot_x=new_orientation.x,
            rot_y=new_orientation.y,
            rot_z=new_orientation.z,
            rot_w=new_orientation.w,
        )
        _, yaw_closest, _ = quaternion_to_euler(
            rot_x=orientation_closest_waypoint.x,
            rot_y=orientation_closest_waypoint.y,
            rot_z=orientation_closest_waypoint.z,
            rot_w=orientation_closest_waypoint.w,
        )

        assert abs(
            new_state.relative_orientation
        ) <= MAX_ORIENTATION_CHANGE_DONKEY or math.isclose(
            abs(new_state.relative_orientation),
            MAX_ORIENTATION_CHANGE_DONKEY,
            rel_tol=0.01,
        ), (
            f"(Position valid) Error when setting orientation. Current orientation {yaw_current} "
            f"vs orientation closest waypoint {yaw_closest}. "
            f"Difference: {abs(new_state.relative_orientation)} > {MAX_ORIENTATION_CHANGE_DONKEY}"
        )

        assert abs(new_state.relative_orientation) >= abs(self.relative_orientation), (
            f"Error when computing the relative orientation after mutation. It should be >= than the old one: "
            f"found {abs(new_state.relative_orientation)} < {abs(self.relative_orientation)}"
        )

        new_state.velocity_magnitude = get_velocity_magnitude(
            point=Point(
                new_state.impl["vel_x"],
                new_state.impl["vel_y"],
                new_state.impl["vel_z"],
            ),
            convert_to_kmh=True,
        )

        assert abs(new_state.velocity_magnitude) <= MAX_SPEED_DONKEY or math.isclose(
            abs(new_state.velocity_magnitude), MAX_SPEED_DONKEY, rel_tol=0.01
        ), f"Error in setting velocity: {abs(new_state.velocity_magnitude)} > {MAX_SPEED_DONKEY}"

        assert abs(new_state.velocity_magnitude) >= abs(self.velocity_magnitude), (
            f"Error when computing the velocity magnitude after mutation. It should be >= than the old one: "
            f"found {abs(new_state.velocity_magnitude)} < {abs(self.velocity_magnitude)}"
        )

        new_state.cte = compute_cte(
            position=Point(
                new_state.impl["pos_x"],
                new_state.impl["pos_y"],
                new_state.impl["pos_z"],
            ),
            road=new_state.road,
        )
        assert abs(new_state.cte) <= MAX_CTE_ERROR_DONKEY or math.isclose(
            abs(new_state.cte), MAX_CTE_ERROR_DONKEY, rel_tol=0.01
        ), f"Current cte {abs(new_state.cte)} is >= than the maximum {MAX_CTE_ERROR_DONKEY}"

        assert abs(new_state.cte) >= abs(self.cte), (
            f"Error when computing the cross track error after mutation. It should be >= than the old one: "
            f"found {abs(new_state.cte)} < {abs(self.cte)}"
        )

        # FIXME Moving velocity all to z component
        vel_x, vel_y, vel_z = get_velocity_components(
            env_name=DONKEY_SIM_NAME,
            velocity_magnitude=new_state.velocity_magnitude,
            convert_to_ms=True,
        )
        new_state.impl["vel_x"] = vel_x
        new_state.impl["vel_y"] = vel_y
        new_state.impl["vel_z"] = vel_z

        self.logger.debug(
            "Position valid time: {:.2f}s".format(time.perf_counter() - start_time)
        )
        return new_state, True
    
    def to_dict(self) -> Dict[str, Union[str, float, int, bool]]:
        return self.impl

    def get_representation(self, csv: bool = False) -> str:
        if csv:
            return ",".join("{}".format(self.impl[key]) for key in self.keys)
        # only print the variables that are being used
        res = dict()
        for key in DONKEY_REFERENCE_TRACE_USED_KEYS.split(","):
            res[key] = self.impl[key]
        return res.__str__()

    def get_keys(self) -> List[str]:
        return self.keys

    def get_values(self) -> List[Union[str, float, int, bool]]:
        # only computing the real components of the velocity once they are needed
        assert "vel_x" in self.keys, "The key vel_x should be in keys: {}".format(
            self.keys
        )
        assert "vel_z" in self.keys, "The key vel_z should be in keys: {}".format(
            self.keys
        )
        assert (
            "rotation_angle" in self.keys
        ), "The key rotation_angle should be in keys: {}".format(self.keys)

        vel_x_relative = self.impl["vel_x"]
        vel_z_relative = self.impl["vel_z"]
        rotation_angle = np.radians(self.impl["rotation_angle"])

        # https://en.wikipedia.org/wiki/Rotation_of_axes
        # vel_x_absolute = vel_x_relative * np.cos(rotation_angle) - vel_z_relative * np.sin(rotation_angle)
        # vel_z_absolute = vel_x_relative * np.sin(rotation_angle) + vel_z_relative * np.cos(rotation_angle)

        # # those formulas above were used with a rotation_angle opposite of what is considered to be positive in Unity
        # # -> change sign of the rotation_angle with, using the properties of the trigonometric functions
        # # i.e., sin(-x) = -sin(x), cos(-x) = cos(x)
        vel_x_absolute = vel_x_relative * np.cos(
            rotation_angle
        ) + vel_z_relative * np.sin(rotation_angle)
        vel_z_absolute = -vel_x_relative * np.sin(
            rotation_angle
        ) + vel_z_relative * np.cos(rotation_angle)

        result = []
        for key in self.keys:
            if key == "vel_x":
                result.append(vel_x_absolute)
            elif key == "vel_z":
                result.append(vel_z_absolute)
            else:
                result.append(self.impl[key])

        new_dict = dict()
        for i, key in enumerate(self.keys):
            new_dict[key] = result[i]

        return result

    def set_bounding_box(self, bounding_box: BoundingBox) -> None:
        self.bounding_box = bounding_box

    def is_different(self, other: "DonkeyCarState") -> bool:
        equal_length = len(self.keys) == len(other.keys)
        if not equal_length:
            return True

        for i in range(len(self.keys)):
            assert (
                self.keys[i] == other.keys[i]
            ), f"The keys of the two state are different {self.keys[i]} != {other.keys[i]}"

            if self.impl[self.keys[i]] != other.impl[other.keys[i]]:
                return True

        return False

    def __eq__(self, other: "DonkeyCarState") -> bool:
        if isinstance(other, DonkeyCarState):
            return not self.is_different(other=other)

    def __hash__(self) -> int:
        return hash(self.get_representation())
