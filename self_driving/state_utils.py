import os.path
import csv
import json
from typing import List, Dict, Callable, Tuple, Union

from config import (
    SIMULATOR_NAMES,
    DONKEY_SIM_NAME,
    REFERENCE_TRACE_CONSTANT_HEADER,
    REFERENCE_TRACE_CONSTANT_HEADER_TYPES,
    CURRENT_WAYPOINT_KEY,
    MOCK_SIM_NAME,
)
from envs.donkey.config import (
    DONKEY_REFERENCE_TRACE_HEADER,
    DONKEY_REFERENCE_TRACE_HEADER_TYPES,
)
from envs.donkey.donkey_env_utils import make_simulator_scene
from envs.donkey.scenes.simulator_scenes import SimulatorScene, GENERATED_TRACK_NAME
from global_log import GlobalLog
from self_driving.bounding_box import BoundingBox
from self_driving.bounding_box_utils import get_bounding_box
from self_driving.donkey_car_state import DonkeyCarState
from self_driving.orientation_utils import get_orientation_checker, Quaternion
from self_driving.road import Road
from self_driving.road_utils import get_road
from self_driving.state import State
from self_driving.velocity_utils import get_velocity_checker


def get_state(
    road: Road,
    env_name: str,
    bounding_box: BoundingBox,
    donkey_simulator_scene: SimulatorScene = None,
) -> State:
    assert (
        env_name in SIMULATOR_NAMES
    ), "Env name {} not supported. Choose between: {}".format(env_name, SIMULATOR_NAMES)

    if env_name == DONKEY_SIM_NAME or env_name == MOCK_SIM_NAME:
        return DonkeyCarState(
            road=road,
            bounding_box=bounding_box,
            velocity_checker=get_velocity_checker(env_name=env_name),
            orientation_checker=get_orientation_checker(env_name=env_name),
            donkey_simulator_scene=donkey_simulator_scene,
        )

    raise RuntimeError("Unknown env name: {}".format(env_name))


def get_state_condition_function(
    env_name: str, constant_road: bool
) -> Union[Callable[[int], bool], None]:
    assert (
        env_name in SIMULATOR_NAMES
    ), "Env name {} not supported. Choose between: {}".format(env_name, SIMULATOR_NAMES)

    if env_name == DONKEY_SIM_NAME and constant_road:

        def fn(current_waypoint: int) -> bool:
            # see is_game_over function in donkey_sim
            return (
                current_waypoint == 15
                or current_waypoint == 16
                or current_waypoint == 17
                or current_waypoint == 18
            )

        return fn

    return None


class ReferenceTrace:

    def __init__(self, path_to_csv_file: str, env_name: str, bounding_box: BoundingBox):
        assert os.path.exists(path_to_csv_file), "{} does not exist".format(
            path_to_csv_file
        )
        assert path_to_csv_file.endswith(".csv"), "{} not a csv file".format(
            path_to_csv_file
        )

        self.path_to_csv_file = path_to_csv_file
        self.rows = []
        self.env_name = env_name
        self.bounding_box = bounding_box

        self.logger = GlobalLog("reference_trace")

    def get_header(self) -> List[str]:
        return self.get_rows()[0]

    def get_rows(self) -> List[List[str]]:
        if len(self.rows) > 0:
            return self.rows

        with open(self.path_to_csv_file) as csv_file:
            for row in csv.reader(csv_file, delimiter=","):
                self.rows.append(row)

        return self.rows

    @staticmethod
    def parse_row_item(
        row_item: str, row_item_type: str
    ) -> Union[int, float, bool, str]:
        if row_item_type == "int":
            result = int(row_item)
        elif row_item_type == "float":
            result = float(row_item)
        elif row_item_type == "str":
            result = row_item
        elif row_item_type == "bool":
            result = bool(row_item)
        else:
            raise RuntimeError("Type {} not supported".format(row_item_type))
        return result

    def parse_row(self, row: List[str]) -> Tuple[Dict, Dict]:
        result_constant = dict()
        result = dict()

        if self.env_name == DONKEY_SIM_NAME or MOCK_SIM_NAME:
            num_columns = len(DONKEY_REFERENCE_TRACE_HEADER.split(","))
            columns_keys = DONKEY_REFERENCE_TRACE_HEADER
            columns_types = DONKEY_REFERENCE_TRACE_HEADER_TYPES
        else:
            raise RuntimeError("Unknown env name: {}".format(self.env_name))

        constant_columns_keys = REFERENCE_TRACE_CONSTANT_HEADER
        constant_columns_types = REFERENCE_TRACE_CONSTANT_HEADER_TYPES
        num_constant_columns = len(constant_columns_keys.split(","))
        assert (
            len(row) == num_constant_columns + num_columns
        ), "Number of columns of row {} is {}. Expected: {}".format(
            row, len(row), num_constant_columns + num_columns
        )
        assert num_columns == len(
            columns_types.split(",")
        ), "Num columns {} != Num types {}".format(
            num_columns, len(columns_types.split(","))
        )

        for i, column_key_constant in enumerate(constant_columns_keys.split(",")):
            row_item = row[i]
            row_item_type = constant_columns_types.split(",")[i]
            result_constant[column_key_constant] = self.parse_row_item(
                row_item=row_item, row_item_type=row_item_type
            )

        for i, column_key in enumerate(columns_keys.split(",")):
            row_item = row[i + num_constant_columns]
            row_item_type = columns_types.split(",")[i]
            result[column_key] = self.parse_row_item(
                row_item=row_item, row_item_type=row_item_type
            )

        return result_constant, self.postprocessing_dict(result=result)

    def postprocessing_dict(self, result: Dict) -> Dict:
        if self.env_name == DONKEY_SIM_NAME or MOCK_SIM_NAME:
            pitch, yaw, roll = 0.0, result["rotation_angle"], 0.0
            q = Quaternion.from_euler(pitch=pitch, yaw=yaw, roll=roll)
            result["rot_x"], result["rot_y"], result["rot_z"], result["rot_w"] = (
                round(q.x, 2),
                q.y,
                round(q.z, 2),
                q.w,
            )

            result["vel_y"] = 0.0

        return result

    def get_states(
        self,
        road: Road,
        state_condition_fn: Callable[[int], bool] = None,
        other_bounding_box: BoundingBox = None,
        donkey_simulator_scene: SimulatorScene = None,
    ) -> List[State]:

        if os.path.exists(self.path_to_csv_file.replace(".csv", "_valid.json")):
            with open(self.path_to_csv_file.replace(".csv", "_valid.json")) as f:
                json_array = json.load(f)
                result = []
                for item in json_array:
                    state = get_state(
                        road=road,
                        env_name=self.env_name,
                        bounding_box=self.bounding_box,
                        donkey_simulator_scene=donkey_simulator_scene,
                    )
                    state.update_state(check_orientation=False, **item)
                    result.append(state)

                return result

        result = []
        _, rows = self.get_rows()[0], self.get_rows()[1:]
        for row in rows:
            state = get_state(
                road=road,
                env_name=self.env_name,
                bounding_box=self.bounding_box,
                donkey_simulator_scene=donkey_simulator_scene,
            )
            constant_header_dict, header_dict = self.parse_row(row=row)
            if (
                state_condition_fn is not None
                and state_condition_fn(constant_header_dict[CURRENT_WAYPOINT_KEY])
                and other_bounding_box is not None
            ):
                state.set_bounding_box(bounding_box=other_bounding_box)
            state.update_state(check_orientation=False, **header_dict)
            if state.is_valid():
                result.append(state)
            else:
                self.logger.debug(
                    "State from reference trace {} is not valid: {}".format(
                        self.path_to_csv_file, state
                    )
                )

        if not os.path.exists(self.path_to_csv_file.replace(".csv", "_valid.json")):

            json_string = json.dumps(
                [state.to_dict() for state in result],
                indent=4,
            )

            with open(self.path_to_csv_file.replace(".csv", "_valid.json"), "w") as f:
                f.write(json_string)

        self.logger.debug(
            "Given {} states, {} are valid".format(len(rows), len(result))
        )
        assert len(result) > 0, "No valid state"

        return result


if __name__ == "__main__":

    env_name = DONKEY_SIM_NAME
    scene_name = GENERATED_TRACK_NAME
    track_num = 0
    simulator_scene = make_simulator_scene(scene_name=scene_name, track_num=track_num)
    road = get_road(
        simulator_name=env_name,
        road_points=[],
        control_points=[],
        road_width=1,
        constant_road=True,
        simulator_scene=simulator_scene,
    )
    bounding_box = get_bounding_box(
        env_name=env_name,
        waypoints=road.get_waypoints(),
        road_width=road.road_width,
        donkey_simulator_scene=simulator_scene,
    )
    # increasing the bounding box when car reaches the last curve in the sandbox_lab track
    # other_bounding_box = get_bounding_box(env_name=env_name, waypoints=road.get_waypoints(), road_width=road.road_width + 3)
    other_bounding_box = None

    reference_trace = ReferenceTrace(
        path_to_csv_file="../logs/donkey/generated_track/reference_trace_0.csv",
        env_name=env_name,
        bounding_box=bounding_box,
    )
    states = reference_trace.get_states(
        road=road,
        state_condition_fn=get_state_condition_function(
            env_name=env_name, constant_road=True
        ),
        other_bounding_box=other_bounding_box,
        donkey_simulator_scene=simulator_scene,
    )
    print(states[0].get_representation())
