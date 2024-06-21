import copy
from typing import Dict, Any

from shapely.geometry import Point

from config import DONKEY_SIM_NAME
from envs.donkey.donkey_env_utils import make_simulator_scene
from envs.donkey.scenes.simulator_scenes import GENERATED_TRACK_NAME
from global_log import GlobalLog
from self_driving.bounding_box_utils import get_bounding_box
from self_driving.road_utils import get_road
from self_driving.state import State
from self_driving.state_utils import get_state
from test.fitness_utils import make_fitness
from test.individual import Individual


class StateIndividual(Individual):

    def __init__(self, state: State, start_id: int = 1):
        super().__init__(start_id=start_id)
        self.state = state

        self.logger = GlobalLog("StateIndividual")

    def clone(self) -> "Individual":
        state_individual = StateIndividual(
            state=self.state.clone(), start_id=self.start_id
        )
        state_individual.set_fitness(fitness=self.fitness.clone())
        state_individual.set_success(is_success=self._is_success)
        state_individual.set_behavioural_metrics(
            speeds=copy.deepcopy(self.behavioural_metrics["speeds"]),
            steering_angles=copy.deepcopy(self.behavioural_metrics["steering_angles"]),
            lateral_positions=copy.deepcopy(
                self.behavioural_metrics["lateral_positions"]
            ),
            ctes=copy.deepcopy(self.behavioural_metrics["ctes"]),
        )
        state_individual.set_observations(observations=copy.deepcopy(self.observations))
        state_individual.set_actions(actions=copy.deepcopy(self.actions))
        return state_individual

    def get_implementation(self) -> Any:
        return self.state

    def get_representation(self) -> str:
        return self.state.get_representation()

    @staticmethod
    def get_state_skeleton(member_export: Dict) -> State:
        # FIXME: refactor with state_pair_individual get_state_skeleton method
        assert (
            member_export.get("env_name", None) is not None
        ), "The key env_name is not present in the member state export"
        env_name = member_export["env_name"]
        simulator_scene_name = None
        simulator_track_num = None
        constant_road = member_export["road"].get("constant_road", False)
        road_width = member_export["road"]["road_width"]
        road_points = []
        control_points = []
        if env_name == DONKEY_SIM_NAME:
            assert (
                member_export["road"].get("simulator_scene_name", None) is not None
            ), "The key simulator_scene_name is not present in the member state export"
            simulator_scene_name = member_export["road"]["simulator_scene_name"]
            if simulator_scene_name == GENERATED_TRACK_NAME:
                assert (
                    member_export["road"].get("simulator_track_num", None) is not None
                ), "The key simulator_track_num is not present in the member state export"
                simulator_track_num = member_export["road"]["simulator_track_num"]

        if not constant_road:
            assert member_export["road"].get(
                "control_points", None
            ), "Control points must be present when road is not constant"
            assert member_export["road"].get(
                "road_points", None
            ), "Road points must be present when road is not constant"
            control_points = [
                Point(cp_tuple[0], cp_tuple[1], cp_tuple[2])
                for cp_tuple in member_export["road"]["control_points"]
            ]
            road_points = [
                Point(cp_tuple[0], cp_tuple[1], cp_tuple[2])
                for cp_tuple in member_export["road"]["road_points"]
            ]

        simulator_scene = make_simulator_scene(
            scene_name=simulator_scene_name, track_num=simulator_track_num
        )
        road = get_road(
            road_points=road_points,
            control_points=control_points,
            road_width=road_width,
            simulator_name=env_name,
            constant_road=constant_road,
            simulator_scene=simulator_scene,
        )
        bounding_box = get_bounding_box(
            env_name=env_name,
            waypoints=road.get_waypoints(),
            road_width=road.road_width,
            donkey_simulator_scene=simulator_scene,
        )
        return get_state(
            road=road,
            env_name=env_name,
            bounding_box=bounding_box,
            donkey_simulator_scene=simulator_scene,
        )

    def parse(self, individual_export: Dict) -> None:
        # FIXME: refactor with state_pair_individual parse method
        state_individual_export = individual_export["representation"]
        m1_state_skeleton = self.get_state_skeleton(
            member_export=state_individual_export["state"]
        )
        m1_state_skeleton.parse(**state_individual_export["state"]["implementation"])
        behavioural_metrics_m1 = individual_export["behavioural_metrics"]
        speeds = behavioural_metrics_m1["speeds"]
        steering_angles = behavioural_metrics_m1["steering_angles"]
        lateral_positions = behavioural_metrics_m1["lateral_positions"]
        ctes = behavioural_metrics_m1["ctes"]
        self.set_behavioural_metrics(
            speeds=speeds,
            steering_angles=steering_angles,
            lateral_positions=lateral_positions,
            ctes=ctes,
        )
        fitness_m1 = make_fitness(
            fitness_name=state_individual_export["fitness"][0],
            lateral_positions=lateral_positions,
            ctes=ctes,
        )
        self.set_fitness(fitness=fitness_m1)
        self.set_success(is_success=state_individual_export["is_success"])

    def export(self) -> Dict:
        result = dict()
        result["representation"] = self.state.export()
        return super().export_common(export_dict=result)

    def mutate(self, bias: bool = False, mutation_extent: int = 1) -> "Individual":
        state, is_valid = self.state.mutate(bias=bias)
        max_iterations = 100
        while not is_valid and max_iterations > 0:
            state, is_valid = self.state.mutate(bias=bias)
            max_iterations -= 1

        if max_iterations == 0:
            raise RuntimeError(
                "Not possible to mutate state {}".format(state.get_representation())
            )

        return StateIndividual(state=state, start_id=self.start_id)

    def reset(self) -> None:
        super().reset()

    def __eq__(self, other: "StateIndividual") -> bool:
        if isinstance(other, Individual):
            # TODO: should call the State __eq__ method
            return self.state == other.state
        raise RuntimeError("other {} is not an individual".format(type(other)))

    def __hash__(self) -> int:
        return self.state.__hash__()
