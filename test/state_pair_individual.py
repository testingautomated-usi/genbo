import copy
from collections import namedtuple
from typing import Dict, Any, List

import numpy as np
from shapely.geometry import Point

from config import DONKEY_SIM_NAME
from envs.donkey.donkey_env_utils import make_simulator_scene
from envs.donkey.scenes.simulator_scenes import GENERATED_TRACK_NAME
from global_log import GlobalLog
from self_driving.bounding_box_utils import get_bounding_box
from self_driving.road_utils import get_road
from self_driving.state import State
from self_driving.state_utils import get_state
from test.fitness import Fitness
from test.fitness_pair_fitness import FitnessPairFitness
from test.fitness_utils import make_fitness
from test.individual import Individual
from utils.cannot_mutate_exception import CannotMutateException


class StatePairIndividual(Individual):

    def __init__(
        self,
        state1: State,
        state2: State,
        maximize: bool = True,
        start_id: int = 1,
        mutate_both_members: bool = False,
    ):
        super().__init__(start_id=start_id)
        self.m1 = namedtuple(
            "member",
            [
                "state",
                "fitness",
                "speeds",
                "steering_angles",
                "lateral_positions",
                "ctes",
                "is_success",
                "observations",
                "actions",
            ],
        )
        self.m2 = namedtuple(
            "member",
            [
                "state",
                "fitness",
                "speeds",
                "steering_angles",
                "lateral_positions",
                "ctes",
                "is_success",
                "observations",
                "actions",
            ],
        )

        self.m1.fitness = None
        self.m1.state = state1
        self.m1.speeds = []
        self.m1.steering_angles = []
        self.m1.lateral_positions = []
        self.m1.ctes = []
        self.m1.is_success = None
        self.m1.observations = []
        self.m1.actions = []

        self.m2.fitness = None
        self.m2.state = state2
        self.m2.speeds = []
        self.m2.steering_angles = []
        self.m2.lateral_positions = []
        self.m2.ctes = []
        self.m2.is_success = None
        self.m2.observations = []
        self.m2.actions = []

        self.maximize = maximize
        self.mutate_both_members = mutate_both_members

        self.logger = GlobalLog("StatePairIndividual")

    def clone(self) -> "Individual":
        state_pair_individual = StatePairIndividual(
            state1=self.m1.state.clone(),
            state2=self.m2.state.clone(),
            maximize=self.maximize,
            start_id=self.start_id,
            mutate_both_members=self.mutate_both_members,
        )
        state_pair_individual.id = self.id
        state_pair_individual.set_fitness(fitness=self.m1.fitness.clone())
        state_pair_individual.set_fitness(fitness=self.m2.fitness.clone())
        state_pair_individual.set_success(is_success=self.m1.is_success)
        state_pair_individual.set_success(is_success=self.m2.is_success)
        state_pair_individual.set_behavioural_metrics(
            speeds=copy.deepcopy(self.m1.speeds),
            steering_angles=copy.deepcopy(self.m1.steering_angles),
            lateral_positions=copy.deepcopy(self.m1.lateral_positions),
            ctes=copy.deepcopy(self.m1.ctes),
        )
        state_pair_individual.set_behavioural_metrics(
            speeds=copy.deepcopy(self.m2.speeds),
            steering_angles=copy.deepcopy(self.m2.steering_angles),
            lateral_positions=copy.deepcopy(self.m2.lateral_positions),
            ctes=copy.deepcopy(self.m2.ctes),
        )
        state_pair_individual.set_observations(
            observations=copy.deepcopy(self.m1.observations)
        )
        state_pair_individual.set_observations(
            observations=copy.deepcopy(self.m2.observations)
        )
        return state_pair_individual

    def get_implementation(self) -> Any:
        if self.m1.fitness is None:
            self.logger.debug("Get implementation of the first member")
            return self.m1.state
        if self.m2.fitness is None:
            self.logger.debug("Get implementation of the second member")
            return self.m2.state

    def get_representation(self) -> str:
        return "m1: {}, m2: {}".format(
            self.m1.state.get_representation(), self.m2.state.get_representation()
        )

    @staticmethod
    def get_state_skeleton(member_export: Dict) -> State:
        # FIXME: refactor with state_individual get_state_skeleton method
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
        # FIXME: refactor with state_individual parse method
        state_pair_individual_export = individual_export["representation"]
        m1_export = state_pair_individual_export["m1"]
        m1_state_skeleton = self.get_state_skeleton(member_export=m1_export["state"])
        m1_state_skeleton.parse(**m1_export["state"]["implementation"])
        self.m1.state = m1_state_skeleton
        behavioural_metrics_m1 = individual_export["behavioural_metrics"]["m1"]
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
            fitness_name=m1_export["fitness"][0],
            lateral_positions=lateral_positions,
            ctes=ctes,
        )
        self.set_fitness(fitness=fitness_m1)
        self.set_success(is_success=m1_export["is_success"])

        m2_export = state_pair_individual_export["m2"]
        m2_state_skeleton = self.get_state_skeleton(member_export=m2_export["state"])
        m2_state_skeleton.parse(**m2_export["state"]["implementation"])
        self.m2.state = m2_state_skeleton
        behavioural_metrics_m2 = individual_export["behavioural_metrics"]["m2"]
        speeds = behavioural_metrics_m2["speeds"]
        steering_angles = behavioural_metrics_m2["steering_angles"]
        lateral_positions = behavioural_metrics_m2["lateral_positions"]
        ctes = behavioural_metrics_m2["ctes"]
        self.set_behavioural_metrics(
            speeds=speeds,
            steering_angles=steering_angles,
            lateral_positions=lateral_positions,
            ctes=ctes,
        )
        fitness_m1 = make_fitness(
            fitness_name=m2_export["fitness"][0],
            lateral_positions=lateral_positions,
            ctes=ctes,
        )
        self.set_fitness(fitness=fitness_m1)
        self.set_success(is_success=m2_export["is_success"])

        self.id = individual_export["id"]
        if "replicable_percentage" in individual_export:
            self.replicable_percentage = individual_export["replicable_percentage"]

    def export(self) -> Dict:
        result = dict()
        result["representation"] = {
            "m1": {
                "state": self.m1.state.export(),
                "fitness": (self.m1.fitness.name, self.m1.fitness.get_value()),
                "is_success": bool(self.m1.is_success),
            },
            "m2": {
                "state": self.m2.state.export(),
                "fitness": (self.m2.fitness.name, self.m2.fitness.get_value()),
                "is_success": bool(self.m2.is_success),
            },
        }
        result["is_frontier_pair"] = bool(self.is_frontier_pair())
        return super().export_common(export_dict=result)

    def set_behavioural_metrics(
        self,
        speeds: List[float],
        steering_angles: List[float],
        lateral_positions: List[float],
        ctes: List[float],
    ) -> None:
        if len(self.m1.speeds) == 0:
            self.logger.debug("Setting behavioural metrics of the first member")
            self.m1.speeds = speeds
            self.m1.steering_angles = steering_angles
            self.m1.lateral_positions = lateral_positions
            self.m1.ctes = ctes
            if len(self.m2.speeds) > 0:
                _ = self.get_behavioural_metrics()
            return
        if len(self.m2.speeds) == 0:
            self.logger.debug("Setting behavioural metrics of the second member")
            self.m2.speeds = speeds
            self.m2.steering_angles = steering_angles
            self.m2.lateral_positions = lateral_positions
            self.m2.ctes = ctes
            if len(self.m1.speeds) > 0:
                _ = self.get_behavioural_metrics()
            return

    def get_behavioural_metrics(self) -> Dict:
        assert (
            len(self.m1.speeds) > 0 and len(self.m2.speeds) > 0
        ), "The behavioural metrics of the two members must be set"
        self.behavioural_metrics["m1"] = {
            "speeds": [float(value) for value in self.m1.speeds],
            "steering_angles": [float(value) for value in self.m1.steering_angles],
            "lateral_positions": [float(value) for value in self.m1.lateral_positions],
            "ctes": [float(value) for value in self.m1.ctes],
        }
        self.behavioural_metrics["m2"] = {
            "speeds": [float(value) for value in self.m2.speeds],
            "steering_angles": [float(value) for value in self.m2.steering_angles],
            "lateral_positions": [float(value) for value in self.m2.lateral_positions],
            "ctes": [float(value) for value in self.m2.ctes],
        }
        return self.behavioural_metrics

    def set_observations(self, observations: List[np.ndarray]) -> None:
        if len(self.m1.observations) == 0:
            self.m1.observations = observations
            self.logger.debug(
                "Setting observations of the first member: {}".format(
                    len(self.m1.observations)
                )
            )
            if len(self.m2.observations) > 0:
                _ = self.get_observations()
            return

        if len(self.m2.observations) == 0:
            self.m2.observations = observations
            self.logger.debug(
                "Setting observations of the second member: {}".format(
                    len(self.m2.observations)
                )
            )
            if len(self.m1.observations) > 0:
                _ = self.get_observations()
            return

    def get_observations(self) -> List[np.ndarray]:
        self.observations = self.m1.observations + self.m2.observations
        return copy.deepcopy(self.observations)

    def set_actions(self, actions: List[np.ndarray]) -> None:
        if len(self.m1.actions) == 0:
            self.m1.actions = actions
            self.logger.debug(
                "Setting actions of the first member: {}".format(len(self.m1.actions))
            )
            if len(self.m2.actions) > 0:
                _ = self.get_actions()
            return

        if len(self.m2.actions) == 0:
            self.m2.actions = actions
            self.logger.debug(
                "Setting actions of the second member: {}".format(len(self.m2.actions))
            )
            if len(self.m1.actions) > 0:
                _ = self.get_actions()
            return

    def get_actions(self) -> List[np.ndarray]:
        self.actions = self.m1.actions + self.m2.actions
        return copy.deepcopy(self.actions)

    def set_fitness(self, fitness: Fitness) -> None:
        if self.m1.fitness is None:
            self.logger.debug(
                "Setting fitness of the first member: {}".format(fitness.get_value())
            )
            self.m1.fitness = fitness
            if self.m2.fitness is not None:
                _ = self.get_fitness()
            return
        if self.m2.fitness is None:
            self.logger.debug(
                "Setting fitness of the second member: {}".format(fitness.get_value())
            )
            self.m2.fitness = fitness
            if self.m1.fitness is not None:
                _ = self.get_fitness()
            return

    def get_fitness(self) -> Fitness:
        assert (
            self.m1.fitness is not None and self.m2.fitness is not None
        ), "The fitness of the two members must not be None"
        self.fitness = FitnessPairFitness(
            fitness_m1=self.m1.fitness,
            fitness_m2=self.m2.fitness,
            maximize=self.maximize,
        )
        return self.fitness

    def is_evaluated(self) -> bool:
        return self.m1.fitness is not None and self.m2.fitness is not None

    def are_members_close(self) -> bool:
        return self.m1.state.is_close_to(other=self.m2.state)

    def set_success(self, is_success: bool) -> None:
        if self.m1.is_success is None:
            self.logger.debug(
                "Set is_success of the first member {}".format(is_success)
            )
            self.m1.is_success = is_success
            return
        if self.m2.is_success is None:
            self.logger.debug(
                "Set is_success of the second member {}".format(is_success)
            )
            self.m2.is_success = is_success
            return

    def is_success(self) -> bool:
        """
        returns True only if both members is_success fields are true
        """
        assert (
            self.m1.is_success is not None and self.m2.is_success is not None
        ), "The is_success flag of the two members must not be None"
        self._is_success = (
            True
            if self.m1.is_success == self.m2.is_success and self.m1.is_success
            else False
        )
        return self._is_success

    def is_frontier_pair(self):
        assert (
            self.m1.is_success is not None and self.m2.is_success is not None
        ), "The is_success flag of the two members must not be None"
        self.logger.debug(
            "IS FRONTIER PAIR. M1: {}, M2: {}, CLOSE: {}".format(
                self.m1.is_success, self.m2.is_success, self.are_members_close()
            )
        )
        return self.m1.is_success != self.m2.is_success and self.are_members_close()

    def mutate(self, bias: bool = False, mutation_extent: int = 1) -> "Individual":

        is_valid = False

        if self.mutate_both_members:

            max_iterations = 20

            # mutate the second member, which is the member mutated at initialization. If the bias is active
            # then the mutation is increasing and hence the second member dominates the first one. Therefore,
            # if we change the second member of a certain delta and we change the first member of the same delta,
            # then the dominance relation between the two members is invariant.
            while not is_valid and max_iterations > 0:
                new_state_m2, is_mutation_valid_m2 = self.m2.state.mutate(
                    other_state=self.m1.state, is_close_constraint=True, bias=bias
                )
                if is_mutation_valid_m2:
                    new_state_m1, is_mutation_valid_m1 = self.m1.state.mutate(
                        other_state=new_state_m2,
                        is_close_constraint=True,
                        bias=bias,
                        previous_state=self.m2.state,
                    )
                    if is_mutation_valid_m1:
                        self.m1.state = new_state_m1
                        self.m2.state = new_state_m2
                        is_valid = True

                max_iterations -= 1

            if max_iterations == 0:
                raise CannotMutateException("Not possible to mutate state both members")

            state_pair_individual = StatePairIndividual(
                state1=self.m1.state.clone(),
                state2=self.m2.state.clone(),
                start_id=self.start_id,
                mutate_both_members=self.mutate_both_members,
            )

        else:
            max_iterations = 10
            random_outcome = np.random.random() < 0.5

            self.logger.debug("Before mutation")
            self.logger.debug("M1: {}".format(self.m1.state))
            self.logger.debug("M2: {}".format(self.m2.state))

            if random_outcome:
                self.logger.debug("Mutate M1")
            else:
                self.logger.debug("Mutate M2")

            while not is_valid and max_iterations > 0:

                if random_outcome:
                    state, is_mutation_valid = self.m1.state.mutate(
                        other_state=self.m2.state, is_close_constraint=True, bias=bias
                    )
                else:
                    state, is_mutation_valid = self.m2.state.mutate(
                        other_state=self.m1.state, is_close_constraint=True, bias=bias
                    )

                if is_mutation_valid:
                    if random_outcome:
                        self.m1.state = state
                    else:
                        self.m2.state = state
                    is_valid = True

                max_iterations -= 1
                self.logger.debug(
                    f"+++++++++++++ MAX ITERATIONS (1) {max_iterations} +++++++++++++"
                )

            if max_iterations == 0:
                # try to mutate the other member
                self.logger.warn(
                    "Not possible to mutate state {}".format(
                        self.m1.state if random_outcome else self.m2.state
                    )
                )
                random_outcome = not random_outcome
                max_iterations = 10

                while not is_valid and max_iterations > 0:

                    if random_outcome:
                        state, is_mutation_valid = self.m1.state.mutate(
                            other_state=self.m2.state,
                            is_close_constraint=True,
                            bias=bias,
                        )
                    else:
                        state, is_mutation_valid = self.m2.state.mutate(
                            other_state=self.m1.state,
                            is_close_constraint=True,
                            bias=bias,
                        )

                    if is_mutation_valid:
                        if random_outcome:
                            self.m1.state = state
                        else:
                            self.m2.state = state
                        is_valid = True

                    max_iterations -= 1
                    self.logger.debug(
                        f"+++++++++++++ MAX ITERATIONS (2) {max_iterations} +++++++++++++"
                    )

                if max_iterations == 0:
                    raise CannotMutateException(
                        "Not possible to mutate state both members"
                    )

            state_pair_individual = StatePairIndividual(
                state1=self.m1.state.clone(),
                state2=self.m2.state.clone(),
                start_id=self.start_id,
                mutate_both_members=self.mutate_both_members,
            )
            if random_outcome:
                state_pair_individual.m2.fitness = self.m2.fitness
                state_pair_individual.m2.speeds = self.m2.speeds
                state_pair_individual.m2.steering_angles = self.m2.steering_angles
                state_pair_individual.m2.ctes = self.m2.ctes
                state_pair_individual.m2.lateral_positions = self.m2.lateral_positions
                state_pair_individual.m2.is_success = self.m2.is_success
                state_pair_individual.m2.observations = self.m2.observations
            else:
                state_pair_individual.m1.fitness = self.m1.fitness
                state_pair_individual.m1.speeds = self.m1.speeds
                state_pair_individual.m1.steering_angles = self.m1.steering_angles
                state_pair_individual.m1.ctes = self.m1.ctes
                state_pair_individual.m1.lateral_positions = self.m1.lateral_positions
                state_pair_individual.m1.is_success = self.m1.is_success
                state_pair_individual.m1.observations = self.m1.observations

        assert self.are_members_close(), "The two members are not close"

        self.logger.debug("After mutation")
        self.logger.debug("M1: {}".format(self.m1.state))
        self.logger.debug("M2: {}".format(self.m2.state))

        return state_pair_individual

    def reset(self) -> None:
        self.m1.fitness = None
        self.m1.speeds = []
        self.m1.steering_angles = []
        self.m1.lateral_positions = []
        self.m1.ctes = []
        self.m1.is_success = None
        self.m1.observations = []

        self.m2.fitness = None
        self.m2.speeds = []
        self.m2.steering_angles = []
        self.m2.lateral_positions = []
        self.m2.ctes = []
        self.m2.is_success = None
        self.m2.observations = []

        super().reset()

    def __str__(self) -> str:
        return self.get_representation()

    def __eq__(self, other: "StatePairIndividual") -> bool:
        if isinstance(other, Individual):
            # TODO: should call the State __eq__ method
            return self.m1.state == other.m1.state and self.m2.state == other.m2.state
        raise RuntimeError("other {} is not an individual".format(type(other)))

    def __hash__(self) -> int:
        return self.m1.state.__hash__() + self.m2.state.__hash__()
