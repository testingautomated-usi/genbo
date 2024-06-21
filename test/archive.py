import os
import cv2
from typing import List, Dict, Set, cast

import numpy as np

from config_variables import LOGGING_LEVEL
from global_log import GlobalLog
from self_driving.donkey_car_state import DonkeyCarState
from self_driving.road import Road
from self_driving.state import State
from test.individual import Individual

import json

from test.individual_utils import make_individual
from test.state_individual import StateIndividual
from test.state_pair_individual import StatePairIndividual


class Archive:

    def __init__(
        self,
        replay: bool = False,
        archive_logdir: str = None,
        archive_filename: str = None,
        mutate_both_members: bool = False,
    ):
        self._ids = []
        self._all_ids = []
        self._idx_all_individuals = 0
        self._individuals: List[Individual] = []
        self._all_states: Set[State] = set()
        self._all_single_individuals: List[Individual] = []
        self._all_individuals: Set[Individual] = set()
        self.logger = GlobalLog("Archive")
        self.archive_logdir = archive_logdir
        self.archive_filename = archive_filename
        self.mutate_both_members = mutate_both_members

        if not replay and archive_logdir is not None and archive_filename is not None:
            os.makedirs(name=self.archive_logdir, exist_ok=True)

        # Do not create directory to store the first frames of every episode
        # if not replay and LOGGING_LEVEL == "DEBUG":
        #     os.makedirs(name=os.path.join(self.archive_logdir, self.archive_filename), exist_ok=True)

    def is_present(self, individual: Individual) -> bool:
        return individual in self._all_individuals

    def set_individual_properties(self, individual_runs: Dict) -> None:
        assert len(self._individuals) == len(
            individual_runs.items()
        ), "Number of individuals {} != {} number of properties to set".format(
            len(self._individuals), len(individual_runs.items())
        )
        for key in individual_runs.keys():
            properties_dict = individual_runs[key]
            individual = next(
                filter(lambda ind: ind.id == key, self._individuals), None
            )
            assert (
                individual is not None
            ), "Not possible to find an individual with id {}".format(key)
            for prop in properties_dict.keys():
                if prop == "replicable_percentage":
                    individual.replicable_percentage = properties_dict[prop]

    def add_executed_individual(self, individual: Individual) -> bool:
        """
        adds individual that has been executed
        """
        self._all_individuals.add(individual)

        if individual.id not in self._all_ids:
            self._all_ids.append(individual.id)

            if isinstance(individual, StatePairIndividual):
                _individual = cast(StatePairIndividual, individual)
                state_m1 = _individual.m1.state
                state_m2 = _individual.m2.state

                # if isinstance(state_m1, DonkeyCarState) and self.mutate_both_members:
                #     # check additional metrics consistency
                #     relative_orientation_m1 = cast(DonkeyCarState, state_m1).relative_orientation
                #     relative_orientation_m2 = cast(DonkeyCarState, state_m2).relative_orientation
                #     cte_m1 = cast(DonkeyCarState, state_m1).cte
                #     cte_m2 = cast(DonkeyCarState, state_m2).cte
                #     velocity_magnitude_m1 = cast(DonkeyCarState, state_m1).velocity_magnitude
                #     velocity_magnitude_m2 = cast(DonkeyCarState, state_m2).velocity_magnitude
                #     # add round to avoid precision errors when comparing two numbers
                #     p1 = np.asarray([abs(round(cte_m1, 3)), abs(round(velocity_magnitude_m1, 3)), abs(round(relative_orientation_m1, 3))])
                #     p2 = np.asarray([abs(round(cte_m2, 3)), abs(round(velocity_magnitude_m2, 3)), abs(round(relative_orientation_m2, 3))])
                #     p1_dominates_p2 = (p1 > p2).any() and (p1 >= p2).all()
                #     p2_dominates_p1 = (p2 > p1).any() and (p2 >= p1).all()
                #     assert (p1_dominates_p2 or p2_dominates_p1) and (p1_dominates_p2 != p2_dominates_p1), \
                #         f"Not possible to establish a dominance relation between m1 and m2. p1: {p1}, p2: {p2}"

                if state_m1 not in self._all_states:
                    self._all_states.add(state_m1)
                    state_individual_m1 = StateIndividual(state=state_m1)
                    state_individual_m1.set_success(
                        is_success=_individual.m1.is_success
                    )
                    state_individual_m1.set_fitness(fitness=_individual.m1.fitness)
                    state_individual_m1.set_behavioural_metrics(
                        speeds=_individual.m1.speeds,
                        steering_angles=_individual.m1.steering_angles,
                        lateral_positions=_individual.m1.lateral_positions,
                        ctes=_individual.m1.ctes,
                    )
                    state_individual_m1.set_observations(
                        observations=_individual.m1.observations
                    )
                    self._all_single_individuals.append(state_individual_m1)

                if state_m2 not in self._all_states:
                    self._all_states.add(state_m2)
                    state_individual_m2 = StateIndividual(state=state_m2)
                    state_individual_m2.set_success(
                        is_success=_individual.m2.is_success
                    )
                    state_individual_m2.set_fitness(fitness=_individual.m2.fitness)
                    state_individual_m2.set_behavioural_metrics(
                        speeds=_individual.m2.speeds,
                        steering_angles=_individual.m2.steering_angles,
                        lateral_positions=_individual.m2.lateral_positions,
                        ctes=_individual.m2.ctes,
                    )
                    state_individual_m2.set_observations(
                        observations=_individual.m2.observations
                    )
                    self._all_single_individuals.append(state_individual_m2)

            elif isinstance(individual, StateIndividual):
                _individual = cast(StateIndividual, individual)
                if _individual not in self._all_states:
                    self._all_states.add(_individual.state)
                    self._all_single_individuals.append(_individual)
            else:
                raise RuntimeError(
                    "Unknown individual type: {}".format(type(individual))
                )

            if self.archive_logdir is not None and self.archive_filename is not None:
                self.save(
                    filepath=self.archive_logdir,
                    filename_no_ext=self.archive_filename + "_all",
                )

            return True
        return False

    def add_target_individual(self, individual: Individual) -> bool:
        """
        adds individual that respects a certain target criterion, e.g., it is a frontier pair (state_pair_individual)
        or a misbehavior (state_individual)
        """
        if individual.id not in self._ids:
            self.logger.info(
                "Adding individual with id {} to the archive".format(individual.id)
            )
            self._ids.append(individual.id)
            self._individuals.append(individual)
            if self.archive_logdir is not None and self.archive_filename is not None:
                self.save(
                    filepath=self.archive_logdir, filename_no_ext=self.archive_filename
                )
            return True
        return False

    def dump(self, filename: str) -> Dict:
        result = dict()
        entities_to_export = (
            self._all_single_individuals
            if filename.endswith("_all")
            else self._individuals
        )
        for i, entity_to_export in enumerate(entities_to_export):
            result[i] = entity_to_export.export()
        return result

    def save(self, filepath: str, filename_no_ext: str) -> None:
        if filename_no_ext.endswith("_all"):
            obj = json.dumps(self.dump(filename=filename_no_ext))
            for individual in self._all_single_individuals[self._idx_all_individuals :]:
                for i, obs in enumerate(individual.observations):
                    cv2.imwrite(
                        filename=os.path.join(
                            self.archive_logdir,
                            self.archive_filename,
                            "{}_{}.png".format(individual.id, i),
                        ),
                        img=obs,
                    )
            self._idx_all_individuals += len(
                self._all_single_individuals[self._idx_all_individuals :]
            )
        else:
            obj = json.dumps(self.dump(filename=filename_no_ext), indent=4)

        with open(os.path.join(filepath, "{}.json".format(filename_no_ext)), "w") as f:
            f.write(obj)

    def get_individuals(self) -> List[Individual]:
        return self._individuals

    def load(
        self,
        filepath: str,
        filename_no_ext: str,
        individual_name: str,
        start_id: int = 1,
        state: State = None,
        state_other: State = None,
        check_null_state: bool = True,
    ) -> List[Individual]:
        assert os.path.exists(
            os.path.join(filepath, "{}.json".format(filename_no_ext))
        ), "File {} does not exist".format(
            os.path.join(filepath, "{}.json".format(filename_no_ext))
        )
        result = []
        with open(os.path.join(filepath, "{}.json".format(filename_no_ext)), "r") as f:
            json_obj = json.loads(f.read())
            for key in json_obj.keys():
                individual = make_individual(
                    individual_name=individual_name,
                    start_id=start_id,
                    state=state,
                    state_other=state_other,
                    check_null_state=check_null_state,
                    mutate_both_members=self.mutate_both_members,
                )
                individual.parse(individual_export=json_obj[key])
                self._individuals.append(individual)
                self._ids.append(individual.id)
                result.append(individual)
        return result
