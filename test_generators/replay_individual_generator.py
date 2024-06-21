import json
import os
from typing import List, cast, Dict, Tuple

import numpy as np

from code_pipeline.evaluator import Evaluator
from code_pipeline.real_evaluator import RealEvaluator
from envs.vec_env.vec_video_recorder import VecVideoRecorder
from global_log import GlobalLog
from test.config import STATE_PAIR_INDIVIDUAL_NAME
from test.individual import Individual
from test.state_individual import StateIndividual
from test.state_pair_individual import StatePairIndividual


class RecoveryItem:

    def __init__(
        self,
        archive_filename: str,
        agent_name: str,
        runs: Dict,
        recovery_percentage: float,
        num_runs: int,
    ) -> None:
        self.archive_filename = archive_filename
        self.agent_name = agent_name
        self.runs = runs
        self.recovery_percentage = recovery_percentage
        self.num_runs = num_runs


class RecoveryInfo:

    def __init__(
        self, recovery_items: List[RecoveryItem], avg_recovery_percentage: float
    ) -> None:
        self.recovery_items = recovery_items
        self.avg_recovery_percentage = avg_recovery_percentage


class ReplayIndividualGenerator:

    def __init__(self, evaluator: Evaluator, individual_name: str, num_runs: int):
        self.evaluator = evaluator
        self.individual_name = individual_name
        self.index = 0
        self.num_runs = num_runs
        assert self.num_runs > 0, "Num runs must be > 0"
        self.logger = GlobalLog("ReplayIndividualGenerator")

    def save_recovery_file(
        self,
        filepath: str,
        filename_no_ext: str,
        individual_dicts: Dict,
        agent_type: str,
        model_path: str,
    ) -> None:

        d = dict()
        recovery_percentages = []
        for i, key in enumerate(individual_dicts.keys()):

            d[i] = {"archive_filename": key, "agent": agent_type}

            if (
                "replicable_percentage" in individual_dicts[key]
                or len(individual_dicts[key]) == 0
            ):
                if "replicable_percentage" in individual_dicts[key]:
                    d[i]["message"] = individual_dicts[key]["replicable_percentage"]
                else:
                    d[i]["message"] = "No individuals to replicate"

                continue

            if agent_type == "supervised":
                d[i]["model"] = model_path

            d[i]["runs"] = individual_dicts[key]
            num_recoveries = len(
                list(
                    filter(
                        lambda percentage: percentage > 50,
                        individual_dicts[key].values(),
                    )
                )
            )
            recovery_percentage = num_recoveries / len(individual_dicts[key].keys())
            d[i]["recovery_percentage"] = recovery_percentage
            d[i]["num_runs"] = self.num_runs
            recovery_percentages.append(recovery_percentage)

        d["avg_recovery_percentage"] = np.mean(recovery_percentages)

        with open(os.path.join(filepath, "{}.json".format(filename_no_ext)), "w") as f:
            obj = json.dumps(d, indent=4)
            f.write(obj)

    @staticmethod
    def load_recovery_file(filepath: str) -> RecoveryInfo:
        with open(filepath, "r") as f:
            json_obj = json.loads(f.read())
            recovery_items = []
            for key in json_obj.keys():
                if key == "avg_recovery_percentage":
                    avg_recovery_percentage = json_obj[key]
                # otherwise the run does not contain any boundary state to replicate
                elif "message" not in json_obj[key]:
                    recovery_item = RecoveryItem(
                        archive_filename=json_obj[key]["archive_filename"],
                        agent_name=json_obj[key]["agent"],
                        num_runs=json_obj[key]["num_runs"],
                        runs=json_obj[key]["runs"],
                        recovery_percentage=json_obj[key]["recovery_percentage"],
                    )
                    recovery_items.append(recovery_item)

        return RecoveryInfo(
            recovery_items=recovery_items,
            avg_recovery_percentage=avg_recovery_percentage,
        )

    def replay_for_recovery(
        self, individuals: List[Individual], execute_success: bool = True
    ) -> Tuple[Dict, List[StateIndividual]]:
        """
        Does not close the environment
        """

        assert (
            self.individual_name == STATE_PAIR_INDIVIDUAL_NAME
        ), "This method is available only for {}".format(STATE_PAIR_INDIVIDUAL_NAME)

        individual_successful_runs = dict()
        state_individuals = []

        for idx, ind in enumerate(individuals):
            ind_clone = ind.clone()
            state_pair_individual_clone = cast(StatePairIndividual, ind_clone)
            assert (
                state_pair_individual_clone.is_frontier_pair()
            ), "The individual with id {} must be a frontier pair".format(
                state_pair_individual_clone.id
            )

            if state_pair_individual_clone.m1.is_success:
                member = "first" if execute_success else "second"
            elif state_pair_individual_clone.m2.is_success:
                member = "second" if execute_success else "first"
            else:
                raise RuntimeError("The two members do not form a frontier point")

            if ind.id not in individual_successful_runs:
                individual_successful_runs[ind.id] = 0

            for i in range(self.num_runs):

                self.logger.info(
                    "Executing {} member of individual with id {}".format(
                        member, ind.id
                    )
                )

                if isinstance(self.evaluator, RealEvaluator) and isinstance(
                    self.evaluator.env, VecVideoRecorder
                ):
                    self.evaluator.env.id = ind.id

                state_individual = StateIndividual(
                    state=(
                        state_pair_individual_clone.m1.state
                        if member == "first"
                        else state_pair_individual_clone.m2.state
                    )
                )
                self.evaluator.run_sim(individual=state_individual)
                state_individuals.append(state_individual)

                if state_individual.is_success():
                    individual_successful_runs[ind.id] += 1

        for key, value in individual_successful_runs.items():
            individual_successful_runs[key] = value * 100 / self.num_runs
            self.logger.info(
                "Individual with id {}. Number of successful runs: {:.2f}%".format(
                    key, individual_successful_runs[key]
                )
            )

        return individual_successful_runs, state_individuals

    def replicate(self, individuals: List[Individual], close: bool = True) -> Dict:

        self.logger.info("Found {} individuals to replicate".format(len(individuals)))

        individual_runs = dict()

        for ind in individuals:
            for i in range(self.num_runs):
                self.logger.info("Replicating individual with id {}".format(ind.id))

                if ind.id not in individual_runs:
                    individual_runs[ind.id] = 0

                ind_clone = ind.clone()

                # resetting fitness and is_success flag
                ind_clone.fitness = None
                ind_clone.set_success(is_success=None)
                if self.individual_name == STATE_PAIR_INDIVIDUAL_NAME:
                    state_pair_individual_clone = cast(StatePairIndividual, ind_clone)
                    state_pair_individual_clone.m1.fitness = None
                    state_pair_individual_clone.m2.fitness = None
                    state_pair_individual_clone.m1.is_success = None
                    state_pair_individual_clone.m2.is_success = None

                try:
                    while not ind_clone.is_evaluated():
                        self.evaluator.run_sim(individual=ind_clone)

                    assert (
                        ind.is_success() == ind_clone.is_success()
                    ), "Replicating the individual produces different outcome {} != {}".format(
                        ind_clone.is_success(), ind.is_success()
                    )

                    if self.individual_name == STATE_PAIR_INDIVIDUAL_NAME:
                        state_pair_individual = cast(StatePairIndividual, ind)
                        state_pair_individual_clone = cast(
                            StatePairIndividual, ind_clone
                        )

                        assert (
                            state_pair_individual.m1.is_success
                            == state_pair_individual_clone.m1.is_success
                        ), "The outcome of the first member should be the same {} != {}".format(
                            state_pair_individual.m1.is_success,
                            state_pair_individual_clone.m1.is_success,
                        )

                        assert (
                            state_pair_individual.m2.is_success
                            == state_pair_individual_clone.m2.is_success
                        ), "The outcome of the second member should be the same {} != {}".format(
                            state_pair_individual.m2.is_success,
                            state_pair_individual_clone.m2.is_success,
                        )

                        assert (
                            state_pair_individual.is_frontier_pair()
                            == state_pair_individual_clone.is_frontier_pair()
                        ), "Replicating the individual produces different is_frontier_pair outcome {} != {}".format(
                            state_pair_individual.is_frontier_pair(),
                            state_pair_individual_clone.is_frontier_pair(),
                        )

                    individual_runs[ind.id] += 1

                except AssertionError as e:
                    self.logger.error(e)

        if close:
            self.evaluator.close()

        for key, value in individual_runs.items():
            individual_runs[key] = {
                "replicable_percentage": value * 100 / self.num_runs
            }
            self.logger.info(
                "Replicated individual with id {}. Number of successful runs: {:.2f}%".format(
                    key, individual_runs[key]["replicable_percentage"]
                )
            )

        return individual_runs
