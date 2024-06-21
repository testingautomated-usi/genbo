import time
from abc import ABC, abstractmethod
from typing import cast, Tuple, Optional

from code_pipeline.evaluator import Evaluator
from global_log import GlobalLog
from test.archive import Archive
from test.config import STATE_PAIR_INDIVIDUAL_NAME
from test.individual import Individual
from test.individual_utils import make_individual
from test.state_individual import StateIndividual
from test.state_pair_individual import StatePairIndividual
from test_generators.road_test_generator import RoadTestGenerator
from test_generators.state_test_generator import SeedStateTestGenerator


class IndividualGenerator(ABC):

    def __init__(
        self,
        individual_name: str,
        generator_type: str,
        evaluator: Evaluator,
        archive_logdir: str,
        archive_filename: str,
        start_id: int = 1,
        maximize: bool = True,
        seed_state_test_generator: SeedStateTestGenerator = None,
        road_test_generator: RoadTestGenerator = None,
        num_runs_failure: int = 1,
        bias: bool = False,
        mutate_both_members: bool = False,
    ):
        assert generator_type in [
            "state",
            "road",
        ], "Generator type must be in {}".format(["state", "road"])
        self.generator_type = generator_type
        self.individual_name = individual_name
        self.evaluator = evaluator
        self.mutate_both_members = mutate_both_members
        self.archive = Archive(
            archive_logdir=archive_logdir,
            archive_filename=archive_filename,
            mutate_both_members=mutate_both_members,
        )
        self.seed_state_test_generator = seed_state_test_generator
        self.road_test_generator = road_test_generator
        self.start_id = start_id
        self.maximize = maximize
        self.bias = bias
        self.num_runs_failure = num_runs_failure
        self.selected_states = set()
        self.logger = GlobalLog("IndividualGenerator")

    @abstractmethod
    def evolve(self, num_iterations: int, close_at_last: bool = True) -> Archive:
        raise NotImplemented("Not implemented")

    def repeat_execution(self, individual: Individual) -> Individual:

        clone_individual = individual.clone()

        if self.individual_name == STATE_PAIR_INDIVIDUAL_NAME:
            clone_individual_cast = cast(StatePairIndividual, clone_individual)
            assert clone_individual_cast.are_members_close(), (
                "The two members of the individual need to be close to each other. "
                "There must be an error in the mutation operators that must ensure that. Individual: {}".format(
                    clone_individual_cast
                )
            )

            if clone_individual_cast.is_frontier_pair():
                self.logger.debug(
                    "The individual with id {} is a frontier pair. Let's verify it again using {} executions".format(
                        clone_individual.id, self.num_runs_failure
                    )
                )
                num_runs_frontier_pair = 0
                mutated_individual_no_frontier_pair = None
                mutated_individual_frontier_pair = None
                for i in range(self.num_runs_failure):
                    mutated_individual_clone = cast(
                        StatePairIndividual, clone_individual.clone()
                    )
                    mutated_individual_clone.reset()

                    while not mutated_individual_clone.is_evaluated():
                        self.logger.debug(
                            "Evaluating mutated individual for run #{}".format(i)
                        )
                        self.evaluator.run_sim(individual=mutated_individual_clone)

                    if mutated_individual_clone.is_frontier_pair():
                        num_runs_frontier_pair += 1
                        mutated_individual_frontier_pair = (
                            mutated_individual_clone.clone()
                        )
                    else:
                        mutated_individual_no_frontier_pair = (
                            mutated_individual_clone.clone()
                        )

                if self.num_runs_failure == 0:
                    num_runs_frontier_pair = 1

                self.logger.debug(
                    "Given {} runs, the mutated individual with id {} "
                    "has resulted in a frontier pair {:.2f}% of the times".format(
                        self.num_runs_failure,
                        clone_individual.id,
                        num_runs_frontier_pair * 100 / self.num_runs_failure,
                    )
                )

                if num_runs_frontier_pair / self.num_runs_failure < 0.5:
                    assert (
                        mutated_individual_no_frontier_pair is not None
                    ), "Mutated individual that does not form a frontier pair cannot be None"
                    # replace mutated individual that formed a frontier pair the first time
                    # with the last individual that did not form it (it could have been
                    # the first but the result would not change)
                    clone_individual = mutated_individual_no_frontier_pair
                    self.logger.debug(
                        "Considering the individual which is not a frontier pair"
                    )
                    assert not cast(
                        StatePairIndividual, clone_individual
                    ).is_frontier_pair(), "The individual must not be frontier pair: {}".format(
                        clone_individual
                    )
                else:
                    assert (
                        mutated_individual_frontier_pair is not None
                    ), "Mutated individual that forms a frontier pair cannot be None"
                    clone_individual = mutated_individual_frontier_pair
                    assert cast(
                        StatePairIndividual, clone_individual
                    ).is_frontier_pair(), "The individual must be a frontier pair: {}".format(
                        clone_individual
                    )
        else:
            if not clone_individual.is_success():
                self.logger.debug(
                    "The individual with id {} has failed. Let's verify it again using {} executions".format(
                        clone_individual.id, self.num_runs_failure
                    )
                )

                clone_individual_cast = cast(StateIndividual, clone_individual)
                num_failures = 0
                mutated_individual_fail = None
                mutated_individual_success = None
                for i in range(self.num_runs_failure):
                    mutated_individual_clone = clone_individual_cast.clone()
                    mutated_individual_clone.reset()

                    while not mutated_individual_clone.is_evaluated():
                        self.logger.debug(
                            "Evaluating mutated individual for run #{}".format(i)
                        )
                        self.evaluator.run_sim(individual=mutated_individual_clone)

                    if not mutated_individual_clone.is_success():
                        num_failures += 1
                        mutated_individual_fail = mutated_individual_clone.clone()
                    else:
                        mutated_individual_success = mutated_individual_clone.clone()

                if self.num_runs_failure == 0:
                    num_failures = 1

                self.logger.debug(
                    "Given {} runs, the mutated individual with id {} "
                    "has resulted in a failure {:.2f}% of the times".format(
                        self.num_runs_failure,
                        clone_individual.id,
                        num_failures * 100 / self.num_runs_failure,
                    )
                )

                if num_failures / self.num_runs_failure < 0.5:
                    assert (
                        mutated_individual_success is not None
                    ), "Mutated individual that succeeds cannot be None"
                    # replace mutated individual that failed the first time
                    # with the last individual that succeeded (it could have been
                    # the first but the result would not change)
                    clone_individual = mutated_individual_success
                    self.logger.debug(
                        "Considering the individual which is not a failure"
                    )
                    assert (
                        clone_individual.is_success()
                    ), "The individual must be successful: {}".format(clone_individual)
                else:
                    assert (
                        mutated_individual_fail is not None
                    ), "Mutated individual that does not succeed cannot be None"
                    clone_individual = mutated_individual_fail

        return clone_individual

    def initialize_individual(self) -> Tuple[Optional[Individual], bool]:
        is_success = False
        start_time = time.perf_counter()
        max_iterations_external = 100
        try:
            while not is_success and max_iterations_external > 0:
                if self.generator_type == "state":
                    assert (
                        self.seed_state_test_generator is not None
                    ), "Seed state test generator must not be None"
                    # I assume that in the following state the agent will succeed since it is taken from a nominal
                    # trajectory
                    state = self.seed_state_test_generator.generate()
                    # making sure that the same state is not selected across different restarts
                    while state in self.selected_states:
                        state = self.seed_state_test_generator.generate()

                    self.selected_states.add(state)
                    state_other = None

                    if self.individual_name == STATE_PAIR_INDIVIDUAL_NAME:
                        max_iterations = 10
                        valid = False
                        while not valid and max_iterations > 0:
                            state_other, valid = state.mutate(
                                other_state=state,
                                is_close_constraint=True,
                                bias=self.bias,
                            )
                            if state_other in self.selected_states:
                                valid = False
                            max_iterations -= 1
                            self.logger.debug(
                                f"^^^^^^^^^^^^^^^^ MAX ITERATIONS INIT {max_iterations} ^^^^^^^^^^^^^^^^"
                            )

                        if max_iterations == 0:
                            self.logger.warn(
                                f"Cannot mutate state: {state}. Restarting at init"
                            )
                            return None, True

                        assert state.is_close_to(
                            other=state_other
                        ), "The two members are not close"
                        self.selected_states.add(state)

                    individual = make_individual(
                        individual_name=self.individual_name,
                        start_id=self.start_id,
                        state=state,
                        state_other=state_other,
                        mutate_both_members=self.mutate_both_members,
                    )
                else:
                    raise NotImplemented("Road test generator not yet supported")

                while not individual.is_evaluated():
                    self.logger.debug("Evaluating first individual")
                    self.evaluator.run_sim(individual=individual)

                max_iterations_external -= 1
                self.logger.debug(
                    f"$$$$$$$$$$ MAX ITERATIONS INIT EXTERNAL {max_iterations_external} $$$$$$$$$$"
                )

                individual_to_save = self.repeat_execution(individual=individual)

                if self.individual_name == STATE_PAIR_INDIVIDUAL_NAME:
                    individual_to_save_cast = cast(
                        StatePairIndividual, individual_to_save
                    )

                    self.logger.info(
                        f"Individual at initialization: {individual_to_save}"
                    )

                    if (
                        individual_to_save_cast.m1.state
                        == individual_to_save_cast.m2.state
                    ):
                        self.logger.info(
                            f"Individual was not mutated at initialization: {individual_to_save}. Retrying!"
                        )
                        is_success = False
                        continue

                    if individual_to_save_cast.is_frontier_pair():
                        self.archive.add_executed_individual(
                            individual=individual_to_save
                        )
                        self.archive.add_target_individual(
                            individual=individual_to_save
                        )
                        self.logger.info("{} Restart {}".format("*" * 100, "*" * 100))
                        return individual_to_save, True
                else:
                    individual_to_save_cast = cast(StateIndividual, individual_to_save)
                    if not individual_to_save_cast.is_success():
                        self.archive.add_executed_individual(
                            individual=individual_to_save_cast
                        )
                        self.archive.add_target_individual(
                            individual=individual_to_save_cast
                        )
                        self.logger.info("{} Restart {}".format("*" * 100, "*" * 100))
                        return individual, True

                self.archive.add_executed_individual(individual=individual_to_save)
                is_success = individual_to_save.is_success()

        except AssertionError as e:
            self.logger.error(e)
            return None, False

        if max_iterations_external == 0:
            self.logger.warn(
                f"Cannot mutate state (external): {state}. Restarting at init"
            )
            return None, True

        self.logger.info(
            "Time to initialize individual: {:.2f}s".format(
                time.perf_counter() - start_time
            )
        )
        return individual_to_save, False
