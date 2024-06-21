import time
from typing import Tuple, List, cast, Union

import numpy as np

from code_pipeline.evaluator import Evaluator
from global_log import GlobalLog
from test.archive import Archive
from test.config import STATE_PAIR_INDIVIDUAL_NAME
from test.individual import Individual
from test.state_pair_individual import StatePairIndividual
from test_generators.individual_generator import IndividualGenerator
from test_generators.road_test_generator import RoadTestGenerator
from test_generators.state_test_generator import SeedStateTestGenerator
from utils.cannot_mutate_exception import CannotMutateException


class SequenceIndividualGenerator(IndividualGenerator):

    def __init__(
        self,
        evaluator: Evaluator,
        individual_name: str,
        generator_type: str,
        num_restarts: int,
        length_exponential_factor: Union[float, int] = 1.1,
        archive_logdir: str = None,
        archive_filename: str = None,
        maximize: bool = True,
        start_id: int = 1,
        num_runs_failure: int = 1,
        seed_state_test_generator: SeedStateTestGenerator = None,
        road_test_generator: RoadTestGenerator = None,
        bias: bool = False,
        mutate_both_members: bool = False,
    ):

        super().__init__(
            individual_name=individual_name,
            generator_type=generator_type,
            evaluator=evaluator,
            archive_logdir=archive_logdir,
            archive_filename=archive_filename,
            start_id=start_id,
            maximize=maximize,
            seed_state_test_generator=seed_state_test_generator,
            road_test_generator=road_test_generator,
            num_runs_failure=num_runs_failure,
            bias=bias,
            mutate_both_members=mutate_both_members,
        )
        assert (
            self.individual_name == STATE_PAIR_INDIVIDUAL_NAME
        ), "Sequence generation not supported at the moment for single state individual"

        self.logger = GlobalLog("SequenceIndividualGenerator")
        self.selected_states = set()
        self.num_restarts = num_restarts
        self.times_elapsed_sequence = []
        self.start_time = time.perf_counter()
        self.length_exponential_factor = length_exponential_factor
        self.current_length = length_exponential_factor

        self.fitnesses: List[float] = []

        assert (
            self.length_exponential_factor > 0
        ), "Length exponential factor must be > 0. Found: {}".format(
            self.length_exponential_factor
        )

    def evolve(self, num_iterations: int, close_at_last: bool = True) -> Archive:

        for i in range(self.num_restarts):
            self.logger.info("Restart #{}".format(i))
            self.fitnesses.clear()
            restart, exception = self.evolve_sequence(num_iterations=num_iterations)
            self.logger.info(
                "Fitness values at restart {}: {}".format(i, self.fitnesses)
            )

            if exception:
                break

            if restart:
                if self.individual_name == STATE_PAIR_INDIVIDUAL_NAME:
                    self.logger.info("Frontier point found!")
                else:
                    self.logger.info("Misbehaviour found!")
            else:
                if self.individual_name == STATE_PAIR_INDIVIDUAL_NAME:
                    self.logger.info(
                        "Could not find frontier point in {} iterations".format(
                            num_iterations
                        )
                    )
                else:
                    self.logger.info(
                        "Could not find any misbehaviour in {} iterations".format(
                            num_iterations
                        )
                    )

        self.logger.info(
            "{} Time elapsed: {:.2f}s {}".format(
                "-" * 100, time.perf_counter() - self.start_time, "-" * 100
            )
        )

        if close_at_last:
            self.evaluator.close()

        return self.archive

    def evolve_sequence(self, num_iterations: int) -> Tuple[bool, bool]:
        """
        num_iterations in this case counts the number of executions of an individual
        returns: <restart, exception>
        """
        individual, restart = self.initialize_individual()

        if individual is None:
            self.current_length = self.length_exponential_factor
            return False, True

        if restart:
            self.current_length = self.length_exponential_factor
            return True, False

        try:
            individual_sequence: List[Individual] = [individual]
            current_iteration = 0
            cannot_mutate_exception = False
            while current_iteration < num_iterations:

                self.logger.info(
                    "{} Time elapsed: {:.2f}s at the beginning of iteration {} {}".format(
                        "-" * 100,
                        time.perf_counter() - self.start_time,
                        current_iteration,
                        "-" * 100,
                    )
                )

                start_time_sequence_gen = time.perf_counter()
                if isinstance(self.length_exponential_factor, int):
                    self.current_length = self.length_exponential_factor
                elif isinstance(self.length_exponential_factor, float):
                    self.current_length = min(
                        50, self.current_length * self.length_exponential_factor
                    )

                for seq_length in range(int(self.current_length)):
                    try:
                        mutated_individual = individual_sequence[-1].mutate(
                            bias=self.bias
                        )
                    except CannotMutateException:
                        cannot_mutate_exception = True
                        self.logger.warn(
                            f"Cannot mutate individual {individual_sequence[-1]}. "
                            f"Stopping the sequence at length {seq_length}"
                        )
                        break

                    individual_sequence.append(mutated_individual)

                self.times_elapsed_sequence.append(
                    time.perf_counter() - start_time_sequence_gen
                )
                self.logger.debug(
                    "Time spent generating a sequence of {} individuals: {:.2f}s +- {:.2f}s".format(
                        int(self.current_length),
                        np.mean(self.times_elapsed_sequence),
                        np.std(self.times_elapsed_sequence),
                    )
                )

                last_individual_in_sequence = individual_sequence[-1]

                if (
                    not self.archive.is_present(individual=last_individual_in_sequence)
                    and not last_individual_in_sequence.is_evaluated()
                ):

                    while not last_individual_in_sequence.is_evaluated():
                        self.logger.debug(
                            "Evaluating mutated individual at iteration {}".format(
                                current_iteration
                            )
                        )
                        self.evaluator.run_sim(individual=last_individual_in_sequence)

                    self.archive.add_executed_individual(
                        individual=last_individual_in_sequence
                    )
                    self.fitnesses.append(
                        last_individual_in_sequence.get_fitness().get_value()
                    )
                    current_iteration += 1

                    if self.individual_name == STATE_PAIR_INDIVIDUAL_NAME:
                        last_individual_in_sequence = cast(
                            StatePairIndividual, last_individual_in_sequence
                        )

                        if last_individual_in_sequence.is_frontier_pair():
                            assert last_individual_in_sequence.are_members_close(), (
                                "The two members of the individual need to be close to each other. "
                                "There must be an error in the mutation operators that must ensure that."
                            )

                            last_individual_in_sequence_repeated = (
                                self.repeat_execution(
                                    individual=last_individual_in_sequence
                                )
                            )
                            last_individual_in_sequence_repeated = cast(
                                StatePairIndividual,
                                last_individual_in_sequence_repeated,
                            )

                            if last_individual_in_sequence_repeated.is_frontier_pair():
                                assert (
                                    last_individual_in_sequence_repeated.are_members_close()
                                ), (
                                    "The two members of the individual need to be close to each other. "
                                    "There must be an error in the mutation operators that must ensure that."
                                )

                                self.archive.add_target_individual(
                                    individual=last_individual_in_sequence_repeated
                                )
                                self.logger.info(
                                    "{} Restart {}".format("*" * 100, "*" * 100)
                                )
                                self.current_length = self.length_exponential_factor
                                return True, False

                            self.logger.info(
                                "Not able to successfully replicate the frontier property of "
                                "individual with id {} at iteration {}".format(
                                    last_individual_in_sequence.id, current_iteration
                                )
                            )
                            self.logger.info(
                                "{} End Iteration {} {}".format(
                                    "*" * 100, current_iteration, "*" * 100
                                )
                            )

                        elif (
                            last_individual_in_sequence.m1.is_success is False
                            and last_individual_in_sequence.m2.is_success is False
                        ):
                            self.logger.info(
                                "Both members of individual with id {} failed at iteration {}.".format(
                                    last_individual_in_sequence.id, current_iteration
                                )
                            )
                            # we do not repeat the execution of an individual when both members fail; it is unlikely
                            # that this individual is flaky, given that it is probably far from the frontier

                            # let's run binary search to find a frontier pair
                            left, right = 0, len(individual_sequence) - 1
                            while left <= right and current_iteration < num_iterations:
                                m = (left + right) // 2
                                middle_individual = individual_sequence[m]
                                middle_individual = cast(
                                    StatePairIndividual, middle_individual
                                )
                                while not middle_individual.is_evaluated():
                                    self.logger.debug(
                                        "Evaluating mutated individual at iteration {} (binary search)".format(
                                            current_iteration
                                        )
                                    )
                                    self.evaluator.run_sim(individual=middle_individual)

                                current_iteration += 1
                                if middle_individual.is_success():
                                    left = m + 1
                                    self.logger.info(
                                        "Both members of individual with id {} are still successful "
                                        "at iteration {} (binary search). Going right!".format(
                                            middle_individual.id, current_iteration
                                        )
                                    )
                                elif (
                                    middle_individual.m1.is_success is False
                                    and middle_individual.m2.is_success is False
                                ):
                                    right = m - 1
                                    self.logger.info(
                                        "Both members of individual with id {} failed "
                                        "at iteration {} (binary search). Going left!".format(
                                            middle_individual.id, current_iteration
                                        )
                                    )
                                elif middle_individual.is_frontier_pair():

                                    assert middle_individual.are_members_close(), (
                                        "The two members of the individual need to be close to each other. "
                                        "There must be an error in the mutation operators that must ensure that."
                                    )

                                    middle_individual_repeated = self.repeat_execution(
                                        individual=middle_individual
                                    )
                                    middle_individual_repeated = cast(
                                        StatePairIndividual, middle_individual_repeated
                                    )

                                    if middle_individual_repeated.is_frontier_pair():
                                        assert (
                                            middle_individual_repeated.are_members_close()
                                        ), (
                                            "The two members of the individual need to be close to each other. "
                                            "There must be an error in the mutation operators that must ensure that."
                                        )

                                        self.archive.add_target_individual(
                                            individual=middle_individual_repeated
                                        )
                                        self.logger.info(
                                            "{} Restart {}".format("*" * 100, "*" * 100)
                                        )
                                        self.current_length = (
                                            self.length_exponential_factor
                                        )
                                        return True, False

                                    self.logger.info(
                                        "Not able to successfully replicate the frontier property of "
                                        "individual with id {} at iteration {} (binary search)".format(
                                            last_individual_in_sequence.id,
                                            current_iteration,
                                        )
                                    )
                                    # FIXME: here I'm just assuming that the last individual is representative af
                                    #  all the repeated executions
                                    if (
                                        middle_individual_repeated.m1.is_success
                                        is False
                                        and middle_individual_repeated.m2.is_success
                                        is False
                                    ):
                                        right = m - 1
                                        self.logger.info(
                                            "Both members of the failed to replicate individual with id {} failed "
                                            "at iteration {} (binary search). Going left!".format(
                                                middle_individual_repeated.id,
                                                current_iteration,
                                            )
                                        )
                                    elif middle_individual_repeated.is_success():
                                        left = m + 1
                                        self.logger.info(
                                            "Both members of the failed to replicate individual with id {} "
                                            "are successful at iteration {} (binary search). Going right!".format(
                                                middle_individual_repeated.id,
                                                current_iteration,
                                            )
                                        )
                                    else:
                                        raise RuntimeError(
                                            "During binary search it was not possible to replicate the frontier "
                                            "pair property of individual with id {} and the individual coming out "
                                            "from the repetition seems to be a frontier pair: {}: {}".format(
                                                middle_individual.id,
                                                middle_individual_repeated,
                                                middle_individual_repeated.is_frontier_pair(),
                                            )
                                        )

                                else:
                                    raise RuntimeError(
                                        "Unknown combination of is_success flags in the members "
                                        "of the individual with id: {} (binary search). {}, m1: {}, m2: {}, "
                                        "frontier pair property: {}".format(
                                            middle_individual.id,
                                            middle_individual,
                                            middle_individual.m1.is_success,
                                            middle_individual.m2.is_success,
                                            middle_individual.is_frontier_pair(),
                                        )
                                    )

                                self.logger.info(
                                    "{} End Iteration {} {}".format(
                                        "*" * 100, current_iteration, "*" * 100
                                    )
                                )

                        elif last_individual_in_sequence.is_success():
                            self.logger.info(
                                "Both members of individual with id {} are still successful at iteration {}. "
                                "Mutating the individual again for another {} times".format(
                                    last_individual_in_sequence.id,
                                    current_iteration,
                                    int(self.current_length),
                                )
                            )
                            # we do not repeat the execution of an individual when both members are successful;
                            # it is unlikely that this individual is flaky, given that it is probably far from
                            # the frontier
                            self.logger.info(
                                "{} End Iteration {} {}".format(
                                    "*" * 100, current_iteration, "*" * 100
                                )
                            )
                        else:
                            raise RuntimeError(
                                "Unknown combination of is_success flags in the members "
                                "of the individual with id: {} (binary search). {}, m1: {}, m2: {}, "
                                "frontier pair property: {}".format(
                                    last_individual_in_sequence.id,
                                    last_individual_in_sequence,
                                    last_individual_in_sequence.m1.is_success,
                                    last_individual_in_sequence.m2.is_success,
                                    last_individual_in_sequence.is_frontier_pair(),
                                )
                            )
                    else:
                        raise NotImplementedError(
                            "Individual name {} not supported yet".format(
                                self.individual_name
                            )
                        )

                if cannot_mutate_exception:
                    self.current_length = self.length_exponential_factor
                    self.logger.info(
                        f"Cannot mutate individual {individual_sequence[-1]} "
                        f"any further! Stopping the iterations at iteration "
                        f"{current_iteration} and restart."
                    )
                    return False, False

        except (AssertionError, RuntimeError, NotImplementedError) as e:
            self.logger.error(e)
            self.current_length = self.length_exponential_factor
            return False, True

        self.current_length = self.length_exponential_factor
        return False, False
