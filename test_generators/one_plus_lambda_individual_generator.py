import time
from typing import cast, Tuple, List

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


class OnePlusLambdaIndividualGenerator(IndividualGenerator):

    def __init__(
        self,
        evaluator: Evaluator,
        individual_name: str,
        generator_type: str,
        num_restarts: int,
        archive_logdir: str = None,
        archive_filename: str = None,
        maximize: bool = True,
        lam: int = 1,
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
        self.lam = lam
        self.logger = GlobalLog("OnePlusLambdaIndividualGenerator")
        self.times_elapsed_mutation = []
        self.start_time = time.perf_counter()
        self.num_restarts = num_restarts

        self.fitnesses = []

    def evolve(self, num_iterations: int, close_at_last: bool = True) -> Archive:

        for i in range(self.num_restarts):
            self.logger.info("Restart #{}".format(i))
            self.fitnesses.clear()
            restart, exception = self.evolve_single(num_iterations=num_iterations)
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

    def evolve_single(self, num_iterations: int) -> Tuple[bool, bool]:

        individual, restart = self.initialize_individual()

        if individual is None:
            return False, True

        if restart:
            return True, False

        try:

            best_fitness = individual.get_fitness().get_value()
            self.logger.info("Fitness first individual: {}".format(best_fitness))
            self.logger.info("{} Start Evolution {}".format("=" * 100, "=" * 100))

            current_iteration = 0
            while current_iteration < num_iterations:

                self.logger.info(
                    "{} Time elapsed: {:.2f}s at the beginning of iteration {} {}".format(
                        "-" * 100,
                        time.perf_counter() - self.start_time,
                        current_iteration,
                        "-" * 100,
                    )
                )

                neighborhood: List[Individual] = []
                fitness_values: List[float] = []
                max_iterations = 10

                while len(neighborhood) < self.lam and max_iterations > 0:
                    start_time_mutation = time.perf_counter()
                    try:
                        mutated_individual = individual.mutate(bias=self.bias)
                    except CannotMutateException:
                        self.logger.warn(
                            f"Cannot mutate individual {individual}. Stopping the iterations at {current_iteration} and restart."
                        )
                        return False, False

                    end_time_mutation = time.perf_counter()
                    self.times_elapsed_mutation.append(
                        end_time_mutation - start_time_mutation
                    )
                    self.logger.debug(
                        "Time spent finding a valid mutation: {:.2f}s +- {:.2f}".format(
                            np.mean(self.times_elapsed_mutation),
                            np.std(self.times_elapsed_mutation),
                        )
                    )
                    if not self.archive.is_present(individual=mutated_individual):

                        while not mutated_individual.is_evaluated():
                            self.logger.debug(
                                "Evaluating mutated individual at iteration {}".format(
                                    current_iteration
                                )
                            )
                            self.evaluator.run_sim(individual=mutated_individual)

                        mutated_individual_to_save = self.repeat_execution(
                            individual=mutated_individual
                        )
                        self.archive.add_executed_individual(
                            individual=mutated_individual_to_save
                        )
                        fitness_values.append(
                            mutated_individual_to_save.get_fitness().get_value()
                        )
                        self.logger.debug(
                            "Fitness mutated individual: {}".format(
                                mutated_individual_to_save.get_fitness().get_value()
                            )
                        )

                        neighborhood.append(mutated_individual_to_save)

                    else:
                        max_iterations -= 1

                assert (
                    max_iterations > 0
                ), "Mutation produces individuals that are too similar"

                if self.maximize:
                    idx = np.argmax(fitness_values)
                else:
                    idx = np.argmin(fitness_values)

                current_best_fitness = fitness_values[idx]
                self.fitnesses.append(current_best_fitness)
                if current_best_fitness > best_fitness:

                    if self.individual_name == STATE_PAIR_INDIVIDUAL_NAME:
                        candidate_individual = cast(
                            StatePairIndividual, neighborhood[idx]
                        )
                        if (
                            candidate_individual.m1.is_success
                            or candidate_individual.m2.is_success
                        ):
                            individual = cast(StatePairIndividual, neighborhood[idx])
                            best_fitness = current_best_fitness

                            assert individual.are_members_close(), (
                                "The two members of the individual need to be close to each other. "
                                "There must be an error in the mutation operators that must ensure that."
                            )
                            if individual.is_frontier_pair():
                                self.archive.add_target_individual(
                                    individual=individual
                                )
                                self.logger.info(
                                    "{} Restart {}".format("*" * 100, "*" * 100)
                                )
                                return True, False

                    else:
                        individual = neighborhood[idx]
                        best_fitness = current_best_fitness

                        if not individual.is_success():
                            self.archive.add_target_individual(individual=individual)
                            self.logger.info(
                                "{} Restart {}".format("*" * 100, "*" * 100)
                            )
                            return True, False

                    self.logger.info(
                        "Best individual at iteration {} with fitness {}".format(
                            current_iteration, best_fitness
                        )
                    )

                current_iteration += 1
                self.logger.info(
                    "{} End Iteration {} {}".format(
                        "*" * 100, current_iteration, "*" * 100
                    )
                )
        except (AssertionError, RuntimeError, NotImplementedError) as e:
            self.logger.error(e)
            return False, True

        return False, False
