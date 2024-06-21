from abc import ABC, abstractmethod
from typing import List, Dict, Any

import numpy as np

from test.fitness import Fitness
from test.id_generator import IdGenerator


class Individual(ABC):

    def __init__(self, start_id: int = 1):
        self.id = IdGenerator.get_instance(start_count=start_id).get_id()
        self.fitness: Fitness = None
        self.observations = []
        self.actions = []
        self.behavioural_metrics = dict()
        self.start_id = start_id
        self._is_success = None
        self.replicable_percentage = -1

    @abstractmethod
    def clone(self) -> "Individual":
        raise NotImplemented("Not implemented")

    @abstractmethod
    def get_implementation(self) -> Any:
        raise NotImplemented("Not implemented")

    @abstractmethod
    def get_representation(self) -> str:
        raise NotImplemented("Not implemented")

    @abstractmethod
    def parse(self, individual_export: Dict) -> None:
        raise NotImplemented("Not implemented")

    @abstractmethod
    def export(self) -> Dict:
        raise NotImplemented("Not implemented")

    def export_common(self, export_dict: Dict) -> Dict:
        result = dict()
        result["id"] = self.id
        result["fitness"] = (self.fitness.name, self.fitness.get_value())

        for key in export_dict.keys():
            result[key] = export_dict[key]

        if len(self.behavioural_metrics) > 0:
            result["behavioural_metrics"] = self.behavioural_metrics

        if self.replicable_percentage != -1:
            result["replicable_percentage"] = self.replicable_percentage

        return result

    @abstractmethod
    def mutate(self, bias: bool = False, mutation_extent: int = 1) -> "Individual":
        raise NotImplemented("Not implemented")

    def reset(self) -> None:
        self.fitness = None
        self.observations = []
        self.actions = []
        self.behavioural_metrics = dict()
        self._is_success = None

    def is_evaluated(self) -> bool:
        return self.fitness is not None

    def get_fitness(self) -> Fitness:
        assert self.fitness is not None, "Fitness has not been set"
        return self.fitness

    def set_fitness(self, fitness: Fitness) -> None:
        self.fitness = fitness

    def set_success(self, is_success: bool) -> None:
        self._is_success = is_success

    def is_success(self) -> bool:
        assert self._is_success is not None, "Is success not set"
        return self._is_success

    def get_behavioural_metrics(self) -> Dict:
        assert len(self.behavioural_metrics) > 0, "Behavioural metrics not set"
        return self.behavioural_metrics

    def set_behavioural_metrics(
        self,
        speeds: List[float],
        steering_angles: List[float],
        lateral_positions: List[float],
        ctes: List[float],
    ) -> None:
        self.behavioural_metrics["speeds"] = speeds
        self.behavioural_metrics["steering_angles"] = steering_angles
        self.behavioural_metrics["lateral_positions"] = lateral_positions
        self.behavioural_metrics["ctes"] = ctes

    def set_observations(self, observations: List[np.ndarray]) -> None:
        self.observations = observations

    def get_observations(self) -> List[np.ndarray]:
        assert len(self.observations) > 0, "Observations not set"
        return self.observations

    def set_actions(self, actions: List[np.ndarray]) -> None:
        self.actions = actions

    def get_actions(self) -> List[np.ndarray]:
        assert len(self.actions) > 0, "Actions not set"
        return self.actions
