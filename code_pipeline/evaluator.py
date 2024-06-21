from abc import ABC, abstractmethod
from test.individual import Individual


class Evaluator(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def run_sim(self, individual: Individual) -> None:
        raise NotImplemented("Not implemented")

    @abstractmethod
    def close(self) -> None:
        raise NotImplemented("Not implemented")
