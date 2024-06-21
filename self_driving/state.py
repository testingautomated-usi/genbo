from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Dict, Callable

from shapely.geometry import Point

from self_driving.bounding_box import BoundingBox
from self_driving.orientation_utils import Quaternion
from self_driving.road import Road


class State(ABC):

    def __init__(
        self,
        road: Road,
        bounding_box: BoundingBox,
        velocity_checker: Callable[[Point], bool],
        orientation_checker: Callable[[Quaternion, Quaternion], bool],
    ):
        self.road = road
        self.bounding_box = bounding_box
        self.velocity_checker = velocity_checker
        self.orientation_checker = orientation_checker

    @abstractmethod
    def get_performance(self, normalize: bool = False) -> float:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def export(self) -> Dict:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def mutate(
        self,
        other_state: "State" = None,
        is_close_constraint: bool = False,
        bias: bool = False,
        previous_state: "State" = None,
    ) -> Tuple["State", bool]:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def parse(self, **kwargs) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def is_valid(self) -> bool:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def is_close_to(self, other: "State") -> bool:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def set_bounding_box(self, bounding_box: BoundingBox) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def to_dict(self) -> Dict:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def update_state(self, check_orientation: bool = True, **kwargs) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_representation(self, csv: bool = False) -> str:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_keys(self) -> List[str]:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_values(self) -> List[Union[str, float, int, bool]]:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def clone(self) -> "State":
        raise NotImplementedError("Not implemented")

    def __str__(self):
        return self.get_representation()
