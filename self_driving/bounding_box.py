from abc import ABC, abstractmethod
from typing import List

from shapely.geometry import Point, Polygon

from self_driving.waypoint import Waypoint


class BoundingBox(ABC):

    def __init__(self, waypoints: List[Waypoint], road_width: float):
        self.waypoints = waypoints
        self.road_width = road_width

    @abstractmethod
    def clone(self) -> "BoundingBox":
        raise NotImplemented("Not implemented")

    @abstractmethod
    def is_in(self, point: Point) -> bool:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def build_bounding_box(
        self, waypoints: List[Waypoint] = None, road_width: int = None
    ) -> Polygon:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def plot_bounding_box(
        self, point: Point = None, waypoints: List[Waypoint] = None
    ) -> None:
        raise NotImplementedError("Not implemented")
