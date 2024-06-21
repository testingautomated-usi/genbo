from shapely.geometry import Point

from self_driving.orientation_utils import Quaternion


class Waypoint:

    def __init__(self, position: Point, rotation: Quaternion):
        self.position = position
        # meant to be accessed only through its getter
        self._rotation = rotation

    def get_rotation(self) -> Quaternion:
        assert self._rotation is not None
        return self._rotation

    def __str__(self):
        return f"Pos: {self.position}, rotation: {self._rotation}"
