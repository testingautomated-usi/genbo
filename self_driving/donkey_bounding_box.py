import copy
from typing import List

from shapely.geometry import Point, Polygon
import shapely.geometry as shp
import numpy as np

import matplotlib.pyplot as plt

from envs.donkey.donkey_env_utils import make_simulator_scene
from envs.donkey.scenes.simulator_scenes import (
    SANDBOX_LAB_NAME,
    GENERATED_TRACK_NAME,
    SimulatorScene,
)
from self_driving.bounding_box import BoundingBox
from self_driving.donkey_constant_road import DonkeyConstantRoad
from self_driving.waypoint import Waypoint


class DonkeyBoundingBox(BoundingBox):

    def __init__(
        self,
        waypoints: List[Waypoint],
        road_width: float,
        simulator_scene: SimulatorScene,
    ):
        super().__init__(waypoints=waypoints, road_width=road_width)
        assert simulator_scene is not None, "Simulator scene cannot be None"
        self.simulator_scene = simulator_scene

    def clone(self) -> "BoundingBox":
        new_bounding_box = DonkeyBoundingBox(
            waypoints=copy.deepcopy(self.waypoints),
            road_width=self.road_width,
            simulator_scene=self.simulator_scene,
        )
        return new_bounding_box

    def is_in(self, point: Point) -> bool:
        assert point.has_z, "Point {} has no z component".format(point)
        poly = self.build_bounding_box()
        # strip y component, i.e., height
        _point = Point(point.x, point.z)
        return poly.contains(other=_point)

    def plot_bounding_box(
        self, point: Point = None, waypoints: List[Waypoint] = None
    ) -> None:
        assert point.has_z, "Point {} has no z component".format(point)
        # strip y component, i.e., height
        _point = Point(point.x, point.z)

        poly = self.build_bounding_box()

        # Turn polygon points into numpy arrays for plotting
        poffafpolypts = np.array(poly.exterior)

        # Plot points
        plt.rcParams["axes.facecolor"] = "white"
        plt.plot(*poffafpolypts.T, color="blue", linestyle="-")

        # If there are any Interiors
        # Retrieve coordinates for all interiors
        for inner in poly.interiors:
            xi, yi = zip(*inner.coords[:])
            plt.plot(xi, yi, color="red")

        if waypoints is not None:
            for i, waypoint in enumerate(waypoints):
                plt.scatter(
                    waypoint.position.x, waypoint.position.z, s=20, color="black"
                )
                plt.annotate(str(i), (waypoint.position.x, waypoint.position.z))

        if point is not None:
            plt.scatter(_point.x, _point.y, s=20, color="green")

        plt.axis("equal")
        plt.show()

    def build_bounding_box(
        self, waypoints: List[Waypoint] = None, road_width: int = None
    ) -> Polygon:

        _waypoints = waypoints if waypoints is not None else self.waypoints
        _road_width = road_width if road_width is not None else self.road_width

        poly_through_checkpoints = shp.Polygon(
            [[p.position.x, p.position.z] for p in _waypoints]
        )

        if self.simulator_scene.get_scene_name() == SANDBOX_LAB_NAME:
            # One lane track

            # Create offset airfoils, both inward and outward
            poly_outward_offset = poly_through_checkpoints.buffer(
                _road_width
            )  # Outward offset
            poly_inward_offset = poly_through_checkpoints.buffer(
                -_road_width
            )  # Inward offset

            poly = Polygon(
                poly_outward_offset.exterior,
                holes=[np.array(poly_inward_offset.exterior)],
            )
        elif self.simulator_scene.get_scene_name() == GENERATED_TRACK_NAME:
            # Two lanes track; valid is the internal one

            # Create offset airfoils, both inward and outward
            # The car starts in the middle of the internal lane at position 125. On the outside it has 1.6 meters
            # before the car is in the middle of the yellow line separating the two lanes (i.e., at 124.4); on the
            # inside it has 1.6 meters before the car is in the middle of the white line separating the road from
            # the grass (i.e., at 126.6).
            poly_outward_offset = poly_through_checkpoints.buffer(1.6)  # Outward offset
            poly_inward_offset = poly_through_checkpoints.buffer(-1.6)  # Inward offset

            poly = Polygon(
                poly_outward_offset.exterior,
                holes=[np.array(poly_inward_offset.exterior)],
            )
        else:
            raise RuntimeError(
                "Unknown simulator scene: {}".format(
                    self.simulator_scene.get_scene_name()
                )
            )

        return poly


if __name__ == "__main__":

    # scene_name = SANDBOX_LAB_NAME
    scene_name = GENERATED_TRACK_NAME
    track_num = 0
    simulator_scene = make_simulator_scene(scene_name=scene_name, track_num=track_num)
    donkey_constant_road = DonkeyConstantRoad(simulator_scene=simulator_scene)
    bounding_box = DonkeyBoundingBox(
        waypoints=donkey_constant_road.get_waypoints(),
        road_width=donkey_constant_road.road_width,
        simulator_scene=simulator_scene,
    )
    # point_out = Point(5.500693, 3.134553)
    # print(bounding_box.is_in(point=point_out))
    # bounding_box.plot_bounding_box(point=point_out)
    # print(donkey_constant_road.get_closest_control_point(point=point_out))

    # point_in = Point(1.500693, 0.61)
    # print(bounding_box.is_in(point=point_in))
    # bounding_box.plot_bounding_box(point=point_in)
    # print(donkey_constant_road.get_closest_control_point(point=point_in))

    point = Point(132.3431, 0.5991301, 52.70278)
    print(bounding_box.is_in(point=point))
    bounding_box.plot_bounding_box(
        point=point, waypoints=donkey_constant_road.get_waypoints()
    )
