"""
MIT License

Copyright (c) 2020 testingautomated-usi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import List, Tuple

import numpy as np
from self_driving.pose import Pose

List4DTuple = List[Tuple[float, float, float, float]]
List2DTuple = List[Tuple[float, float]]


class RoadPoints:

    @classmethod
    def from_nodes(cls, middle_nodes: List4DTuple):
        res = RoadPoints()
        res.add_middle_nodes(middle_nodes)
        return res

    def __init__(self):
        self.middle: List4DTuple = []
        self.right: List2DTuple = []
        self.left: List2DTuple = []
        self.n = 0

    def add_middle_nodes(self, middle_nodes):
        n = len(self.middle) + len(middle_nodes)

        assert n >= 2, f"At least, two nodes are needed"

        assert all(
            len(point) >= 4 for point in middle_nodes
        ), f"A node is a tuple of 4 elements (x,y,z,road_width)"

        self.n = n
        self.middle += list(middle_nodes)
        self.left += [None] * len(middle_nodes)
        self.right += [None] * len(middle_nodes)
        self._recalculate_nodes()
        return self

    def _recalculate_nodes(self):
        for i in range(self.n - 1):
            l, r = self.calc_point_edges(self.middle[i], self.middle[i + 1])
            self.left[i] = l
            self.right[i] = r

        # the last middle point
        self.right[-1], self.left[-1] = self.calc_point_edges(
            self.middle[-1], self.middle[-2]
        )

    @classmethod
    def calc_point_edges(cls, p1, p2) -> Tuple[Tuple, Tuple]:
        origin = np.array(p1[0:2])

        a = np.subtract(p2[0:2], origin)

        # calculate the vector which length is half the road width
        v = (a / np.linalg.norm(a)) * p1[3] / 2
        # add normal vectors
        l = origin + np.array([-v[1], v[0]])
        r = origin + np.array([v[1], -v[0]])
        return tuple(l), tuple(r)

    def vehicle_start_pose(
        self, meters_from_road_start=2.5, road_point_index=0
    ) -> Pose:
        assert (
            self.n > road_point_index
        ), f"road length is {self.n} it does not have index {road_point_index}"
        p1 = self.middle[road_point_index]
        p1r = self.right[road_point_index]
        p2 = self.middle[road_point_index + 1]

        p2v = np.subtract(p2[0:2], p1[0:2])
        v = (p2v / np.linalg.norm(p2v)) * meters_from_road_start
        origin = np.add(p1[0:2], p1r[0:2]) / 2
        deg = np.degrees(np.arctan2([-v[0]], [-v[1]]))
        res = Pose(pos=tuple(origin + v) + (p1[2],), rot=(0, 0, deg[0]))
        return res

    def new_imagery(self):
        from .road_imagery import RoadImagery

        return RoadImagery(self)

    def plot_on_ax(self, ax):
        def _plot_xy(points, color, linewidth):
            tup = list(zip(*points))
            ax.plot(tup[0], tup[1], color=color, linewidth=linewidth)

        ax.set_facecolor("#7D9051")
        _plot_xy(self.middle, "#FEA952", linewidth=1)
        _plot_xy(self.left, "white", linewidth=1)
        _plot_xy(self.right, "white", linewidth=1)
        ax.axis("equal")
