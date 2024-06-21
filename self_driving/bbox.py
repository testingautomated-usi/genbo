"""
]
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

from shapely.geometry import Point, box, LineString, Polygon
from typing import List, Tuple
from self_driving.road_polygon import RoadPolygon


class RoadBoundingBox:
    """A class representing the bounding box that contains the road."""

    def __init__(self, bbox_size: Tuple[float, float, float, float]):
        assert len(bbox_size) == 4
        self.bbox = box(*bbox_size)

    def intersects_sides(self, point: Point) -> bool:
        for side in self.get_sides():
            if side.intersects(point):
                return True
        return False

    def intersects_vertices(self, point: Point) -> bool:
        for vertex in self.get_vertices():
            if vertex.intersects(point):
                return True
        return False

    def intersects_boundary(self, other: Polygon) -> bool:
        return other.intersects(self.bbox.boundary)

    def contains(self, other: RoadPolygon) -> bool:
        return self.bbox.contains(other.polyline)

    def get_sides(self) -> List[LineString]:
        sides = []
        xs, ys = self.bbox.exterior.coords.xy
        xys = list(zip(xs, ys))
        for p1, p2 in zip(xys[:-1], xys[1:]):
            sides.append(LineString([p1, p2]))
        return sides

    def get_vertices(self) -> List[Point]:
        xs, ys = self.bbox.exterior.coords.xy
        xys = list(zip(xs, ys))
        return [Point(xy) for xy in xys]
