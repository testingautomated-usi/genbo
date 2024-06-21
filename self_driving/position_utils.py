import math
from enum import Enum
from typing import Tuple

import numpy as np
from euclid import Vector3
from shapely.geometry import Point

from envs.donkey.config import EPS_POSITION_DONKEY_GENERATED
from self_driving.road import Road


class Line3d:
    """
    Copyright (c) 2017, Tawn Kramer
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
        * Neither the name of the <organization> nor the
          names of its contributors may be used to endorse or promote products
          derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
    """

    def __init__(self, a: Vector3, b: Vector3):
        self.m_origin = a
        self.m_dir = a - b
        self.m_dir.normalize()

    # produce a vector normal to this line passing through this point
    def closest_vector_to(self, v: Vector3) -> Vector3:
        delta_point = self.m_origin - v
        dot = delta_point.dot(other=self.m_dir)
        return (self.m_dir * dot) - delta_point

    # transform the point by the normal vector that places it on the line
    def closest_point_on_line_to(self, v: Vector3) -> Vector3:
        vector_to = self.closest_vector_to(v=v)
        return v - vector_to

    def abs_angle_between(self, line3d: "Line3d") -> float:
        return math.fabs(self.m_dir.angle(other=line3d.m_dir))


class SegResult(Enum):
    on_span = 1
    less_than_origin = 2
    greater_than_end = 3


class LineSeg3d(Line3d):
    """
    Copyright (c) 2017, Tawn Kramer
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
        * Neither the name of the <organization> nor the
          names of its contributors may be used to endorse or promote products
          derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
    """

    def __init__(self, a: Vector3, b: Vector3):
        super().__init__(a=a, b=b)
        self.m_end = b
        self.m_length = (a - b).magnitude

    # find the closest point, clamping it to the ends
    def closest_point_on_segment_to(self, v: Vector3) -> Tuple[Vector3, SegResult]:
        delta_point = self.m_origin - v
        dot = delta_point.dot(self.m_dir)

        # clamp to the ends of the line segment
        if dot <= 0.0:
            return self.m_origin, SegResult.less_than_origin

        if dot >= self.m_length:
            return self.m_end, SegResult.greater_than_end

        vector_to = (self.m_dir * dot) - delta_point
        return v - vector_to, SegResult.on_span


def compute_cte(position: Point, road: Road) -> float:
    """
    Copyright (c) 2017, Tawn Kramer
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
        * Neither the name of the <organization> nor the
          names of its contributors may be used to endorse or promote products
          derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
    """
    lookahead = 1
    vector_position = Vector3(x=position.x, y=position.y, z=position.z)
    active_road_point = road.get_closest_road_point_index(point=position)
    ahead_active_road_point = (active_road_point + lookahead) % len(road.road_points)
    v_a = Vector3(
        x=road.road_points[active_road_point].x,
        y=road.road_points[active_road_point].y,
        z=road.road_points[active_road_point].z,
    )
    v_c = Vector3(
        x=road.road_points[ahead_active_road_point].x,
        y=road.road_points[ahead_active_road_point].y,
        z=road.road_points[ahead_active_road_point].z,
    )
    path_seg = LineSeg3d(a=v_a, b=v_c)
    err_vec = path_seg.closest_vector_to(v=vector_position)
    sign = 1
    cp = path_seg.m_dir.normalized().cross(other=err_vec.normalized())

    if cp.y > 0.0:
        sign = -1

    return err_vec.magnitude() * sign


def get_higher_cte(
    curr_pos_x: float,
    curr_pos_y: float,
    curr_pos_z: float,
    curr_cte: float,
    val_a: float,
    val_b: float,
    val_c: float,
    index: int,
    change_x: bool,
    road: Road,
    neighborhood_size: int = 20,
) -> Tuple[float, float]:
    assert index in [0, 1, 2, 3, 4], f"Unknown index: {index}"
    # FIXME: for now only random search (a bit slow). Refactor with actual hill climbing
    fitness = curr_cte
    max_iterations_while = 100
    _curr_pos_x = curr_pos_x
    _curr_pos_z = curr_pos_z
    while fitness <= curr_cte and max_iterations_while > 0:
        values_x, values_z = [], []
        for _ in range(neighborhood_size):
            if change_x:
                if index == 0 or index == 1:
                    val_x = np.random.uniform(low=val_a - val_c, high=val_a + val_c)
                    val_k = np.sqrt(
                        -(val_a**2) + 2 * val_a * val_x + val_c**2 - val_x**2
                    )
                    val_z = np.random.uniform(low=val_b - val_k, high=val_b + val_k)

                elif index == 2 or index == 3:
                    val_x = np.random.uniform(low=val_a - val_c, high=val_a + val_c)
                    val_k = np.sqrt(
                        -(val_a**2) + 2 * val_a * val_x + val_c**2 - val_x**2
                    )
                    val_z = np.random.uniform(low=val_b - val_k, high=val_b + val_k)

                else:
                    raise RuntimeError(f"Unknown index: {index}")
            else:
                if index == 0 or index == 1:
                    val_z = np.random.uniform(low=val_b - val_c, high=val_b + val_c)
                    val_k = np.sqrt(
                        -(val_b**2) + 2 * val_b * val_z + val_c**2 - val_z**2
                    )
                    val_x = np.random.uniform(low=val_a - val_k, high=val_a + val_k)

                elif index == 2 or index == 3:
                    val_z = np.random.uniform(low=val_b - val_c, high=val_b + val_c)
                    val_k = np.sqrt(
                        -(val_b**2) + 2 * val_b * val_z + val_c**2 - val_z**2
                    )
                    val_x = np.random.uniform(low=val_a - val_k, high=val_a + val_k)

                else:
                    raise RuntimeError(f"Unknown index: {index}")

            values_x.append(val_x)
            values_z.append(val_z)

        fitness_values = [
            abs(
                compute_cte(
                    position=Point(values_x[idx], curr_pos_y, values_z[idx]), road=road
                )
            )
            for idx in range(len(values_x))
        ]
        idx_max_fitness = np.argmax(fitness_values)
        indices_fitness_greater_current = [
            idx
            for idx, fitness_value in enumerate(fitness_values)
            if fitness_value > fitness
            and abs(fitness_value - fitness) <= EPS_POSITION_DONKEY_GENERATED
        ]

        if (
            len(indices_fitness_greater_current) == 0
            and fitness_values[idx_max_fitness] > fitness
        ):
            indices_sorted = np.argsort(fitness_values)
            idx = indices_sorted[0]
            fitness = fitness_values[idx]
            _curr_pos_x = values_x[idx]
            _curr_pos_z = values_z[idx]
        elif len(indices_fitness_greater_current) > 0:
            idx = np.random.choice(indices_fitness_greater_current)
            fitness = fitness_values[idx]
            _curr_pos_x = values_x[idx]
            _curr_pos_z = values_z[idx]

        max_iterations_while -= 1

    if max_iterations_while == 0:
        return -1.0, -1.0

    return _curr_pos_x, _curr_pos_z
