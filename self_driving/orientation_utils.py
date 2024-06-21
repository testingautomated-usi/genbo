import math
from typing import Callable, Tuple, Union

import numpy as np

from config import SIMULATOR_NAMES, DONKEY_SIM_NAME, MOCK_SIM_NAME
from envs.donkey.config import MAX_ORIENTATION_CHANGE_DONKEY, EPS_ORIENTATION_DONKEY


class Quaternion:

    # euclid graphics maths module
    #
    # Copyright (c) 2019, Alex Holkner
    # All rights reserved.

    # Redistribution and use in source and binary forms, with or without
    # modification, are permitted provided that the following conditions are met:

    # 1. Redistributions of source code must retain the above copyright notice, this
    #    list of conditions and the following disclaimer.

    # 2. Redistributions in binary form must reproduce the above copyright notice,
    #    this list of conditions and the following disclaimer in the documentation
    #    and/or other materials provided with the distribution.

    # 3. Neither the name of the copyright holder nor the names of its
    #    contributors may be used to endorse or promote products derived from
    #    this software without specific prior written permission.

    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    def __init__(self, v: np.ndarray):
        assert len(v) == 4, "Dimensions of the array must be 4"
        self.v = v
        self.x = v[0]
        self.y = v[1]
        self.z = v[2]
        self.w = v[3]

    # from https://github.com/aholkner/euclid/blob/master/euclid.py
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)

    # from https://github.com/aholkner/euclid/blob/master/euclid.py
    def normalize(self) -> "Quaternion":
        d = self.magnitude()
        x, y, z, w = self.x, self.y, self.z, self.w
        if d > 0:
            x /= d
            y /= d
            z /= d
            w /= d
        return Quaternion(v=np.asarray([x, y, z, w]))

    # from https://github.com/aholkner/euclid/blob/master/euclid.py#L1405
    def get_euler(self) -> Tuple[float, float, float]:
        t = self.x * self.y + self.z * self.w
        if t > 0.4999:
            yaw = 2 * math.atan2(self.x, self.w)
            pitch = math.pi / 2
            roll = 0
        elif t < -0.4999:
            yaw = -2 * math.atan2(self.x, self.w)
            pitch = -math.pi / 2
            roll = 0
        else:
            sqx = self.x**2
            sqy = self.y**2
            sqz = self.z**2
            yaw = math.atan2(
                2 * self.y * self.w - 2 * self.x * self.z, 1 - 2 * sqy - 2 * sqz
            )
            pitch = math.asin(2 * t)
            roll = math.atan2(
                2 * self.x * self.w - 2 * self.y * self.z, 1 - 2 * sqx - 2 * sqz
            )

        angles = self.normalize_angles(v=np.rad2deg([pitch, yaw, roll]))
        pitch, yaw, roll = angles[0], angles[1], angles[2]

        assert math.isclose(
            pitch, 0.0, rel_tol=1e-2
        ), "Pitch should be close to 0.0. Found {}. Quaternion: {}".format(pitch, self)
        assert math.isclose(
            roll, 0.0, rel_tol=1e-2
        ), "Roll should be close to 0.0. Found {}. Quaternion: {}".format(roll, self)

        return pitch, yaw, roll

    @staticmethod
    def normalize_angles(v: np.ndarray) -> np.ndarray:
        return v % 360

    # from https://gist.github.com/aeroson/043001ca12fe29ee911e#file-myquaternion-cs-L586
    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > 360:
            angle -= 360
        while angle < 0:
            angle += 360
        return angle

    # from http://blog.lexique-du-net.com/index.php?post/Calculate-the-real-difference-between-two-angles-keeping-the-sign
    @staticmethod
    def compute_angle_difference(first_angle: float, second_angle: float) -> float:
        difference = first_angle - second_angle
        while difference < -180:
            difference += 360
        while difference > 180:
            difference -= 360
        return difference

    # from https://github.com/aholkner/euclid/blob/master/euclid.py#L1465
    @staticmethod
    def from_euler(pitch: float, yaw: float, roll: float) -> "Quaternion":

        # euclid graphics maths module
        #
        # Copyright (c) 2019, Alex Holkner
        # All rights reserved.

        # Redistribution and use in source and binary forms, with or without
        # modification, are permitted provided that the following conditions are met:

        # 1. Redistributions of source code must retain the above copyright notice, this
        #    list of conditions and the following disclaimer.

        # 2. Redistributions in binary form must reproduce the above copyright notice,
        #    this list of conditions and the following disclaimer in the documentation
        #    and/or other materials provided with the distribution.

        # 3. Neither the name of the copyright holder nor the names of its
        #    contributors may be used to endorse or promote products derived from
        #    this software without specific prior written permission.

        # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)
        roll_rad = np.deg2rad(roll)

        c1 = math.cos(pitch_rad / 2)
        c2 = math.cos(yaw_rad / 2)
        c3 = math.cos(roll_rad / 2)
        s1 = math.sin(pitch_rad / 2)
        s2 = math.sin(yaw_rad / 2)
        s3 = math.sin(roll_rad / 2)

        w = c1 * c2 * c3 - s1 * s2 * s3

        # from https://github.com/opentk/opentk/blob/549d718eba3df5b22ff9c797472e982610f5e0cc/src/OpenTK.Mathematics/Data/Quaternion.cs#L551
        # MIT License
        #
        # Copyright (c) 2006-2020 Stefanos Apostolopoulos for the Open Toolkit project.
        #
        # Permission is hereby granted, free of charge, to any person obtaining a copy
        # of this software and associated documentation files (the "Software"), to deal
        # in the Software without restriction, including without limitation the rights
        # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        # copies of the Software, and to permit persons to whom the Software is
        # furnished to do so, subject to the following conditions:
        #
        # The above copyright notice and this permission notice shall be included in all
        # copies or substantial portions of the Software.
        #
        # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        # SOFTWARE.
        #
        # Third party licenses may be applicable. These have been disclosed in THIRD_PARTIES.md

        x = s1 * c2 * c3 + c1 * s2 * s3
        y = c1 * s2 * c3 - s1 * c2 * s3
        z = c1 * c2 * s3 + s1 * s2 * c3

        return Quaternion(v=np.asarray([x, y, z, w]))

    # https://forum.unity.com/threads/quaternion-dot-what-is-supposed-to-return.583048/
    def dot(self, other: "Quaternion") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

    # https://forum.unity.com/threads/quaternion-angle-implementation.572632/
    def angle(self, other: "Quaternion") -> float:
        dot_product = self.dot(other=other)
        return math.acos(min(abs(dot_product), 1.0)) * 2.0 * (180 / np.pi)

    def __str__(self):
        return "({}, {}, {}, {})".format(self.x, self.y, self.z, self.w)


def quaternion_to_euler(
    rot_x: float, rot_y: float, rot_z: float, rot_w: float
) -> Tuple[float, float, float]:
    v = np.asarray([0.0, 0.0, 0.0, 0.0])
    v[0] = rot_x
    v[1] = rot_y
    v[2] = rot_z
    v[3] = rot_w
    q = Quaternion(v=v)
    return q.get_euler()


def intersect(angle: float, min_angle: float, max_angle: float) -> bool:
    # https://stackoverflow.com/questions/11775473/check-if-two-segments-on-the-same-circle-overlap-intersect
    # https://creativecommons.org/licenses/by-sa/3.0/
    if min_angle > max_angle:
        if angle >= min_angle or angle <= max_angle:
            return True
    else:
        if min_angle <= angle <= max_angle:
            return True
    return False


def get_higher_orientation(
    current_relative_orientation: float,
    orientation_source: float,
    orientation_value: float,
    range_orientation_values: Tuple[float, float],
    direction: str,
    neighborhood: int = 20,
) -> float:
    # FIXME: for now only random search (a bit slow). Refactor with actual hill climbing
    assert direction in ["plus", "minus"], f"Unknown direction {direction}"
    fitness = current_relative_orientation
    max_iterations_while = 1000
    _orientation_value = orientation_value
    while fitness <= current_relative_orientation and max_iterations_while > 0:
        if direction == "plus":
            l, h = (
                max(range_orientation_values[0], _orientation_value),
                range_orientation_values[-1],
            )
            values = [np.random.uniform(low=l, high=h) for _ in range(neighborhood)]
        else:
            l, h = range_orientation_values[0], min(
                _orientation_value, range_orientation_values[-1]
            )
            if h < l:
                h = min(_orientation_value + 360, range_orientation_values[-1])
            values = [np.random.uniform(low=l, high=h) for _ in range(neighborhood)]

        fitness_values = [
            abs(get_angle_difference(source=orientation_source, target=value))
            for value in values
        ]
        indices_fitness_greater_current = [
            idx
            for idx, fitness_value in enumerate(fitness_values)
            if fitness_value > fitness
            and abs(fitness_value - fitness) <= EPS_ORIENTATION_DONKEY
        ]

        idx_max_fitness = np.argmax(fitness_values)
        if (
            len(indices_fitness_greater_current) == 0
            and fitness_values[idx_max_fitness] > fitness
        ):
            indices_sorted = np.argsort(fitness_values)
            idx = indices_sorted[0]
            fitness = fitness_values[idx]
            _orientation_value = values[idx]
        elif len(indices_fitness_greater_current) > 0:
            idx = np.random.choice(indices_fitness_greater_current)
            fitness = fitness_values[idx]
            _orientation_value = values[idx]

        max_iterations_while -= 1

    if max_iterations_while == 0:
        return -1.0

    return _orientation_value


def get_angle_difference(
    source: Union[Quaternion, int, float], target: Union[Quaternion, int, float]
) -> float:
    if (isinstance(source, float) or isinstance(source, int)) and (
        isinstance(target, float) or isinstance(target, int)
    ):
        return Quaternion.compute_angle_difference(
            first_angle=source, second_angle=target
        )
    elif isinstance(source, Quaternion) and isinstance(target, Quaternion):
        pitch_source, yaw_source, roll_source = quaternion_to_euler(
            rot_x=source.x, rot_y=source.y, rot_z=source.z, rot_w=source.w
        )
        pitch_target, yaw_target, roll_target = quaternion_to_euler(
            rot_x=target.x, rot_y=target.y, rot_z=target.z, rot_w=target.w
        )
        return Quaternion.compute_angle_difference(
            first_angle=yaw_source, second_angle=yaw_target
        )

    raise RuntimeError("Source and target are not of the same type")


def get_orientation_checker(env_name: str) -> Callable[[Quaternion, Quaternion], bool]:
    assert (
        env_name in SIMULATOR_NAMES
    ), "Env name {} not supported. Choose between: {}".format(env_name, SIMULATOR_NAMES)

    if env_name == DONKEY_SIM_NAME or env_name == MOCK_SIM_NAME:
        max_orientation_change = MAX_ORIENTATION_CHANGE_DONKEY

        def _check_orientation(source: Quaternion, target: Quaternion) -> bool:
            diff = abs(get_angle_difference(source=source, target=target))
            return diff < max_orientation_change or math.isclose(
                diff, max_orientation_change, rel_tol=0.01
            )

        return _check_orientation

    raise RuntimeError("Unknown env name: {}".format(env_name))
