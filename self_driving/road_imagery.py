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

import matplotlib.pyplot as plt

from self_driving.road_points import RoadPoints


class RoadImagery:
    def __init__(self, road_points: RoadPoints):
        self.road_points = road_points
        self._fig, self._ax = None, None

    def plot(self):
        self._close()
        self._fig, self._ax = plt.subplots(1)
        self.road_points.plot_on_ax(self._ax)
        self._ax.axis("equal")

    def save(self, image_path):
        if not self._fig:
            self.plot()
        self._fig.savefig(image_path)

    @classmethod
    def from_sample_nodes(cls, sample_nodes):
        return RoadImagery(RoadPoints().add_middle_nodes(sample_nodes))

    def _close(self):
        if self._fig:
            plt.close(self._fig)
            self._fig = None
            self._ax = None

    def __del__(self):
        self._close()
