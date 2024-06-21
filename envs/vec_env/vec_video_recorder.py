"""
The MIT License

Copyright (c) 2019 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import glob
import os
from typing import Callable

from gym.wrappers.monitoring import video_recorder

from envs.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from envs.vec_env.dummy_vec_env import DummyVecEnv


class VecVideoRecorder(VecEnvWrapper):
    """
    Wraps a VecEnv or VecEnvWrapper object to record rendered image as mp4 video.
    It requires ffmpeg or avconv to be installed on the machine.
    :param venv:
    :param video_folder: Where to save videos
    :param record_video_trigger: Function that defines when to start recording.
                                        The function takes the current number of step,
                                        and returns whether we should start recording or not.
    :param name_prefix: Prefix to the video name
    """

    def __init__(
        self,
        venv: VecEnv,
        video_folder: str,
        record_video_trigger: Callable[[int], bool],
        name_prefix: str = "rl-video",
    ):
        VecEnvWrapper.__init__(self, venv)

        self.env = venv
        # Temp variable to retrieve metadata
        temp_env = venv

        # Unwrap to retrieve metadata dict
        # that will be used by gym recorder
        while isinstance(temp_env, VecEnvWrapper):
            temp_env = temp_env.venv

        if isinstance(temp_env, DummyVecEnv):
            metadata = temp_env.get_attr("metadata")[0]
        else:
            metadata = temp_env.metadata

        self.env.metadata = metadata

        self.record_video_trigger = record_video_trigger
        self.video_recorder = None

        self.video_folder = os.path.join(os.path.abspath(video_folder), name_prefix)
        os.makedirs(name=self.video_folder, exist_ok=True)
        self.step_id = 0

        self.recording = False
        self.recorded_frames = 0

        self.id = -1  # set externally

    def reset(self, individual=None, skip_generation=None) -> VecEnvObs:
        obs = self.venv.reset(individual=individual, skip_generation=skip_generation)
        self.start_video_recorder()
        return obs

    def start_video_recorder(self) -> None:
        self.close_video_recorder()

        assert self.id != -1, "id not set"

        video_filename = f"individual_id_{self.id}"

        video_file_paths = glob.glob(
            os.path.join(self.video_folder, f"{video_filename}_*")
        )
        if len(video_file_paths) == 0:
            video_filename += "_0"
        else:
            max_run_id = max(
                [
                    int(video_file_path.split(os.sep)[-1].split("_")[-1].split(".")[0])
                    for video_file_path in video_file_paths
                ]
            )
            video_filename += f"_{max_run_id + 1}"

        base_path = os.path.join(self.video_folder, video_filename)
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env, base_path=base_path, metadata={"step_id": self.step_id}
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self) -> bool:
        return self.record_video_trigger(self.step_id)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.venv.step_wait()

        self.step_id += 1
        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
        elif self._video_enabled():
            self.start_video_recorder()

        return obs, rews, dones, infos

    def close_video_recorder(self) -> None:
        if self.recording:
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 1

    def close(self) -> None:
        VecEnvWrapper.close(self)
        self.close_video_recorder()
        # remove meta.json files
        meta_json_filepaths = glob.glob(os.path.join(self.video_folder, "*.meta.json"))
        for meta_json_filepath in meta_json_filepaths:
            os.remove(meta_json_filepath)

        # remove last video which is spurious
        video_filepaths = glob.glob(os.path.join(self.video_folder, "*.mp4"))
        sorted_video_filepaths = sorted(
            video_filepaths,
            key=lambda video_file_path: os.path.getctime(video_file_path),
            reverse=True,
        )
        os.remove(sorted_video_filepaths[0])

    def exit_scene(self) -> None:
        self.venv.exit_scene()

    def close_connection(self) -> None:
        self.venv.close_connection()
