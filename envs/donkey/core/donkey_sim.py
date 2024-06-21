"""
MIT License

Copyright (c) 2018 Roma Sokolkov
Copyright (c) 2018 Antonin Raffin

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

# Original author: Tawn Kramer
import base64
import os.path
import time
from io import BytesIO
from typing import List, Tuple, cast
import numpy as np
from PIL import Image

from config import (
    DONKEY_SIM_NAME,
    REFERENCE_TRACE_FILENAME,
    REFERENCE_TRACE_CONSTANT_HEADER,
)
from envs.donkey.config import (
    INPUT_DIM,
    MAX_CTE_ERROR_DONKEY,
    DONKEY_REFERENCE_TRACE_HEADER,
    MAX_EPISODE_STEPS_DONKEY_SANDBOX,
    MAX_EPISODE_STEPS_DONKEY_GENERATED_TRACK,
)
from envs.donkey.core.fps import FPSTimer
from envs.donkey.core.message import IMesgHandler
from envs.donkey.core.sim_client import SimClient
from envs.donkey.scenes.simulator_scenes import (
    SimulatorScene,
    SANDBOX_LAB_NAME,
    GENERATED_TRACK_NAME,
    GeneratedTrack,
)
from global_log import GlobalLog, CSVLogger
from self_driving.road import Road
from self_driving.state import State
from test.individual import Individual
from test.state_individual import StateIndividual
from test.state_pair_individual import StatePairIndividual
from test_generators.constant_road_test_generator import ConstantRoadTestGenerator
from test_generators.road_test_generator import RoadTestGenerator
from custom_types import ObserveData


class DonkeyUnitySimController:
    """
    Wrapper for communicating with unity simulation.
    """

    def __init__(
        self,
        seed: int,
        port: int,
        socket_local_address: int,
        simulator_scene: SimulatorScene,
        road_test_generator: RoadTestGenerator = None,
        collect_trace: bool = False,
        max_steps: int = None,
    ):
        self.port = port
        self.address = ("127.0.0.1", port)
        self.socket_local_address = socket_local_address

        # Socket message handler
        self.handler = DonkeyUnitySimHandler(
            road_test_generator=road_test_generator,
            seed=seed,
            simulator_scene=simulator_scene,
            collect_trace=collect_trace,
            max_steps=max_steps,
        )

        self.client = SimClient(self.address, self.socket_local_address, self.handler)
        self.logger = GlobalLog("DonkeyUnitySimController")

    def close_connection(self):
        return self.client.handle_close()

    def wait_until_loaded(self):
        """
        Wait for a client (Unity simulator).
        """
        sleep_time = 0
        while not self.handler.loaded:
            time.sleep(0.1)
            sleep_time += 0.1
            if sleep_time > 3:
                self.logger.info(
                    "Waiting for sim to start..."
                    "if the simulation is running, press EXIT to go back to the menu"
                )
        # self.regen_track()

    def reset(self, skip_generation: bool = False, individual: Individual = None):
        self.handler.reset(skip_generation=skip_generation, individual=individual)

    def regen_track(self):
        self.handler.generate_track()

    def seed(self, seed):
        self.handler.seed = seed

    def get_sensor_size(self):
        """
        :return: (int, int, int)
        """
        return self.handler.get_sensor_size()

    def take_action(self, action):
        self.handler.take_action(action)

    def observe(self) -> ObserveData:
        """
        :return: (np.ndarray)
        """
        return self.handler.observe()

    def quit(self):
        self.logger.info("Stopping client")
        self.client.stop()

    def render(self, mode):
        pass

    def is_game_over(self):
        return self.handler.is_game_over()


class DonkeyUnitySimHandler(IMesgHandler):
    """
    Socket message handler.
    """

    def __init__(
        self,
        seed: int,
        simulator_scene: SimulatorScene,
        road_test_generator: RoadTestGenerator = None,
        collect_trace: bool = False,
        max_steps: int = None,
    ):

        self.logger = GlobalLog("DonkeyUnitySimHandler")
        self.road_test_generator = road_test_generator
        self.is_success = 0

        self.loaded = False
        self.control_timer = FPSTimer(timer_name="control", verbose=1)
        self.observation_timer = FPSTimer(timer_name="observation", verbose=1)
        self.max_cte_error = MAX_CTE_ERROR_DONKEY
        self.seed = seed
        self.simulator_scene = simulator_scene

        self.collect_trace = collect_trace
        self.max_steps = max_steps

        # sensor size - height, width, depth
        self.camera_img_size = INPUT_DIM
        self.image_array = np.zeros(self.camera_img_size)
        self.original_image = None
        self.last_obs = None
        self.last_throttle = 0.0
        # Disabled: hit was used to end episode when bumping into an object
        self.hit = "none"
        # Cross track error
        self.cte = 0.0
        self.cte_pid = 0.0
        self.prev_cte = 0.0

        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0

        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0

        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0
        self.rot_w = 0.0

        self.rotation_angle = 0.0

        self.forward_x = 0.0
        self.forward_y = 0.0
        self.forward_z = 0.0

        self.total_nodes = 0
        self.active_node = 0
        # self.end_lap = False
        self.current_track_string = "none"
        self.current_track = None
        self.start_time = time.time()

        self.episode_counter = 0
        self.current_episode_counter_in_log = 0

        self.car_trajectory: List[Tuple[float, float]] = []
        self.images: List[str] = []
        self.actions: List[Tuple[float, float]] = []
        self.agent_state: dict = dict()
        self.reconstruction_losses: List[float] = []
        self.steerings: List[float] = []
        self.speeds: List[float] = []
        self.ctes: List[float] = []

        self.steering_angle = 0.0
        self.current_step = 0
        self.total_steps = 0
        self.speed = 0
        self.steering = 0.0
        self.last_steering = 0.0

        self.track_strings = []
        self.start = True
        self.frame_count = 0

        self.is_paused = False
        # Define which method should be called
        # for each type of message
        self.fns = {
            "telemetry": self.on_telemetry,
            "scene_selection_ready": self.on_scene_selection_ready,
            "scene_names": self.on_recv_scene_names,
            "car_loaded": self.on_car_loaded,
            "send_track": self.on_recv_track,
            "need_car_config": self.on_need_car_config,
        }

        self.reference_trace_filepath = None
        self.reference_trace_filename = REFERENCE_TRACE_FILENAME
        self.csv_logger = None
        self.header = "{},{}".format(
            REFERENCE_TRACE_CONSTANT_HEADER, DONKEY_REFERENCE_TRACE_HEADER
        )
        self.csv_log_lines = []
        if self.collect_trace:
            self.reference_trace_filepath = os.path.join(
                "logs", DONKEY_SIM_NAME, simulator_scene.get_scene_name()
            )
            if self.simulator_scene.get_scene_name() == GENERATED_TRACK_NAME:
                _simulator_scene = cast(GeneratedTrack, self.simulator_scene)
                if _simulator_scene.track_num is not None:
                    self.reference_trace_filename = REFERENCE_TRACE_FILENAME.split(".")[
                        0
                    ] + "_{}.{}".format(
                        _simulator_scene.track_num,
                        REFERENCE_TRACE_FILENAME.split(".")[1],
                    )

            if os.path.exists(
                os.path.join(
                    self.reference_trace_filepath, self.reference_trace_filename
                )
            ):
                self.logger.warn(
                    "Removing reference trace file: {}".format(
                        os.path.join(
                            self.reference_trace_filepath, self.reference_trace_filename
                        )
                    )
                )
                os.remove(
                    os.path.join(
                        self.reference_trace_filepath, self.reference_trace_filename
                    )
                )
            else:
                os.makedirs(name=self.reference_trace_filepath, exist_ok=True)

            self.csv_logger = CSVLogger(
                header=self.header,
                path_to_csv_file=os.path.join(
                    self.reference_trace_filepath, self.reference_trace_filename
                ),
            )

    def send_cam_config(
        self,
        img_w=0,
        img_h=0,
        img_d=0,
        img_enc=0,
        fov=0,
        fish_eye_x=0,
        fish_eye_y=0,
        offset_x=0,
        offset_y=0,
        offset_z=0,
        rot_x=0,
    ):
        """Camera config
        set any field to Zero to get the default camera setting.
        offset_x moves camera left/right
        offset_y moves camera up/down
        offset_z moves camera forward/back
        rot_x will rotate the camera
        with fish_eye_x/y == 0.0 then you get no distortion
        img_enc can be one of JPG|PNG|TGA
        """
        msg = {
            "msg_type": "cam_config",
            "fov": str(fov),
            "fish_eye_x": str(fish_eye_x),
            "fish_eye_y": str(fish_eye_y),
            "img_w": str(img_w),
            "img_h": str(img_h),
            "img_d": str(img_d),
            "img_enc": str(img_enc),
            "offset_x": str(offset_x),
            "offset_y": str(offset_y),
            "offset_z": str(offset_z),
            "rot_x": str(rot_x),
        }
        self.blocking_send(msg)
        time.sleep(0.1)

    def blocking_send(self, msg):
        if self.client is None:
            print(f"skiping: \n {msg}")
            return
        self.client.send_now(msg)

    def send_config(self):
        print("sending car config.")
        self.send_cam_config()
        print("done sending car config.")

    def on_need_car_config(self):
        print("on need car config")
        self.loaded = True
        self.send_config()

    def on_connect(self, client):
        """
        :param client: (client object)
        """
        self.client = client

    def on_disconnect(self):
        """
        Close client.
        """
        self.client = None

    def on_abort(self):
        self.client.stop()

    def on_recv_message(self, message):
        """
        Distribute the received message to the appropriate function.

        :param message: (dict)
        """
        if "msg_type" not in message:
            # print('Expected msg_type field')
            return

        msg_type = message["msg_type"]
        if msg_type in self.fns:
            self.fns[msg_type](message)
        else:
            print("Unknown message type", msg_type)

    def reset(self, skip_generation: bool = False, individual: Individual = None):
        """
        Global reset, notably it
        resets car to initial position.
        """

        self.image_array = np.zeros(self.camera_img_size)
        self.last_obs = None
        self.hit = "none"
        self.steering = 0.0
        self.last_steering = 0.0
        self.cte = 0.0
        self.cte_pid = 0.0
        self.prev_cte = 0.0

        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0

        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0

        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0
        self.rot_w = 0.0

        self.rotation_angle = 0.0

        self.forward_x = 0.0
        self.forward_y = 0.0
        self.forward_z = 0.0

        self.actions.clear()
        self.images.clear()
        self.car_trajectory.clear()
        self.is_success = 0
        self.speeds.clear()
        self.steerings.clear()
        self.ctes.clear()

        self.current_step = 0
        self.total_nodes = 0
        # self.end_lap = False
        self.active_node = 0

        # reference trace should be long; this is to avoid that spurious log lines are written
        # to the csv file when the environment is shut down
        if len(self.csv_log_lines) > 100:
            for csv_log_line in self.csv_log_lines:
                self.csv_logger.log(msg=csv_log_line)

        self.csv_log_lines.clear()

        if not skip_generation and individual is None:
            if self.simulator_scene.get_scene_name() == GENERATED_TRACK_NAME:
                assert (
                    self.road_test_generator is not None
                ), "Test generator is not instantiated"
                simulator_scene = cast(GeneratedTrack, self.simulator_scene)
                if (
                    not isinstance(self.road_test_generator, ConstantRoadTestGenerator)
                    or simulator_scene.track_num is not None
                ):
                    self.generate_track()
        elif individual is not None:
            if isinstance(individual, StateIndividual) or isinstance(
                individual, StatePairIndividual
            ):
                self.generate_track()
                self.send_reset_state(individual=individual)
            else:
                raise RuntimeError(
                    "Individual of type {} not supported".format(type(individual))
                )

        if individual is None or (
            not isinstance(individual, StateIndividual)
            and not isinstance(individual, StatePairIndividual)
        ):
            self.send_reset_car()
            # self.send_reset_random_waypoint()

        time.sleep(1.0)

    def get_sensor_size(self):
        """
        :return: (tuple)
        """
        return self.camera_img_size

    def take_action(self, action: np.ndarray) -> None:
        """
        :param action: ([float]) Steering and throttle
        """
        throttle = action[1]
        self.steering = action[0]
        self.last_throttle = throttle
        self.actions.append((float(self.steering), float(self.last_throttle)))
        self.current_step += 1
        self.total_steps += 1
        self.send_control(self.steering, throttle)

    def observe(self) -> ObserveData:
        while self.last_obs is self.image_array:
            time.sleep(1.0 / 120.0)

        self.last_obs = self.image_array
        observation = self.image_array

        done = self.is_game_over()
        if done:
            self.episode_counter += 1

        self.last_steering = self.steering

        # FIXME DOES NOT WORK WITH SANDBOX_LAB_SCENE
        if self.simulator_scene.get_scene_name() == GENERATED_TRACK_NAME:
            # rescaling cte to new range (-2, 2)
            old_max = self.max_cte_error
            old_min = -self.max_cte_error
            cte_old_range = old_max - old_min
            new_max = 2
            new_min = -2
            cte_new_range = new_max - new_min

            new_cte = (((self.cte - old_min) * cte_new_range) / cte_old_range) + new_min

            if abs(new_cte) > 2:
                lateral_position = float(-abs(abs(new_cte) - 2))
            else:
                lateral_position = float(abs(abs(new_cte) - 2))
        else:
            lateral_position = float(self.cte)

        # self.logger.debug("Speed: {} - {}".format(self.speed, get_velocity_magnitude(point=Point(self.vel_x, self.vel_y, self.vel_z), convert_to_kmh=True)))

        info = {
            "is_success": self.is_success,
            "track": self.current_track,
            "speed": self.speed,
            "pos": (self.pos_x, self.pos_z),
            "cte": self.cte,
            "cte_pid": self.cte_pid,
            "lateral_position": lateral_position,
        }
        self.control_timer.on_frame()
        # adding reward (always 0) to respect the RL gym interface
        return observation, 0, done, info

    def is_game_over(self) -> bool:
        """
        :return: (bool)
        """
        if self.simulator_scene.get_scene_name() == GENERATED_TRACK_NAME:
            simulator_scene = cast(GeneratedTrack, self.simulator_scene)
            if simulator_scene.track_num is None:
                if abs(self.cte) > self.max_cte_error or self.hit != "none":
                    if abs(self.cte) > self.max_cte_error:
                        self.is_success = 0
                    else:
                        self.is_success = 1
                    return True
            else:
                max_steps = (
                    self.max_steps
                    if self.max_steps is not None
                    else MAX_EPISODE_STEPS_DONKEY_GENERATED_TRACK
                )
                if abs(self.cte) > self.max_cte_error or self.current_step >= max_steps:
                    if abs(self.cte) > self.max_cte_error:
                        self.is_success = 0
                    else:
                        self.is_success = 1
                    return True
            return False

        if self.simulator_scene.get_scene_name() == SANDBOX_LAB_NAME:
            max_steps = (
                self.max_steps
                if self.max_steps is not None
                else MAX_EPISODE_STEPS_DONKEY_SANDBOX
            )
            if self.hit != "none":
                self.is_success = 0
                return True
            elif (
                self.active_node == 15
                or self.active_node == 16
                or self.active_node == 17
                or self.active_node == 18
            ) and abs(self.cte) > self.max_cte_error + 3:
                self.is_success = 0
                return True
            elif (
                self.active_node != 15
                and self.active_node != 16
                and self.active_node != 17
                and self.active_node != 18
            ) and abs(self.cte) > self.max_cte_error:
                self.is_success = 0
                return True
            elif self.current_step >= max_steps:
                self.is_success = 1
                return True
            return False

        raise RuntimeError(
            "Simulator scene {} not supported".format(
                self.simulator_scene.get_scene_name()
            )
        )

    # ------ Socket interface ----------- #

    def on_telemetry(self, data):
        """
        Update car info when receiving telemetry message.

        :param data: (dict)
        """
        # self.send_restart_simulation()
        img_string = data["image"]
        self.images.append(img_string)
        image = Image.open(BytesIO(base64.b64decode(img_string)))
        # Resize and crop image
        image = np.array(image)
        # Save original image for render
        self.original_image = np.copy(image)
        self.image_array = image

        self.pos_x = data["pos_x"]
        self.pos_y = data["pos_y"]
        self.pos_z = data["pos_z"]

        self.vel_x = data["vel_x"]
        self.vel_y = data["vel_y"]
        self.vel_z = data["vel_z"]

        self.car_trajectory.append((self.pos_x, self.pos_z))

        self.rot_x = data["rot_x"]
        self.rot_y = data["rot_y"]
        self.rot_z = data["rot_z"]
        self.rot_w = data["rot_w"]

        self.rotation_angle = data["rotation_angle"]

        self.forward_x = data["forward_x"]
        self.forward_y = data["forward_y"]
        self.forward_z = data["forward_z"]

        self.steering_angle = float(data["steering_angle"])
        self.steerings.append(self.steering_angle)
        self.speed = float(data["speed"] * 3.6)  # conversion m/s to km/h
        self.speeds.append(self.speed)

        self.cte = float(data["cte"])
        self.cte_pid = float(data["cte_pid"])
        self.prev_cte = self.cte
        self.ctes.append(self.cte)

        self.total_nodes = data["totalNodes"]
        self.active_node = data["activeNode"]

        # self.end_lap = data["endLap"]

        self.hit = data["hit"]

        if self.collect_trace and self.csv_logger is not None:
            if len(self.csv_log_lines) == 0:
                self.current_episode_counter_in_log = self.episode_counter
            if self.current_episode_counter_in_log == self.episode_counter:
                self.csv_log_lines.append(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(
                        self.episode_counter,
                        self.active_node,
                        self.pos_x,
                        self.pos_y,
                        self.pos_z,
                        self.rot_x,
                        self.rot_y,
                        self.rot_z,
                        self.rot_w,
                        self.rotation_angle,
                        self.vel_x,
                        self.vel_y,
                        self.vel_z,
                        self.forward_x,
                        self.forward_y,
                        self.forward_z,
                    )
                )

    # DO NOT REMOVE PARAMETER data. The method on_recv_message() calls this method with a message parameter.
    def on_scene_selection_ready(self, data):
        """
        Get the level names when the scene selection screen is ready
        """
        self.logger.info("Scene Selection Ready")
        self.send_get_scene_names()

    # DO NOT REMOVE PARAMETER data. The method on_recv_message() calls this method with a message parameter.
    def on_car_loaded(self, data):
        self.loaded = True

    def on_recv_track(self, data):
        if data is not None:
            self.current_track_string = data["track_string"]

    def on_recv_scene_names(self, data):
        """
        Select the level.

        :param data: (dict)
        """
        if data is not None:
            names = data["scene_names"]
            assert (
                self.simulator_scene.get_scene_name() in names
            ), "{} not in the list of possible scenes {}".format(
                self.simulator_scene.get_scene_name(), names
            )
            self.send_load_scene(self.simulator_scene.get_scene_name())

    def generate_track(self, generated_track: Road = None):
        # self.send_pause_simulation()

        if generated_track is None:
            start_time = time.perf_counter()
            self.logger.debug("Start generating track")
            track = self.road_test_generator.generate()
            self.logger.debug(
                "Track generated: {:.2f}s".format(time.perf_counter() - start_time)
            )
            self.current_track = track
        else:
            self.current_track = generated_track

        track_string = self.current_track.serialize_concrete_representation(
            cr=self.current_track.get_concrete_representation(to_plot=True)
        )

        self.send_regen_track(track_string=track_string)
        self.track_strings.append(track_string)
        max_iterations = 1000
        time_elapsed = 0
        while (
            self.track_strings[-1] != self.current_track_string and max_iterations > 0
        ):
            time.sleep(0.1)
            time_elapsed += 0.1
            if time_elapsed >= 1.0:
                time_elapsed = 0
                self.send_regen_track(track_string=track_string)
            max_iterations -= 1

        if max_iterations == 0:
            assert (
                self.track_strings[-1] == self.current_track_string
            ), "Track generated {} != {} Track deployed".format(
                self.track_strings[-1], self.current_track_string
            )

        time.sleep(1)

    def send_regen_track(self, track_string: str):
        msg = {
            "msg_type": "regen_track",
            "track_string": track_string,
            "path_type": "point_path",
        }
        self.queue_message(msg)

    def send_pause_simulation(self):
        msg = {"msg_type": "pause_simulation"}
        self.queue_message(msg)
        self.is_paused = True

    def send_set_timescale(self, timescale: float):
        msg = {"msg_type": "set_timescale", "timescale": timescale.__str__()}
        self.queue_message(msg)

    def send_restart_simulation(self):
        msg = {"msg_type": "restart_simulation"}
        self.queue_message(msg)
        self.is_paused = False

    def send_control(self, steer, throttle, brake: float = None):
        """
        Send message to the server for controlling the car.

        :param steer: (float)
        :param throttle: (float)
        :param brake: (float)
        """
        if not self.loaded:
            return
        if brake is not None:
            msg = {
                "msg_type": "control",
                "steering": steer.__str__(),
                "throttle": throttle.__str__(),
                "brake": brake.__str__(),
            }
        else:
            msg = {
                "msg_type": "control",
                "steering": steer.__str__(),
                "throttle": throttle.__str__(),
                "brake": "0.0",
            }
        self.queue_message(msg)

    def send_reset_to_waypoint(self, waypoint_num: int):
        msg = {"msg_type": "reset_to_waypoint", "waypoint_num": waypoint_num.__str__()}
        self.queue_message(msg)

    def send_reset_state(self, individual: Individual) -> None:
        state = cast(State, individual.get_implementation())
        msg = {"msg_type": "reset_with_state"}
        # assuming the list of values is synchronized with the list of keys
        values = state.get_values()
        for i, key in enumerate(state.get_keys()):
            msg[key] = str(values[i])
        self.queue_message(msg)

    def send_reset_random_waypoint(self):
        """
        Reset car to random waypoint decided by the simulator.
        """
        msg = {"msg_type": "reset_random_waypoint"}
        self.queue_message(msg)

    def send_reset_car(self):
        """
        Reset car to initial position.
        """
        msg = {"msg_type": "reset_car"}
        self.queue_message(msg)

    def send_get_scene_names(self):
        """
        Get the different levels available
        """
        msg = {"msg_type": "get_scene_names"}
        self.queue_message(msg)

    def send_load_scene(self, scene_name):
        """
        Load a level.

        :param scene_name: (str)
        """
        msg = {"msg_type": "load_scene", "scene_name": scene_name}
        self.queue_message(msg)

    def send_exit_scene(self):
        """
        Go back to scene selection.
        """
        msg = {"msg_type": "exit_scene"}
        self.queue_message(msg)

    def queue_message(self, msg):
        """
        Add message to socket queue.

        :param msg: (dict)
        """
        if self.client is None:
            return

        self.client.queue_message(msg)
