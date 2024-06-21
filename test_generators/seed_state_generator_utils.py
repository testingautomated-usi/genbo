import os

from config import SIMULATOR_NAMES, DONKEY_SIM_NAME, REFERENCE_TRACE_FILENAME
from envs.donkey.donkey_env_utils import make_simulator_scene
from envs.donkey.scenes.simulator_scenes import GENERATED_TRACK_NAME
from self_driving.state_utils import get_state_condition_function
from test.config import (
    SEED_STATE_GENERATOR_NAMES,
    RANDOM_SEED_STATE_GENERATOR_NAME,
    PERFORMANCE_SEED_STATE_GENERATOR_NAME,
)
from test_generators.performance_seed_state_test_generator import (
    PerformanceSeedStateTestGenerator,
)
from test_generators.random_seed_state_test_generator import (
    RandomSeedStateTestGenerator,
)
from test_generators.state_test_generator import SeedStateTestGenerator


def make_seed_state_generator(
    generator_name: str,
    env_name: str,
    constant_road: bool,
    donkey_scene_name: str = None,
    track_num: int = 0,
    folder: str = "logs",
) -> SeedStateTestGenerator:
    assert env_name in SIMULATOR_NAMES, "Env {} not found. Choose between {}".format(
        env_name, SIMULATOR_NAMES
    )
    assert (
        generator_name in SEED_STATE_GENERATOR_NAMES
    ), "Test generator {} not found. Choose between {}".format(
        generator_name, SEED_STATE_GENERATOR_NAMES
    )
    if (
        generator_name == RANDOM_SEED_STATE_GENERATOR_NAME
        or PERFORMANCE_SEED_STATE_GENERATOR_NAME
    ):
        project_root = os.path.dirname(os.path.dirname(__file__))
        path_to_csv_file = os.path.join(project_root, folder, env_name)
        if env_name == DONKEY_SIM_NAME:
            assert donkey_scene_name is not None, "Donkey scene name must be specified"
            path_to_csv_file = os.path.join(path_to_csv_file, donkey_scene_name)
            if donkey_scene_name == GENERATED_TRACK_NAME:
                path_to_csv_file = os.path.join(
                    path_to_csv_file,
                    "{}_{}.csv".format(
                        REFERENCE_TRACE_FILENAME.split(".")[0], str(track_num)
                    ),
                )

        assert os.path.exists(path_to_csv_file), "Path to csv file does not exist"

        if generator_name == RANDOM_SEED_STATE_GENERATOR_NAME:
            return RandomSeedStateTestGenerator(
                env_name=env_name,
                road_points=[],
                control_points=[],
                road_width=1,
                constant_road=constant_road,
                state_condition_fn=get_state_condition_function(
                    env_name=env_name, constant_road=constant_road
                ),
                path_to_csv_file=path_to_csv_file,
                simulator_scene=make_simulator_scene(
                    scene_name=donkey_scene_name, track_num=track_num
                ),
            )

        if generator_name == PERFORMANCE_SEED_STATE_GENERATOR_NAME:
            return PerformanceSeedStateTestGenerator(
                env_name=env_name,
                road_points=[],
                control_points=[],
                road_width=1,
                constant_road=constant_road,
                state_condition_fn=get_state_condition_function(
                    env_name=env_name, constant_road=constant_road
                ),
                path_to_csv_file=path_to_csv_file,
                simulator_scene=make_simulator_scene(
                    scene_name=donkey_scene_name, track_num=track_num
                ),
            )

    raise NotImplementedError("{} not supported yet".format(generator_name))
