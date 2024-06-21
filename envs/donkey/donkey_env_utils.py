from envs.donkey.scenes.simulator_scenes import (
    SimulatorScene,
    SIMULATOR_SCENES_DICT,
    GENERATED_TRACK_NAME,
    GeneratedTrack,
)


def make_simulator_scene(scene_name: str, track_num: int) -> SimulatorScene:
    for simulator_scene in SIMULATOR_SCENES_DICT.keys():
        if simulator_scene == scene_name:
            if simulator_scene == GENERATED_TRACK_NAME and track_num is not None:
                return GeneratedTrack(track_num=track_num)
            return SIMULATOR_SCENES_DICT[simulator_scene]
    raise RuntimeError("Simulator scene {} not found".format(scene_name))
