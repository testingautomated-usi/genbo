GENERATED_TRACK_NAME = "generated_track"
SANDBOX_LAB_NAME = "sandbox_lab"
SIMULATOR_SCENE_NAMES = [GENERATED_TRACK_NAME, SANDBOX_LAB_NAME]


class SimulatorScene:

    def __init__(self, scene_name: str):
        self.scene_name = scene_name

    def get_scene_name(self) -> str:
        return self.scene_name


class GeneratedTrack(SimulatorScene):

    def __init__(self, track_num: int = None):
        super(GeneratedTrack, self).__init__(scene_name=SIMULATOR_SCENE_NAMES[0])
        self.track_num = track_num


class SandboxLab(SimulatorScene):

    def __init__(self):
        super(SandboxLab, self).__init__(scene_name=SIMULATOR_SCENE_NAMES[1])


SIMULATOR_SCENES_DICT = {
    GENERATED_TRACK_NAME: GeneratedTrack(),
    SANDBOX_LAB_NAME: SandboxLab(),
}
