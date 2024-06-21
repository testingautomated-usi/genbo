import argparse
import os
import time

from config import DONKEY_SIM_NAME
from envs.donkey.scenes.simulator_scenes import (
    SIMULATOR_SCENE_NAMES,
    GENERATED_TRACK_NAME,
)
from envs.env_utils import make_env
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--donkey-exe-path",
    help="Path to the donkey simulator executor",
    type=str,
    default=None,
)
parser.add_argument(
    "--donkey-scene-name",
    help="Scene name for the donkey simulator",
    choices=SIMULATOR_SCENE_NAMES,
    type=str,
    default=GENERATED_TRACK_NAME,
)
parser.add_argument(
    "--track-num",
    help="Track number when simulator_name is Donkey and simulator_scene is GeneratedTrack",
    type=int,
    default=0,
)
args = parser.parse_args()

if __name__ == "__main__":

    env_name = DONKEY_SIM_NAME
    seed = 0

    env = make_env(
        simulator_name=env_name,
        seed=seed,
        donkey_exe_path=args.donkey_exe_path,
        donkey_scene_name=args.donkey_scene_name,
        port=-1,
        collect_trace=False,
        headless=True,
        track_num=args.track_num,
    )

    obs = env.reset(skip_generation=True)
    plt.figure()
    plt.imshow(obs)
    plt.savefig(os.path.join("logs", "donkey_test.png"), format="png")

    time.sleep(2)
    env.exit_scene()
    env.close_connection()

    time.sleep(5)
    env.close()
