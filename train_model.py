import argparse
import os
import time

import numpy as np

from autopilot_model import AutopilotModel
from config import SIMULATOR_NAMES, INPUT_SHAPE
from global_log import GlobalLog
from utils.dataset_utils import load_archive_into_dataset
from utils.randomness import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="Random seed", type=int, default=-1)
parser.add_argument("--archive-path", help="Archive path", type=str, default="logs")
parser.add_argument(
    "--env-name",
    help="Simulator name",
    type=str,
    choices=[*SIMULATOR_NAMES, "mixed"],
    required=True,
)
parser.add_argument(
    "--archive-names",
    nargs="+",
    help="Archive name to analyze (with extension, .npz)",
    type=str,
    required=True,
)
parser.add_argument(
    "--model-save-path",
    help="Path where model will be saved",
    type=str,
    default=os.path.join("logs", "models"),
)
parser.add_argument(
    "--model-name", help="Model name (without the extension)", type=str, required=True
)
parser.add_argument(
    "--predict-throttle",
    help="Predict steering and throttle",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--no-preprocess",
    help="Do not preprocess data during training",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--no-augment",
    help="Do not apply data augmentation during training",
    action="store_true",
    default=False,
)
parser.add_argument("--test-split", help="Test split", type=float, default=0.2)
parser.add_argument(
    "--keep-probability", help="Keep probability (dropout)", type=float, default=0.5
)
parser.add_argument("--learning-rate", help="Learning rate", type=float, default=1e-4)
parser.add_argument("--nb-epoch", help="Number of epochs", type=int, default=200)
parser.add_argument("--batch-size", help="Batch size", type=int, default=128)
parser.add_argument(
    "--percentage-data", help="Percentage of data to consider", type=float, default=1.0
)
parser.add_argument(
    "--save-with-epoch",
    help="Save model files with epoch number",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--model-path", help="Path to agent model with extension", type=str, default=None
)
parser.add_argument(
    "--model-name-suffix",
    help="Model name suffix to use in the filename (both model and pdf loss)",
    type=str,
    default=None,
)
parser.add_argument(
    "--finetune", help="Finetune existing model", action="store_true", default=False
)
parser.add_argument(
    "--early-stopping-patience",
    help="Number of epochs of no validation loss improvement used to stop training",
    type=int,
    default=3,
)
args = parser.parse_args()

if __name__ == "__main__":

    logg = GlobalLog("train_model")

    start_time = time.perf_counter()

    if args.seed == -1:
        try:
            args.seed = np.random.randint(2**32 - 1)
        except ValueError as e:
            args.seed = np.random.randint(2**30 - 1)

    logg.info("Random seed: {}".format(args.seed))
    set_random_seed(seed=args.seed)

    train_data, test_data, train_labels, test_labels = load_archive_into_dataset(
        archive_path=args.archive_path,
        archive_names=args.archive_names,
        seed=args.seed,
        test_split=args.test_split,
        predict_throttle=args.predict_throttle,
        percentage_data=args.percentage_data,
        finetune=args.finetune,
    )

    autopilot_model = AutopilotModel(
        env_name=args.env_name,
        input_shape=INPUT_SHAPE,
        predict_throttle=args.predict_throttle,
    )

    compile_model = False

    if args.finetune:
        assert args.model_path is not None, "Specify model path when finetuning"
        model_path = args.model_path.replace(args.env_name + "-", "")
        last_slash_index = model_path.rindex(os.path.sep)
        last_dot_index = model_path.rindex(".")
        model_path = model_path[last_slash_index + 1 : last_dot_index]
        autopilot_model.load(model_path=args.model_path, compile_model=compile_model)
    else:
        model_path = None

    autopilot_model.train_model(
        X_train=train_data,
        X_val=test_data,
        y_train=train_labels,
        y_val=test_labels,
        save_path=args.model_save_path,
        model_name=args.model_name,
        save_best_only=True,
        keep_probability=args.keep_probability,
        learning_rate=args.learning_rate,
        nb_epoch=args.nb_epoch,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping_patience,
        save_plots=True,
        preprocess=not args.no_preprocess,
        augment=not args.no_augment,
        save_with_epoch=args.save_with_epoch,
        finetune=args.finetune,
        model_path=model_path,
        model_name_suffix=args.model_name_suffix,
        compile_model=not compile_model,
    )

    logg.info(f"Time elapsed: {time.perf_counter() - start_time:.2f}s")
