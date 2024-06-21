import datetime
import os
import tensorflow as tf
from typing import Tuple

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Lambda, Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

from config import SIMULATOR_NAMES, INPUT_SHAPE

import matplotlib.pyplot as plt

from utils.dataset_utils import DataGenerator
from tensorflow.keras.models import load_model


class AutopilotModel:

    def __init__(
        self,
        env_name: str,
        input_shape: Tuple[int] = INPUT_SHAPE,
        predict_throttle: bool = True,
    ):
        # cropped input_shape: height, width, channels. Allow for mixed datasets
        assert (
            env_name in SIMULATOR_NAMES or env_name == "mixed"
        ), "Unknown simulator name {}. Choose among {}".format(
            env_name, SIMULATOR_NAMES
        )
        self.input_shape = input_shape
        self.env_name = env_name
        self.predict_throttle = predict_throttle
        self.model = None

    def build_model(self, keep_probability: float = 0.5) -> Sequential:
        """
        Modified NVIDIA model
        """
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=self.input_shape))
        model.add(Conv2D(24, (5, 5), activation="elu", strides=(2, 2)))
        model.add(Conv2D(36, (5, 5), activation="elu", strides=(2, 2)))
        model.add(Conv2D(48, (5, 5), activation="elu", strides=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation="elu"))
        model.add(Conv2D(64, (3, 3), activation="elu"))
        model.add(Dropout(keep_probability))
        model.add(Flatten())
        model.add(Dense(100, activation="elu"))
        model.add(Dense(50, activation="elu"))
        model.add(Dense(10, activation="elu"))

        if self.predict_throttle:
            model.add(Dense(2))
        else:
            model.add(Dense(1))

        model.summary()

        return model

    def load(self, model_path: str, compile_model: bool = False) -> None:
        assert os.path.exists(model_path) or os.path.exists(
            model_path.replace(".h5", ""),
        ), "Model path {} not found".format(model_path)
        if compile_model:
            if os.path.exists(model_path):
                self.model = load_model(filepath=model_path, compile=compile_model)
            elif os.path.exists(model_path.replace(".h5", "")):
                self.model = load_model(
                    filepath=model_path.replace(".h5", ""), compile=compile_model
                )
            else:
                raise RuntimeError(f"Model file does not exist: {model_path}")
        else:
            with tf.device("cpu:0"):
                if os.path.exists(model_path):
                    self.model = load_model(filepath=model_path, compile=compile_model)
                elif os.path.exists(model_path.replace(".h5", "")):
                    self.model = load_model(
                        filepath=model_path.replace(".h5", ""), compile=compile_model
                    )
                else:
                    raise RuntimeError(f"Model file does not exist: {model_path}")

    def train_model(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        save_path: str,
        model_name: str,
        save_best_only: bool = True,
        keep_probability: float = 0.5,
        learning_rate: float = 1e-4,
        nb_epoch: int = 200,
        batch_size: int = 128,
        early_stopping_patience: int = 3,
        save_plots: bool = True,
        preprocess: bool = True,
        augment: bool = True,
        save_with_epoch: bool = False,
        finetune: bool = False,
        model_path: str = None,
        model_name_suffix: str = None,
        compile_model: bool = True,
    ) -> None:
        os.makedirs(save_path, exist_ok=True)
        if not finetune:
            self.model = self.build_model(keep_probability=keep_probability)
            if model_name_suffix is None:
                filename = "{}-{}-{}".format(
                    self.env_name,
                    model_name,
                    datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
                )
        else:
            assert model_path is not None, "Specify model path when finetune"
            filename = f"{self.env_name}-{model_path}-finetune"

        if model_name_suffix:
            if not finetune:
                filename = model_name_suffix
            else:
                filename += f"-{model_name_suffix}"

        if save_with_epoch:
            filename += "_{epoch:02d}"

        checkpoint = ModelCheckpoint(
            filepath=os.path.join(save_path, filename),
            monitor="val_loss",
            verbose=0,
            save_best_only=save_best_only,
            mode="auto",
        )

        if compile_model:
            self.model.compile(
                loss="mean_squared_error", optimizer=Adam(lr=learning_rate)
            )

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=early_stopping_patience
        )

        train_generator = DataGenerator(
            X=X_train,
            y=y_train,
            batch_size=batch_size,
            is_training=True,
            env_name=self.env_name,
            input_shape=self.input_shape,
            predict_throttle=self.predict_throttle,
            preprocess=preprocess,
            augmentation=augment,
        )
        validation_generator = DataGenerator(
            X=X_val,
            y=y_val,
            batch_size=batch_size,
            is_training=False,
            env_name=self.env_name,
            input_shape=self.input_shape,
            predict_throttle=self.predict_throttle,
            preprocess=preprocess,
            augmentation=augment,
        )

        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=nb_epoch,
            use_multiprocessing=False,
            max_queue_size=10,
            workers=8,
            callbacks=[checkpoint, early_stopping],
            verbose=1,
        )

        print(
            f"Validation loss: {history.history['val_loss']}. Best: {min(history.history['val_loss'])}"
        )

        if save_plots:

            plt.figure()
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.title("model loss")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train", "val"], loc="upper left")
            plt.savefig(
                os.path.join(save_path, f"{filename}.pdf"),
                format="pdf",
            )
