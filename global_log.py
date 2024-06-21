import logging
import os.path
import sys

from config_variables import LOGGING_LEVEL

logging.getLogger("matplotlib").setLevel(logging.WARNING)


class GlobalLog:
    def __init__(self, logger_prefix: str, verbose: bool = True):
        """
        When verbose = False, all the logs with level < INFO are ignored
        """
        self.logger = logging.getLogger(logger_prefix)
        self.verbose = verbose
        # avoid creating another logger if it already exists
        if len(self.logger.handlers) == 0:
            self.logger = logging.getLogger(logger_prefix)
            self.logger.setLevel(level=LOGGING_LEVEL)

            # FIXME: it seems that it is not needed to stream log to sdtout
            formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            ch.setLevel(level=logging.DEBUG)
            self.logger.addHandler(ch)

    def debug(self, message):
        if self.verbose:
            self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warn(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


# from https://gist.github.com/robdmc/d78d48467e3daea22fe6
class CSVLogger:  # pragma: no cover
    def __init__(self, path_to_csv_file: str, header: str):
        assert path_to_csv_file.endswith(".csv"), "{} not a csv file".format(
            path_to_csv_file
        )
        assert (
            len(header.split(",")) > 0
        ), "Header {} should be a list of comma separated items".format(header)
        # create logger on the current module and set its level
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.header = header

        self.log_file = path_to_csv_file
        if self.log_file:
            # create a channel for handling the logger (stderr) and set its format
            ch = logging.FileHandler(path_to_csv_file)
            assert os.path.exists(path_to_csv_file), "{} does not exist".format(
                path_to_csv_file
            )
        else:
            # create a channel for handling the logger (stderr) and set its format
            ch = logging.StreamHandler()

        # connect the logger to the channel
        self.logger.addHandler(ch)
        self.logger.debug(self.header)

    def log(self, msg: str):
        self.logger.debug(msg=msg)
