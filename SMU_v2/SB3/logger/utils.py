import datetime
import os
import sys
import tempfile
from pathlib import Path

from .logger import Logger
from .output_format import HumanOutputFormat, KVWriter, TensorBoardOutputFormat


def get_latest_run_id(log_path: str = "", log_name: str = "") -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: Path to the log folder containing several runs.
    :param log_name: Name of the experiment. Each run is stored
        in a folder named ``log_name_1``, ``log_name_2``, ...
    :return: latest run number
    """
    max_run_id = 0
    for path in Path(log_path).glob(f"{log_name}_[0-9]*"):
        file_name = path.name
        ext = file_name.split("_")[-1]
        if (
            log_name == "_".join(file_name.split("_")[:-1])
            and ext.isdigit()
            and int(ext) > max_run_id
        ):
            max_run_id = int(ext)
    return max_run_id


def make_output_format(_format: str, log_dir: str, log_suffix: str = "") -> KVWriter:
    """
    return a logger for the requested format

    :param _format: the requested format to log to ('stdout', 'log', 'json' or 'csv' or 'tensorboard')
    :param log_dir: the logging directory
    :param log_suffix: the suffix for the log file
    :return: the logger
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    if _format == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif _format == "tensorboard":
        return TensorBoardOutputFormat(log_dir)
    else:
        raise ValueError(f"Unknown format specified: {_format}")


def configure_logger(
    verbose: int = 0,
    tensorboard_log: str | None = None,
    tb_log_name: str = "",
    reset_num_timesteps: bool = True,
) -> Logger:
    """
    Configure the logger's outputs.

    :param verbose: Verbosity level: 0 for no output, 1 for the standard output to be part of the logger outputs
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param tb_log_name: tensorboard log
    :param reset_num_timesteps:  Whether the ``num_timesteps`` attribute is reset or not.
        It allows to continue a previous learning curve (``reset_num_timesteps=False``)
        or start from t=0 (``reset_num_timesteps=True``, the default).
    :return: The logger object
    """
    save_path, format_strings = None, ["stdout"]

    if tensorboard_log is not None:
        latest_run_id = get_latest_run_id(tensorboard_log, tb_log_name)
        if not reset_num_timesteps:
            # Continue training in the same directory
            latest_run_id -= 1
        save_path = Path(tensorboard_log) / f"{tb_log_name}_{latest_run_id + 1}"
        if verbose >= 1:
            format_strings = ["stdout", "tensorboard"]
        else:
            format_strings = ["tensorboard"]
    elif verbose == 0:
        format_strings = [""]

    # Save path: if None, $SB3_LOGDIR, if still None, tempdir/SB3-[date & time]
    if save_path is None:
        base_save_path = os.getenv("SB3_LOGDIR")
        if base_save_path is not None:
            save_path = Path(base_save_path)
        else:
            save_path = Path(tempfile.gettempdir()) / datetime.datetime.now().strftime(
                "SB3-%Y-%m-%d-%H-%M-%S-%f"
            )
    save_path.mkdir(parents=True, exist_ok=True)

    savepath = str(save_path)
    output_formats = [make_output_format(f, savepath) for f in format_strings]
    logger = Logger(folder=savepath, output_formats=output_formats)

    # Only print when some files will be saved
    if format_strings and format_strings != ["stdout"]:
        logger.log(f"Logging to {save_path}")
    return logger
