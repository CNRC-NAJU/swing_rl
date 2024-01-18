import sys
import warnings
from io import TextIOBase
from typing import Any, Mapping, Sequence, TextIO

import numpy as np
import torch
from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm


class HParam:
    """
    Hyperparameter data class storing hyperparameters and metrics in dictionaries

    :param hparam_dict: key-value pairs of hyperparameters to log
    :param metric_dict: key-value pairs of metrics to log
        A non-empty metrics dict is required to display hyperparameters in the corresponding Tensorboard section.
    """

    def __init__(
        self,
        hparam_dict: Mapping[str, bool | str | float | None],
        metric_dict: Mapping[str, float],
    ):
        self.hparam_dict = hparam_dict
        if not metric_dict:
            raise Exception(
                "`metric_dict` must not be empty to display hyperparameters to the"
                " HPARAMS tensorboard tab."
            )
        self.metric_dict = metric_dict


class FormatUnsupportedError(NotImplementedError):
    """
    Custom error to display informative message when
    a value is not supported by some formats.

    :param unsupported_formats: A sequence of unsupported formats,
        for instance ``["stdout"]``.
    :param value_description: Description of the value that cannot be logged by this format.
    """

    def __init__(self, unsupported_formats: Sequence[str], value_description: str):
        if len(unsupported_formats) > 1:
            format_str = f"formats {', '.join(unsupported_formats)} are"
        else:
            format_str = f"format {unsupported_formats[0]} is"
        super().__init__(
            f"The {format_str} not supported for the {value_description} value"
            " logged.\nYou can exclude formats via the `exclude` parameter of the"
            " logger's `record` function."
        )


class KVWriter:
    """
    Key Value writer
    """

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        """
        Write a dictionary to file

        :param key_values:
        :param key_excluded:
        :param step:
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Close owned resources
        """
        raise NotImplementedError


class SeqWriter:
    """
    sequence writer
    """

    def write_sequence(self, sequence: list[str]) -> None:
        """
        write_sequence an array to file

        :param sequence:
        """
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    """A human-readable output format producing ASCII tables of key-value pairs.

    Set attribute ``max_length`` to change the maximum length of keys and values
    to write to output (or specify it when calling ``__init__``).

    :param filename_or_file: the file to write the log to
    :param max_length: the maximum length of keys and values to write to output.
        Outputs longer than this will be truncated. An error will be raised
        if multiple keys are truncated to the same value. The maximum output
        width will be ``2*max_length + 7``. The default of 36 produces output
        no longer than 79 characters wide.
    """

    def __init__(self, filename_or_file: str | TextIO, max_length: int = 36):
        self.max_length = max_length
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "w")
            self.own_file = True
        elif isinstance(filename_or_file, TextIOBase) or hasattr(
            filename_or_file, "write"
        ):
            # Note: in theory `TextIOBase` check should be sufficient,
            # in practice, libraries don't always inherit from it, see GH#1598
            self.file = filename_or_file
            self.own_file = False
        else:
            raise ValueError(f"Expected file or str, got {filename_or_file}")

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        # Create strings for printing
        key2str = {}
        tag = ""
        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):
            if excluded is not None and ("stdout" in excluded or "log" in excluded):
                continue
            elif isinstance(value, HParam):
                raise FormatUnsupportedError(["stdout", "log"], "hparam")
            elif isinstance(value, float):
                value_str = f"{value:<8.3g}"  # Align left
            else:
                value_str = str(value)

            # Find tag and add it to the dict
            if key.find("/") > 0:
                tag = key[: key.find("/") + 1]
                key2str[(tag, self._truncate(tag))] = ""

            # Remove tag from key and indent the key
            if len(tag) > 0 and tag in key:
                key = f"{'':3}{key[len(tag) :]}"

            truncated_key = self._truncate(key)
            if (tag, truncated_key) in key2str:
                raise ValueError(
                    f"Key '{key}' truncated to '{truncated_key}' that already exists."
                    " Consider increasing `max_length`."
                )
            key2str[(tag, truncated_key)] = self._truncate(value_str)

        # Find max widths
        if len(key2str) == 0:
            warnings.warn("Tried to write empty key-value dict")
            return
        else:
            tagless_keys = map(lambda x: x[1], key2str.keys())
            key_width = max(map(len, tagless_keys))
            val_width = max(map(len, key2str.values()))

        # Write out the data
        dashes = "-" * (key_width + val_width + 7)
        lines = [dashes]
        for (_, key), value in key2str.items():
            key_space = " " * (key_width - len(key))
            val_space = " " * (val_width - len(value))
            lines.append(f"| {key}{key_space} | {value}{val_space} |")
        lines.append(dashes)

        if hasattr(self.file, "name") and self.file.name == "<stdout>":
            # Do not mess up with progress bar
            tqdm.write("\n".join(lines) + "\n", file=sys.stdout, end="")
        else:
            self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, string: str) -> str:
        if len(string) > self.max_length:
            string = string[: self.max_length - 3] + "..."
        return string

    def write_sequence(self, sequence: list[str]) -> None:
        for i, elem in enumerate(sequence):
            self.file.write(elem)
            if i < len(sequence) - 1:  # add space unless this is the last one
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self) -> None:
        """
        closes the file
        """
        if self.own_file:
            self.file.close()


class TensorBoardOutputFormat(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format.

    :param folder: the folder to write the log to
    """

    def __init__(self, folder: str):
        self.writer = SummaryWriter(log_dir=folder)
        self._is_closed = False

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        assert (
            not self._is_closed
        ), "The SummaryWriter was closed, please re-create one."
        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):
            if excluded is not None and "tensorboard" in excluded:
                continue
            if isinstance(value, np.ScalarType):
                if isinstance(value, str):
                    # str is considered a np.ScalarType
                    self.writer.add_text(key, value, step)
                else:
                    self.writer.add_scalar(key, value, step)
            if isinstance(value, torch.Tensor):
                self.writer.add_histogram(key, value, step)
            if isinstance(value, HParam):
                # we don't use `self.writer.add_hparams` to have control over the log_dir
                experiment, session_start_info, session_end_info = hparams(
                    value.hparam_dict, metric_dict=value.metric_dict
                )
                assert self.writer.file_writer is not None
                self.writer.file_writer.add_summary(experiment)
                self.writer.file_writer.add_summary(session_start_info)
                self.writer.file_writer.add_summary(session_end_info)

        # Flush the output to the file
        self.writer.flush()

    def close(self) -> None:
        """
        closes the file
        """
        if self.writer:
            self.writer.close()
            self._is_closed = True
