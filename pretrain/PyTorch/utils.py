import sys as _sys

from typing import List
from collections import _iskeyword  # type: ignore
from tensorboardX import SummaryWriter
import os

SUMMARY_WRITER_DIR_NAME = 'runs'


def get_sample_writer(name, base=".."):
    """Returns a tensorboard summary writer
    """
    return SummaryWriter(log_dir=os.path.join(base, SUMMARY_WRITER_DIR_NAME, name))
