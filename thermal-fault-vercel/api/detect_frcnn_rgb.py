import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _common import DetectHandler  # noqa: E402


class handler(DetectHandler):
    FAMILY = "frcnn"
    DATASETS = ("rgb",)
