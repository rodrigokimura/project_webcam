import pytest

from utils import is_webcam_open


def test_is_webcam_open():
    assert is_webcam_open(0) == True
