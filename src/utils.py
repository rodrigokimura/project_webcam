import os
from pathlib import Path

from PIL.Image import init


class LoopCounter:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._counter = 0

    def increment(self) -> None:
        if self.limit_reached:
            self._reset()
        else:
            self._inc()

    def _reset(self) -> None:
        self._counter = 0

    def _inc(self) -> None:
        self._counter += 1

    @property
    def value(self) -> int:
        return self._counter

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def limit_reached(self) -> bool:
        return self._counter >= self._limit - 1

    @property
    def initial_value(self) -> bool:
        return self._counter == 0


def is_webcam_open(source: str | int):
    import cv2

    cap = cv2.VideoCapture(source)
    return cap is not None or cap.isOpened()


class Lock:
    """Simple file based lock"""

    def __init__(self, path: str) -> None:
        self.path = path

    def lock(self):
        Path(self.path).unlink(missing_ok=True)
        with open(self.path, "w") as f:
            f.write(str(os.getpid()))

    def unlock(self):
        Path(self.path).unlink(missing_ok=True)

    @property
    def locked(self):
        if not Path("lock").exists():
            return False
        with open(self.path, "r") as f:
            pid = f.readline()
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, ValueError):
            return False
