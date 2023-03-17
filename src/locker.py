import os
from pathlib import Path

import psutil


class Locker:
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
        try:
            if not Path(self.path).exists():
                return False
            with open(self.path, "r") as f:
                pid = int(f.readline())
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                return os.getcwd() == process.cwd()
            return False
        except Exception:
            return False
