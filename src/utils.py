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
