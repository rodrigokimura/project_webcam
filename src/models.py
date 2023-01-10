from __future__ import annotations

from typing import NamedTuple


class Point(NamedTuple):
    x: int
    y: int

    def __sub__(self, other: Point) -> int:
        import math

        return abs(
            int(
                math.sqrt(
                    math.pow((self.x - other.x), 2) + math.pow((self.y - other.y), 2)
                )
            )
        )


class Rectangle(NamedTuple):
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> Point:
        return Point(self.x + self.w // 2, self.y + self.h // 2)


class Resolution(NamedTuple):
    width: int
    height: int
