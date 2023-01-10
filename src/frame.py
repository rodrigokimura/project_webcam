from abc import ABC, abstractmethod

from models import Rectangle, Resolution


class FramingStrategy(ABC):
    def __init__(self, face: Rectangle, resolution: Resolution):
        self.face = face
        self.resolution = resolution

    def frame(self) -> Rectangle:
        frame = self.get_frame()
        x, y, w, h = frame.x, frame.y, frame.w, frame.h
        if x < 0:
            x = 0
        if x + w > self.resolution.width:
            x = self.resolution.width - w

        y = int(self.face.y - (h - self.face.h) / 2)
        if y < 0:
            y = 0
        if y + h > self.resolution.height:
            y = self.resolution.height - h

        return Rectangle(x, y, w, h)

    @abstractmethod
    def get_frame() -> Rectangle:
        pass


class Center(FramingStrategy):
    def get_frame(self) -> Rectangle:
        h = int(2 * self.face.h)
        w = int(h * (self.resolution.width / self.resolution.height))
        y = int(self.face.y - (h - self.face.h) / 2)
        x = int(self.face.x - (w - self.face.w) / 2)
        return Rectangle(x, y, w, h)


class OneThird(FramingStrategy):
    def get_frame(self) -> Rectangle:
        center_y = self.resolution.height // 3
        h = int(2 * self.face.h)
        y = center_y - h // 2

        w = int(h * (self.resolution.width / self.resolution.height))
        x = int(self.face.x - (w - self.face.w) / 2)
        return Rectangle(x, y, w, h)
