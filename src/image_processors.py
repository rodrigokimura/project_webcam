from abc import ABC, abstractmethod

import cv2
import numpy as np
from mediapipe.python.solutions import selfie_segmentation as mp
from numpy._typing import NDArray

from models import Resolution


class ImageProcessor(ABC):
    @abstractmethod
    def process(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        ...


class Blur(ImageProcessor):
    def __init__(self, blur: int) -> None:
        if blur < 0:
            blur = 0
        is_even = blur % 2 == 0
        if is_even:
            blur += 1
        self.blur = blur

    def process(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return cv2.GaussianBlur(
            frame,
            ksize=(self.blur, self.blur),
            sigmaX=0,
            sigmaY=0,
            borderType=cv2.BORDER_DEFAULT,
        )


class Sepia(ImageProcessor):
    def __init__(self, intensity: float) -> None:
        if intensity <= 0:
            self.intensity = 0
        elif intensity > 1:
            self.intensity = 1
        else:
            self.intensity = intensity

    def process(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        intensity = self.intensity
        if intensity == 0:
            return frame
        normalized_gray = (
            np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), np.float32) / 255
        )
        sepia = np.ones(frame.shape)
        r, g, b = 255, 204, 153
        sepia[:, :, 2] = r * normalized_gray
        sepia[:, :, 1] = g * normalized_gray
        sepia[:, :, 0] = b * normalized_gray
        sepia = np.array(sepia, np.uint8)
        return cv2.addWeighted(frame, 1 - intensity, sepia, intensity, 0)


class Blend(ImageProcessor):
    def __init__(
        self, mask: NDArray[np.uint8], background_frame: NDArray[np.uint8]
    ) -> None:
        self.mask = mask
        self.background_frame = background_frame

    def process(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return cv2.blendLinear(
            frame, self.background_frame, self.mask, 1 - self.mask
        )


class Resize(ImageProcessor):
    def __init__(self, resolution: Resolution) -> None:
        self.resolution = resolution

    def process(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return cv2.resize(
            frame,
            self.resolution,
            interpolation=cv2.INTER_AREA,
        )


class SelfieSegmentation(ImageProcessor):
    def __init__(self) -> None:
        self.engine = mp.SelfieSegmentation(model_selection=0)

    def process(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        mask = self.engine.process(frame).segmentation_mask
        cv2.threshold(mask, 0.9, 1, cv2.THRESH_BINARY, dst=mask)
        cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1, dst=mask)
        cv2.blur(mask, (10, 10), dst=mask)
        return mask
