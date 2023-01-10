from __future__ import annotations

import os
import time
from threading import Event
from typing import NamedTuple

import cv2
import numpy as np
from inotify_simple import INotify, flags
from mediapipe.python.solutions import selfie_segmentation as mp
from numpy._typing import NDArray
from pyfakewebcam import FakeWebcam


class Webcam:
    def __init__(self, path, width, height):
        self._path = path
        self._width = width
        self._height = height
        self._prepare()

    def _prepare(self):
        self._vc = cv2.VideoCapture(self._path, cv2.CAP_V4L2)
        self._vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._vc.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._vc.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

    def read(self) -> np.ndarray:
        if self._vc is None:
            self._prepare()
        while True:
            grabbed, frame = self._vc.read()
            if not grabbed or frame is None:
                continue
            return frame

    def close(self) -> None:
        if self._vc is not None:
            self._vc.release()
            self._vc = None


class VirtualWebcam:
    def __init__(self, face_tracking: bool) -> None:
        self.virtual_webcam_path = "/dev/video2"

        self.background_blur = 75
        self.sigma = 5

        self.engine = mp.SelfieSegmentation(model_selection=0)

        self.resolution = Resolution(1280, 720)

        self.blank_frame = np.zeros(
            (self.resolution.height, self.resolution.width), np.uint8
        )
        self.blank_frame.flags.writeable = False

        self.old_mask = None
        self.webcam = Webcam(
            "/dev/video0", self.resolution.width, self.resolution.height
        )
        self.virtual_webcam = FakeWebcam(
            self.virtual_webcam_path, self.resolution.width, self.resolution.height
        )
        self.running = False
        self.stop_event = Event()

        self.sepia_intensity = 0.5

        self._frame_counter_for_mask = 0
        self._frame_counter_for_face = 0

        self.mask_frame_limit = 10
        self.face_frame_limit = 30

        self.mask = None
        self.face_tracking = face_tracking
        self.face_tracking_threshold = 0.1

        self.current_face: Rectangle = Rectangle(
            self.resolution.width // 6,
            self.resolution.height // 6,
            (self.resolution.width // 3) * 2,
            (self.resolution.height // 3) * 2,
        )
        self.next_face: Rectangle = Rectangle(
            0, 0, self.resolution.width, self.resolution.height
        )

        classifier_path = os.path.join(
            os.path.dirname(__file__), "assets", "haarcascade_frontalface_default.xml"
        )
        self._classifier = (
            cv2.CascadeClassifier(classifier_path)
            if os.path.exists(classifier_path)
            else None
        )
        self.show_guides = False

    def _send_to_virtual_webcam(self, frame):
        try:
            self.virtual_webcam.schedule_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(e)

    def start(self):

        self._start_monitoring_webcam()

        while not self.stop_event.is_set():
            self._check_if_webcam_is_open()

            if self.running:
                frame = self.webcam.read()
                if self._frame_counter_for_mask > self.mask_frame_limit:
                    self._frame_counter_for_mask = 0
                else:
                    self._frame_counter_for_mask += 1
                if self._frame_counter_for_face >= self.face_frame_limit - 1:
                    self._frame_counter_for_face = 0
                else:
                    self._frame_counter_for_face += 1
                frame = self._render_image(frame)
            else:
                frame = self.blank_frame
                self.webcam.close()
                time.sleep(1)

            self._send_to_virtual_webcam(frame)

    def stop(self):
        self.stop_event.set()

    def _start_monitoring_webcam(self):
        self.monitor = INotify(nonblocking=True)
        self.monitor.add_watch(
            self.virtual_webcam_path,
            flags.CREATE | flags.OPEN | flags.CLOSE_NOWRITE | flags.CLOSE_WRITE,
        )

    def _check_if_webcam_is_open(self):
        for event in self.monitor.read(0):
            _flags = flags.from_mask(event.mask)
            if flags.CLOSE_NOWRITE in _flags or flags.CLOSE_WRITE in _flags:
                self.running = False
                print("Detected webcam close")
                break
            if flags.OPEN in _flags:
                self.running = True
                print("Detected webcam open")
                break

    def _render_image(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        cv2.resize(frame, self.resolution)
        if self.face_tracking and self._classifier:
            self._detect_face(frame)
        if self.face_tracking and self.current_face:
            frame = self._crop_face(frame)
        self._compute_mask(frame)
        background_frame = self._apply_blur(frame)
        background_frame = self._apply_sepia(background_frame)
        cv2.blendLinear(frame, background_frame, self.mask, 1 - self.mask, dst=frame)
        return frame

    def _compute_mask(self, frame: NDArray[np.uint8]) -> None:

        if (
            self._frame_counter_for_mask != self.mask_frame_limit
            and self.mask is not None
        ):
            return

        self.mask = self.engine.process(frame).segmentation_mask
        cv2.threshold(self.mask, 0.9, 1, cv2.THRESH_BINARY, dst=self.mask)
        cv2.dilate(self.mask, np.ones((5, 5), np.uint8), iterations=1, dst=self.mask)
        cv2.blur(self.mask, (10, 10), dst=self.mask)
        cv2.accumulateWeighted(self.mask, self.old_mask or self.mask, 0.5)

    def _detect_face(self, frame: NDArray[np.uint8]) -> None:

        if self._frame_counter_for_face != 0 and self.current_face is not None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._classifier.detectMultiScale(gray, 1.1, 6)
        if len(faces) == 0:
            return
        if len(faces) == 1:
            face = Rectangle(*faces[0])
        else:
            max_area = max(w * h for (x, y, w, h) in faces)
            face = Rectangle(
                *next((x, y, w, h) for (x, y, w, h) in faces if w * h == max_area)
            )

        is_too_small = face.h < self.current_face.h * 0.2
        if is_too_small:
            # Propably a false positive
            return

        if self.show_guides:
            self._draw_guides(frame, face)

        d = face.center - self.current_face.center
        is_too_far = d > self.current_face.h * self.face_tracking_threshold
        if is_too_far or is_too_small:
            self.current_face = self._enlarge_face_rect(face)

    def _draw_guides(self, frame: NDArray[np.uint8], face: Rectangle):
        crosshair_size = 10
        cv2.rectangle(
            frame,
            (face.x, face.y),
            (face.x + face.w, face.y + face.h),
            (0, 255, 0),
            2,
        )
        cv2.line(
            frame,
            (
                face.x + face.w // 2 - crosshair_size // 2,
                face.y + face.h // 2,
            ),
            (
                face.x + face.w // 2 + crosshair_size // 2,
                face.y + face.h // 2,
            ),
            (0, 255, 0),
            2,
        )
        cv2.line(
            frame,
            (
                face.x + face.w // 2,
                face.y + face.h // 2 - crosshair_size // 2,
            ),
            (
                face.x + face.w // 2,
                face.y + face.h // 2 + crosshair_size // 2,
            ),
            (0, 255, 0),
            2,
        )

    def _enlarge_face_rect(self, rect: Rectangle) -> Rectangle:
        from frame import OneThird

        return OneThird(rect, self.resolution).frame()

    def _crop_face(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if self.current_face is None:
            return
        rect_to_crop = self.current_face

        if self._frame_counter_for_face == self.face_frame_limit - 1:
            self.next_face = rect_to_crop
        else:
            rect_to_crop = Rectangle(
                *(
                    int(
                        (self._frame_counter_for_face / self.face_frame_limit) * (c - t)
                        + t
                    )
                    for c, t in zip(self.current_face, self.next_face)
                )
            )
        return self._crop_and_resize(frame, rect_to_crop)

    def _crop_and_resize(
        self, frame: NDArray[np.uint8], rect: Rectangle
    ) -> NDArray[np.uint8]:
        if rect.h == 0 or rect.w == 0:
            return frame
        try:
            frame = frame[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]
            return cv2.resize(frame, self.resolution)
        except Exception:
            return frame

    def _apply_blur(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return cv2.GaussianBlur(
            frame,
            ksize=(self.background_blur, self.background_blur),
            sigmaX=0,
            sigmaY=0,
            borderType=cv2.BORDER_DEFAULT,
        )

    def _apply_sepia(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        intensity = self.sepia_intensity
        if intensity <= 0:
            return frame
        if intensity > 1:
            intensity = 1
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

    def set_blur(self, blur: int):
        if blur < 0:
            blur = 0
        is_even = blur % 2 == 0
        if is_even:
            blur += 1
        self.background_blur = blur

    def toggle(self):
        self.running = not self.running
        if self.running:
            print("Running...")
        else:
            print("Stopped...")


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
