import os
import time
from threading import Event

import cv2
import numpy as np
from inotify_simple import INotify, flags
from numpy._typing import NDArray
from pyfakewebcam import FakeWebcam

from frame import OneThird
from image_processors import Blend, Blur, Resize, SelfieSegmentation, Sepia
from models import Rectangle, Resolution
from utils import LoopCounter


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
        self.resolution = Resolution(1280, 720)
        self.webcam = Webcam(
            "/dev/video0", self.resolution.width, self.resolution.height
        )
        self.virtual_webcam_path = "/dev/video2"
        self.virtual_webcam = FakeWebcam(
            self.virtual_webcam_path, self.resolution.width, self.resolution.height
        )
        self.blank_frame = np.zeros(
            (self.resolution.height, self.resolution.width), np.uint8
        )
        self.blank_frame.flags.writeable = False

        self.blur = 75
        self.sepia = 0.5

        self.running = False
        self.stop_event = Event()

        self._frame_counter_for_mask = LoopCounter(10)
        self._frame_counter_for_face = LoopCounter(30)

        self.mask = None
        self.mask_engine = SelfieSegmentation()
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
                self._frame_counter_for_mask.increment()
                self._frame_counter_for_face.increment()
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
        frame = Resize(self.resolution).process(frame)
        if self.face_tracking and self._classifier:
            self._detect_face(frame)
        if self.face_tracking and self.current_face:
            frame = self._crop_face(frame)
        self._compute_mask(frame)
        background_frame = Blur(self.blur).process(frame)
        background_frame = Sepia(self.sepia).process(background_frame)
        frame = Blend(self.mask, background_frame).process(frame)
        return frame

    def _compute_mask(self, frame: NDArray[np.uint8]) -> None:
        if self._frame_counter_for_mask.limit_reached or self.mask is None:
            self.mask = self.mask_engine.process(frame)

    def _detect_face(self, frame: NDArray[np.uint8]) -> None:
        if (
            not self._frame_counter_for_face.initial_value
            and self.current_face is not None
        ):
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
            self.current_face = OneThird(face, self.resolution).frame()

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

    def _crop_face(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if self.current_face is None:
            return
        rect_to_crop = self.current_face

        if self._frame_counter_for_face.limit_reached:
            self.next_face = rect_to_crop
        else:
            rect_to_crop = Rectangle(
                *(
                    int(
                        (
                            self._frame_counter_for_face.value
                            / self._frame_counter_for_face.limit
                        )
                        * (c - t)
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
            return Resize(self.resolution).process(frame)
        except Exception:
            return frame

    def toggle(self):
        self.running = not self.running
        if self.running:
            print("Running...")
        else:
            print("Stopped...")
