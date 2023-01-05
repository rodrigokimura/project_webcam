import time
from threading import Event

import cv2
import numpy as np
from inotify_simple import INotify, flags
from mediapipe.python.solutions import selfie_segmentation as mp
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
    def __init__(self) -> None:
        self.virtual_webcam_path = "/dev/video2"

        self.background_blur = 75
        self.sigma = 5

        self.engine = mp.SelfieSegmentation(model_selection=0)

        self.width, self.height = (1280, 720)

        self.blank_frame = np.zeros((self.height, self.width), np.uint8)
        self.blank_frame.flags.writeable = False

        self.old_mask = None
        self.webcam = Webcam("/dev/video0", self.width, self.height)
        self.virtual_webcam = FakeWebcam(
            self.virtual_webcam_path, self.width, self.height
        )
        self.running = False
        self.consumers = 0
        self.stop_event = Event()

    def _send_to_virtual_webcam(self, frame):
        self.virtual_webcam.schedule_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def start(self):

        self._start_monitoring_webcam()

        while not self.stop_event.is_set():
            self._check_if_webcam_is_open()

            if self.running:
                frame = self.webcam.read()
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

    def _render_image(self, frame: np.ndarray):
        cv2.resize(frame, (self.width, self.height))

        mask = self.engine.process(frame).segmentation_mask

        cv2.threshold(mask, 0.9, 1, cv2.THRESH_BINARY, dst=mask)
        cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1, dst=mask)
        cv2.blur(mask, (10, 10), dst=mask)
        cv2.accumulateWeighted(mask, self.old_mask or mask, 0.5)

        # Blur background
        background_frame = cv2.GaussianBlur(
            frame,
            ksize=(self.background_blur, self.background_blur),
            sigmaX=0,
            sigmaY=0,
            borderType=cv2.BORDER_DEFAULT,
        )

        cv2.blendLinear(frame, background_frame, mask, 1 - mask, dst=frame)

        return frame

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
