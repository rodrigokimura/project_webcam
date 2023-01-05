import signal
import sys
from abc import ABC, abstractmethod
from threading import Thread
from tkinter import HORIZONTAL, Tk, ttk

from webcam import VirtualWebcam


class BaseApp(ABC):
    @abstractmethod
    def run(self):
        pass


class CLIApp(BaseApp):
    def run(self):
        cam = VirtualWebcam()
        signal.signal(signal.SIGINT, lambda *args, **kwargs: cam.toggle())
        signal.signal(signal.SIGQUIT, lambda *args, **kwargs: sys.exit(0))
        cam.start()


class GUIApp(BaseApp):
    def run(self):
        self.cam = VirtualWebcam()
        self.start()
        self.root = Tk()
        self.root.title("Webcam")
        self.root.geometry("300x300")
        ttk.Style(self.root).theme_use("clam")
        self.render_windows()
        self.root.mainloop()

    def render_windows(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.pack()

        ttk.Button(
            frm,
            text="Toggle",
            command=lambda: self.cam.toggle(),
        ).pack(padx=10, pady=10)

        ttk.Button(
            frm,
            text="Exit",
            command=lambda: self.exit(),
        ).pack(padx=10, pady=10)

        self.render_blur_control().pack(padx=10, pady=10)
        self.render_sepia_control().pack(padx=10, pady=10)

    def render_blur_control(self):
        frm = ttk.Frame(self.root, padding=10)
        ttk.Label(frm, text="Blur").pack()
        ttk.Scale(
            frm,
            from_=0,
            to=100,
            orient=HORIZONTAL,
            command=self.set_blur,
            value=self.cam.background_blur,
        ).pack(expand=True, fill="both")
        return frm

    def render_sepia_control(self):
        frm = ttk.Frame(self.root, padding=10)
        ttk.Label(frm, text="Sepia").pack()
        ttk.Scale(
            frm,
            from_=0,
            to=1,
            orient=HORIZONTAL,
            command=self.set_sepia,
            value=self.cam.sepia_intensity,
        ).pack(expand=True, fill="both")
        return frm

    def set_blur(self, value):
        self.cam.set_blur(int(float(value)))

    def set_sepia(self, value):
        self.cam.sepia_intensity = float(value)

    def start(self):
        self.main_thread = Thread(target=lambda: self.cam.start(), name="start_webcam")
        self.main_thread.start()

    def exit(self):
        self.cam.stop()
        self.main_thread.join()
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    app_class = GUIApp
    app_class().run()
