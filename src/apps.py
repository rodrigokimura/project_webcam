import signal
import sys
from abc import ABC, abstractmethod
from threading import Thread
from tkinter import HORIZONTAL, Image, Tk, ttk

from webcam import VirtualWebcam


class BaseApp(ABC):
    @abstractmethod
    def run(self):
        pass


class CLIApp(BaseApp):
    def run(self):
        cam = VirtualWebcam(False)
        signal.signal(signal.SIGINT, lambda *args, **kwargs: cam.toggle())
        signal.signal(signal.SIGQUIT, lambda *args, **kwargs: sys.exit(0))
        cam.start()


class GUIApp(BaseApp):
    def run(self):
        self.cam = VirtualWebcam(face_tracking=False)
        self.start()
        self.root = Tk()
        self.root.title("Webcam")
        self.root.geometry("300x500")
        self.root.configure(background="#dcdad5")
        self.face_tracking = False
        self.show_guides = False
        self._set_icon()
        self.render_windows()
        ttk.Style(self.root).theme_use("clam")
        self.root.mainloop()

    def render_windows(self):

        frm = ttk.Frame(self.root)
        frm.pack()

        self.render_buttons().pack(fill="both", expand=True)
        self.render_blur_control().pack(fill="x", expand=True)
        self.render_sepia_control().pack(fill="x", expand=True)
        self.render_face_following_control().pack(fill="both", expand=True)
        self.render_face_tracking_threshold_control().pack(fill="x", expand=True)
        self.render_show_guides_control().pack(fill="both", expand=True)

    def render_buttons(self):
        frm = ttk.Frame(self.root, padding=10)
        toggle_button = ttk.Button(
            frm,
            text="Toggle",
            command=lambda: self.cam.toggle(),
        )
        exit_button = ttk.Button(
            frm,
            text="Exit",
            command=lambda: self.exit(),
        )
        toggle_button.pack(padx=10, pady=10, fill="both", expand=True)
        exit_button.pack(padx=10, pady=10, fill="both", expand=True)
        return frm

    def render_blur_control(self):
        frm = ttk.Frame(self.root, padding=10)
        ttk.Label(frm, text="Blur").pack()
        ttk.Scale(
            frm,
            from_=0,
            to=100,
            orient=HORIZONTAL,
            command=self.set_blur,
            value=self.cam.blur,
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
            value=self.cam.sepia,
        ).pack(expand=True, fill="both")
        return frm

    def render_face_following_control(self):
        frm = ttk.Frame(self.root, padding=10)
        ttk.Checkbutton(
            frm,
            text="Face Tracking",
            onvalue=True,
            offvalue=False,
            command=self.toggle_face_following,
        ).pack(expand=True, fill="both")
        return frm

    def render_show_guides_control(self):
        frm = ttk.Frame(self.root, padding=10)
        ttk.Checkbutton(
            frm,
            text="Show guides",
            onvalue=True,
            offvalue=False,
            command=self.toggle_show_guides,
        ).pack(expand=True, fill="both")
        return frm

    def render_face_tracking_threshold_control(self):
        frm = ttk.Frame(self.root, padding=10)
        ttk.Label(frm, text="Threshold").pack()
        ttk.Scale(
            frm,
            from_=0,
            to=1,
            orient=HORIZONTAL,
            command=self.set_face_tracking_threshold,
            value=self.cam.face_tracking_threshold,
        ).pack(expand=True, fill="both")
        return frm

    def toggle_face_following(self):
        self.face_tracking = not self.face_tracking
        self.cam.face_tracking = self.face_tracking

    def toggle_show_guides(self):
        self.show_guides = not self.show_guides
        self.cam.show_guides = self.show_guides

    def set_blur(self, value):
        self.cam.blur = int(float(value))

    def set_sepia(self, value):
        self.cam.sepia = float(value)

    def set_face_tracking_threshold(self, value):
        self.cam.face_tracking_threshold = float(value)

    def start(self):
        self.main_thread = Thread(target=lambda: self.cam.start(), name="start_webcam")
        self.main_thread.start()

    def exit(self):
        self.cam.stop()
        self.main_thread.join()
        self.root.destroy()
        sys.exit(0)

    def _set_icon(self):
        img = Image("photo", file="src/assets/icon.png")
        self.root.tk.call("wm", "iconphoto", self.root._w, img)  # type: ignore


if __name__ == "__main__":
    app_class = GUIApp
    app_class().run()
