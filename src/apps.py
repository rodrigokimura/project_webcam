import signal
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Thread
from tkinter import HORIZONTAL, Image, IntVar, Tk, ttk

from dotenv import load_dotenv
from PIL import Image as PillowImage
from pystray import Icon, Menu, MenuItem

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


load_dotenv()


class GUIApp(BaseApp):
    def __init__(self) -> None:
        super().__init__()
        self.window = Tk()
        self.window.title("Webcam")
        self.window.geometry("300x500")
        self.window.configure(background="#dcdad5")

        # UI state
        self.webcam_visibility = IntVar()
        self.face_tracking = IntVar()
        self.show_guides = IntVar()

        self.webcam = VirtualWebcam(face_tracking=False)

        self.window.protocol("WM_DELETE_WINDOW", self.hide)
        signal.signal(signal.SIGQUIT, lambda *args, **kwargs: self.exit())

        self._set_icon()
        self._build_layout()

    def run(self):
        if self.locked:
            return
        self.lock()
        self._start_webcam()
        ttk.Style(self.window).theme_use("clam")
        self.hide()
        self.window.mainloop()

    def lock(self):
        open("lock", "a").close()

    def unlock(self):
        Path("lock").unlink(missing_ok=True)
        # os.remove("lock", missing_ok=True)

    @property
    def locked(self):
        return Path("lock").exists()
        # return os.path.exists("lock")

    def _show_systray_icon(self):
        im = PillowImage.open("src/assets/icon.png")
        self.icon = Icon(
            "virtual_webcam",
            icon=im,
            menu=Menu(
                MenuItem("Show", self.show),
                MenuItem(
                    "Toggle webcam",
                    self.webcam.toggle,
                    checked=lambda _: self.webcam.running,
                ),
                MenuItem(
                    "Toggle face tracking",
                    self.toggle_face_tracking,
                    checked=lambda _: self.webcam.face_tracking,
                ),
                MenuItem(
                    "Toggle face guides",
                    self.toggle_show_guides,
                    checked=lambda _: self.webcam.show_guides,
                ),
                MenuItem("Exit", self.exit),
            ),
        )
        self.icon.run()

    def _build_layout(self):
        frm = ttk.Frame(self.window)
        frm.pack()
        self.render_toggle_webcam_control().pack(fill="both", expand=True)
        self.render_blur_control().pack(fill="x", expand=True)
        self.render_sepia_control().pack(fill="x", expand=True)
        self.render_face_following_control().pack(fill="both", expand=True)
        self.render_face_tracking_threshold_control().pack(fill="x", expand=True)
        self.render_show_guides_control().pack(fill="both", expand=True)
        self.render_buttons().pack(fill="both", expand=True)

    def render_buttons(self):
        frm = ttk.Frame(self.window, padding=10)
        exit_button = ttk.Button(
            frm,
            text="Exit",
            command=lambda: self.exit(),
        )
        hide_button = ttk.Button(
            frm,
            text="Hide",
            command=lambda: self.hide(),
        )
        hide_button.pack(padx=10, pady=10, fill="both", expand=True)
        exit_button.pack(padx=10, pady=10, fill="both", expand=True)
        return frm

    def render_blur_control(self):
        frm = ttk.Frame(self.window, padding=10)
        ttk.Label(frm, text="Blur").pack()
        ttk.Scale(
            frm,
            from_=0,
            to=100,
            orient=HORIZONTAL,
            command=self.set_blur,
            value=self.webcam.blur,
        ).pack(expand=True, fill="both")
        return frm

    def render_sepia_control(self):
        frm = ttk.Frame(self.window, padding=10)
        ttk.Label(frm, text="Sepia").pack()
        ttk.Scale(
            frm,
            from_=0,
            to=1,
            orient=HORIZONTAL,
            command=self.set_sepia,
            value=self.webcam.sepia,
        ).pack(expand=True, fill="both")
        return frm

    def render_face_following_control(self):
        frm = ttk.Frame(self.window, padding=10)
        ttk.Checkbutton(
            frm,
            text="Face Tracking",
            variable=self.face_tracking,
            command=self.toggle_face_tracking,
        ).pack(expand=True, fill="both")
        return frm

    def render_show_guides_control(self):
        frm = ttk.Frame(self.window, padding=10)
        ttk.Checkbutton(
            frm,
            text="Show guides",
            variable=self.show_guides,
            command=self.toggle_show_guides,
        ).pack(expand=True, fill="both")
        return frm

    def render_toggle_webcam_control(self):
        frm = ttk.Frame(self.window, padding=10)
        ttk.Checkbutton(
            frm,
            text="Toggle webcam",
            variable=self.webcam_visibility,
            command=self.toggle_webcam_visibility,
        ).pack(expand=True, fill="both")
        return frm

    def render_face_tracking_threshold_control(self):
        frm = ttk.Frame(self.window, padding=10)
        ttk.Label(frm, text="Threshold").pack()
        ttk.Scale(
            frm,
            from_=0,
            to=1,
            orient=HORIZONTAL,
            command=self.set_face_tracking_threshold,
            value=self.webcam.face_tracking_threshold,
        ).pack(expand=True, fill="both")
        return frm

    def toggle_face_tracking(self):
        self.webcam.face_tracking = not self.webcam.face_tracking

    def toggle_show_guides(self):
        self.webcam.show_guides = not self.webcam.show_guides

    def set_blur(self, value):
        self.webcam.blur = int(float(value))

    def set_sepia(self, value):
        self.webcam.sepia = float(value)

    def set_face_tracking_threshold(self, value):
        self.webcam.face_tracking_threshold = float(value)

    def toggle_webcam_visibility(self):
        self.webcam.toggle()

    def _start_webcam(self):
        self.main_thread = Thread(
            target=lambda: self.webcam.start(), name="start_webcam"
        )
        self.main_thread.start()

    def update_window_state(self):
        self.webcam_visibility.set(int(self.webcam.running))
        self.show_guides.set(int(self.webcam.show_guides))
        self.face_tracking.set(int(self.webcam.face_tracking))

    def show(self):
        self.icon.visible = False
        self.icon.stop()
        self.update_window_state()
        self.window.after(0, self.window.deiconify)

    def hide(self):
        self.window.withdraw()
        self._show_systray_icon()

    def exit(self):
        self.unlock()
        self.webcam.stop()
        self.main_thread.join()
        self.window.destroy()
        sys.exit(0)

    def _set_icon(self):
        img = Image("photo", file="src/assets/icon.png")
        self.window.tk.call("wm", "iconphoto", self.window._w, img)  # type: ignore


if __name__ == "__main__":
    app_class = GUIApp
    app_class().run()
