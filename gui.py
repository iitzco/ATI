import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.simpledialog import askinteger
import tkinter.messagebox
from PIL import ImageTk, Image

from components.menu import Menu
from components.image_workspace import ImageWorkspace, OriginalImageWorkspace, StudioImageWorkspace
from imgmanager import ImageManager

import sys

ZOOM_INTENSITY = 40


class GUI(tk.Frame):

    def __init__(self):
        tk.Frame.__init__(self)

        self.zoom = False
        self.mirror = False
        self.sync = tk.BooleanVar()
        self.sync.set(False)

        self.image_manager = ImageManager()

        self.init_main_frame()
        self.create_subframes()

    def init_main_frame(self):
        self.master.title("GUI")

        # Full Windows Size
        self.master.geometry("{0}x{1}+0+0".format(
            self.master.winfo_screenwidth(),
            self.master.winfo_screenheight()))
        self.pack(fill=tk.BOTH, expand=tk.YES)

    def create_subframes(self):
        self.original = OriginalImageWorkspace(self, title="Original")
        self.studio = StudioImageWorkspace(self, title="Studio")

        self.original.others.append(self.studio)
        self.studio.others.append(self.original)

        self.menu = Menu(self)

        self.studio.pack(fill=tk.BOTH, expand=tk.YES, side=tk.LEFT)
        self.menu.pack(fill=tk.Y, expand=tk.YES, side=tk.RIGHT)

        self.menu.discover()
        self.original.discover()
        self.studio.discover()

    def load_file(self, name=None):
        if (name):
            fname = name
        else:
            fname = askopenfilename()
        if fname:
            img = Image.open(fname)
            try:
                self.image_manager.load_image(img)
                self.load_images()
            except Exception:
                tkinter.messagebox.showinfo(
                    'Alert', 'Unsupported image format')

    def load_images(self):
        self.original.show_image()
        self.studio.show_image()

    def save_file(self):
        fname = asksaveasfilename()
        if fname:
            self.image_manager.get_image().save(fname)

    def zoom_mode_trigger(self):
        self.zoom = not self.zoom
        if self.zoom:
            self.original.activate_zoom_mode()
            self.studio.activate_zoom_mode()
            if self.mirror:
                self.menu.button_sync.grid(pady=5)
        else:
            self.original.deactivate_zoom_mode()
            self.studio.deactivate_zoom_mode()
            self.menu.button_sync.grid_forget()

    def mirror_mode_trigger(self):
        self.mirror = not self.mirror
        if not self.mirror:
            self.original.pack_forget()
            self.menu.button_sync.grid_forget()
        else:
            self.studio.pack_forget()
            self.original.pack(fill=tk.BOTH, expand=tk.YES, side=tk.LEFT)
            self.studio.pack(fill=tk.BOTH, expand=tk.YES, side=tk.LEFT)
            if self.zoom:
                self.menu.button_sync.grid(pady=5)

    def negative(self):
        self.image_manager.negative()
        self.studio.show_image()

    def umbral(self):
        value = askinteger("Umbral", "Umbral Value?", minvalue=0, maxvalue=255)
        try:
            self.image_manager.umbral(value)
            self.studio.show_image()
        except Exception:
            tkinter.messagebox.showinfo(
                'Alert', 'Unsupported operation')


if __name__ == "__main__":
    gui = GUI()
    if (len(sys.argv) > 1 and sys.argv[1]):
        gui.load_file(sys.argv[1])
    gui.mainloop()
