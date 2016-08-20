import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import ImageTk, Image

from imgprocessor import ImageManager

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

    def revert(self):
        self.image_manager.rotate(180)
        self.studio.show_image()

    def load_file(self):
        fname = askopenfilename()
        if fname:
            img = Image.open(fname)
            self.image_manager.load_image(img)
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


class ImageWorkspace(tk.Frame):

    def __init__(self, gui, title):
        tk.Frame.__init__(self, gui)
        self.gui = gui
        self.label = tk.Label(self, text=title)
        self.inner_frame = tk.Frame(self)
        self.xscroll = tk.Scrollbar(self.inner_frame, orient=tk.HORIZONTAL)
        self.yscroll = tk.Scrollbar(self.inner_frame)
        self.canvas_main = tk.Canvas(
            self.inner_frame,
            xscrollcommand=self.xscroll.set,
            yscrollcommand=self.yscroll.set)
        self.canvas_zoom = tk.Canvas(self)
        self.others = []

    def discover(self):
        self.label.pack(side=tk.TOP)
        self.inner_frame.pack(fill=tk.BOTH, expand=tk.YES)
        self.yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.xscroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.xscroll.config(command=self.canvas_main.xview)
        self.yscroll.config(command=self.canvas_main.yview)
        self.canvas_main.config(
            xscrollcommand=self.xscroll.set,
            yscrollcommand=self.yscroll.set)
        self.canvas_main.pack(fill=tk.BOTH, expand=tk.YES)
        self.map_bindings()

    def activate_zoom_mode(self):
        self.canvas_zoom.pack(fill=tk.BOTH, expand=tk.YES)
        self.map_bindings()

    def deactivate_zoom_mode(self):
        self.canvas_zoom.pack_forget()
        self.map_bindings()

    def show_image(self):
        pass

    def show_zoomed_image(self, img):
        self.imageTk_zoomed = ImageTk.PhotoImage(img)
        self.canvas_zoom.create_image(
            0, 0, anchor='nw', image=self.imageTk_zoomed)

    def map_bindings(self):
        if self.gui.zoom:
            self.canvas_main.bind("<B1-Motion>", self.zoom_event)
            self.canvas_main.bind("<Button-1>", self.zoom_event)
        else:
            pass

    def zoom_event(self, event):
        x = self.canvas_main.canvasx(event.x)
        y = self.canvas_main.canvasy(event.y)
        self.zoom(x, y)
        if self.gui.sync.get():
            for each in self.others:
                each.zoom(x, y)

    def zoom(self, x, y):
        pass


class OriginalImageWorkspace(ImageWorkspace):

    def zoom(self, x, y):
        zoomed_img = self.gui.image_manager.get_zoomed_original(
            x, y, self.winfo_width(), self.winfo_width())
        self.show_zoomed_image(zoomed_img)

    def show_image(self):
        self.imageTk = ImageTk.PhotoImage(
            self.gui.image_manager.get_original())
        self.canvas_main.create_image(0, 0, anchor='nw', image=self.imageTk)
        self.canvas_main.config(scrollregion=self.canvas_main.bbox(tk.ALL))


class StudioImageWorkspace(ImageWorkspace):

    def zoom(self, x, y):
        zoomed_img = self.gui.image_manager.get_zoomed_img(
            x, y, self.winfo_width(), self.winfo_width())
        self.show_zoomed_image(zoomed_img)

    def show_image(self):
        self.imageTk = ImageTk.PhotoImage(self.gui.image_manager.get_image())
        self.canvas_main.create_image(0, 0, anchor='nw', image=self.imageTk)
        self.canvas_main.config(scrollregion=self.canvas_main.bbox(tk.ALL))


class Menu(tk.Frame):

    def __init__(self, gui):
        tk.Frame.__init__(self, gui)
        self.gui = gui
        self.add_components()

    def add_components(self):
        self.label = tk.Label(self, text="Menu")
        self.button_load = tk.Button(
            self,
            text="Load Image",
            command=self.gui.load_file,
            width=10)
        self.button_save = tk.Button(
            self, text="Save", command=self.gui.save_file, width=10)
        self.button_revert = tk.Button(
            self, text="Revert", command=self.gui.revert, width=10)
        self.color_slider = tk.Scale(
            self, from_=0, to=255, orient=tk.HORIZONTAL)
        self.button_zoom = tk.Button(self, text="Turn Zoom Mode ON", width=20,
                                     command=self.zoom_mode_trigger)
        self.button_mirror = tk.Button(
            self,
            text="Turn Mirror Mode ON",
            width=20,
            command=self.mirror_mode_trigger)
        self.button_sync = tk.Checkbutton(
            self,
            text="Sync zoom",
            variable=self.gui.sync,
            onvalue=True,
            offvalue=False)

    def discover(self):
        self.label.grid(pady=5)
        self.button_load.grid(pady=5)
        self.button_save.grid(pady=5)
        self.button_zoom.grid(pady=5)
        self.button_mirror.grid(pady=5)
        self.button_revert.grid(pady=5)
        self.color_slider.grid(pady=5)

    def zoom_mode_trigger(self):
        self.gui.zoom_mode_trigger()
        self.button_zoom.config(
            text="Turn Zoom Mode {}".format(
                "OFF" if self.gui.zoom else "ON"))

    def mirror_mode_trigger(self):
        self.gui.mirror_mode_trigger()
        self.button_mirror.config(
            text="Turn Mirror Mode {}".format(
                "OFF" if self.gui.mirror else "ON"))

if __name__ == "__main__":
    GUI().mainloop()
