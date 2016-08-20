import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import ImageTk, Image

import sys

ZOOM_INTENSITY = 40


class GUI(tk.Frame):

    def __init__(self):
        tk.Frame.__init__(self)
        self.master.title("GUI")

        # Full Windows Size
        self.master.geometry("{0}x{1}+0+0".format(
            self.master.winfo_screenwidth(),
            self.master.winfo_screenheight()))

        self.pack(fill=tk.BOTH, expand=tk.YES)
        self.zoom = False
        self.mirror = False
        self.create_subframes()

    def create_subframes(self):
        self.original = ImageWorkspace(self, title="Original")
        self.studio = ImageWorkspace(self, title="Studio")
        self.menu = Menu(self)

        self.studio.pack(fill=tk.BOTH, expand=tk.YES, side=tk.LEFT)
        self.menu.pack(fill=tk.Y, expand=tk.YES, side=tk.RIGHT)

        self.menu.discover()
        self.original.discover()
        self.studio.discover()

    def revert(self):
        self.img = self.img.rotate(180)
        self.imagerotated = ImageTk.PhotoImage(self.img)
        self.studio.canvas.itemconfig(self.id, image=self.imagerotated)

    def load_file(self):
        fname = askopenfilename()
        if fname:
            self.img = Image.open(fname)
            self.original.show_image(self.img)

    def save_file(self):
        fname = asksaveasfilename()
        if fname:
            self.img.save(fname)

    def zoom_mode_trigger(self):
        self.zoom = not self.zoom
        if self.zoom:
            self.original.activate_zoom_mode()
            self.studio.activate_zoom_mode()
        else:
            self.original.deactivate_zoom_mode()
            self.studio.deactivate_zoom_mode()

    def mirror_mode_trigger(self):
        self.mirror = not self.mirror
        if not self.mirror:
            self.original.pack_forget()
        else:
            self.studio.pack_forget()
            self.original.pack(fill=tk.BOTH, expand=tk.YES, side=tk.LEFT)
            self.studio.pack(fill=tk.BOTH, expand=tk.YES, side=tk.LEFT)


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

    def show_image(self, img):
        imgtk = ImageTk.PhotoImage(img)
        self.image = imgtk
        self.canvas_main.create_image(0, 0, anchor='nw', image=self.image)
        self.canvas_main.config(scrollregion=self.canvas_main.bbox(tk.ALL))

    def map_bindings(self):
        if self.gui.zoom:
            self.canvas_main.bind("<B1-Motion>", self.zoom)
            self.canvas_main.bind("<Button-1>", self.zoom)
        else:
            pass

    def zoom(self, event):
        x = self.canvas_main.canvasx(event.x)
        y = self.canvas_main.canvasy(event.y)
        self.imgCropped = self.gui.img.crop(
            (x - ZOOM_INTENSITY,
             y - ZOOM_INTENSITY,
             x + ZOOM_INTENSITY,
             y + ZOOM_INTENSITY))
        self.imgCropped = self.imgCropped.resize(
            (self.winfo_width(), self.winfo_width()))
        self.imagetk = ImageTk.PhotoImage(self.imgCropped)
        self.canvas_zoom.create_image(0, 0, anchor='nw', image=self.imagetk)


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

    def discover(self):
        self.label.grid(pady=10)
        self.button_load.grid(pady=10)
        self.button_save.grid(pady=10)
        self.button_zoom.grid(pady=10)
        self.button_mirror.grid(pady=10)
        self.button_revert.grid(pady=10)
        self.color_slider.grid(pady=10)

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
