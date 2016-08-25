import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import tkinter.messagebox
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
        self.image_manager.reverse()
        self.studio.show_image()

    def load_file(self, name=None):
        if (name):
            fname = name
        else:
            fname = askopenfilename()
        if fname:
            img = Image.open(fname)
            try:
                self.image_manager.load_image(img)
            except Exception:
                tkinter.messagebox.showinfo(
                    'Alert', 'Unsupported image format')
            self.load_images()

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
        self.x_selection, self.y_selection = (None, None)

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
            self.canvas_zoom.bind("<B1-Motion>", self.color_in_zoom_event)
            self.canvas_zoom.bind("<Button-1>", self.color_in_zoom_event)
            self.canvas_zoom.bind("<Button-2>", self.selection_event_in_zoom)
        else:
            self.canvas_main.bind("<B1-Motion>", self.color_event)
            self.canvas_main.bind("<Button-1>", self.color_event)
            self.canvas_main.bind("<Button-2>", self.selection_event)

    def selection_event(self, event):
        self.common_selection_event(event, self.canvas_main, False)

    def selection_event_in_zoom(self, event):
        self.common_selection_event(event, self.canvas_zoom, True)

    def common_selection_event(self, event, canvas, in_zoom):
        if not self.x_selection or not self.y_selection:
            self.x_selection = canvas.canvasx(event.x)
            self.y_selection = canvas.canvasy(event.y)
        else:
            x = canvas.canvasx(event.x)
            y = canvas.canvasy(event.y)
            img = self.get_selection_img(
                self.x_selection, self.y_selection, x, y, in_zoom)
            self.rectangle_selection_id = canvas.create_rectangle(
                self.x_selection, self.y_selection, x, y)
            ret = tkinter.messagebox.askyesno("Selection", "Want statistics?")
            if ret:
                t = self.gui.image_manager.get_statistics(img)
                tkinter.messagebox.showinfo(
                    "Statistics", "# of Pixel: {} --- Mean: {}".format(t[0], t[1]))
            ret = tkinter.messagebox.askyesno(
                "Confirm selection",
                "Want to create new image with that selection? ATTENTION: unsaved studio image will be lost.")
            if ret:
                self.gui.image_manager.load_image(img)
                self.gui.load_images()
            canvas.delete(self.rectangle_selection_id)
            self.x_selection, self.y_selection = (None, None)

    def color_in_zoom_event(self, event):
        x = self.canvas_zoom.canvasx(event.x)
        y = self.canvas_zoom.canvasy(event.y)
        (x, y) = self.gui.image_manager.get_outbound_pixel(
            self.x_zoom, self.y_zoom, x, y, self.winfo_width(), self.winfo_width())
        self.color_common_event(x, y)

    def zoom_event(self, event):
        x = self.canvas_main.canvasx(event.x)
        y = self.canvas_main.canvasy(event.y)
        self.x_zoom = x
        self.y_zoom = y
        self.zoom(x, y)
        if self.gui.sync.get():
            for each in self.others:
                each.x_zoom = x
                each.y_zoom = y
                each.zoom(x, y)

    def zoom(self, x, y):
        pass

    def color_event(self, event):
        x = self.canvas_main.canvasx(event.x)
        y = self.canvas_main.canvasy(event.y)
        self.color_common_event(x, y)

    def color_common_event(self, x, y):
        self.gui.menu.x_pixel = x
        self.gui.menu.y_pixel = y
        self.gui.menu.show_color(self.get_pixel_color(x, y))

    def get_pixel_color(self, x, y):
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

    def get_pixel_color(self, x, y):
        return self.gui.image_manager.get_original_pixel_color(int(x), int(y))

    def get_selection_img(self, x_o, y_o, x_f, y_f, in_zoom):
        if in_zoom:
            t = (
                self.x_zoom,
                self.y_zoom,
                self.winfo_width(),
                self.winfo_width())
        else:
            t = None
        return self.gui.image_manager.get_original_selection(
            x_o, y_o, x_f, y_f, in_zoom, t)


class StudioImageWorkspace(ImageWorkspace):

    def zoom(self, x, y):
        zoomed_img = self.gui.image_manager.get_zoomed_img(
            x, y, self.winfo_width(), self.winfo_width())
        self.show_zoomed_image(zoomed_img)

    def show_image(self):
        self.imageTk = ImageTk.PhotoImage(self.gui.image_manager.get_image())
        self.canvas_main.create_image(0, 0, anchor='nw', image=self.imageTk)
        self.canvas_main.config(scrollregion=self.canvas_main.bbox(tk.ALL))

    def get_pixel_color(self, x, y):
        return self.gui.image_manager.get_img_pixel_color(int(x), int(y))

    def get_selection_img(self, x_o, y_o, x_f, y_f, in_zoom):
        if in_zoom:
            t = (
                self.x_zoom,
                self.y_zoom,
                self.winfo_width(),
                self.winfo_width())
        else:
            t = None
        return self.gui.image_manager.get_studio_selection(
            x_o, y_o, x_f, y_f, in_zoom, t)


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
            self, text="Negative", command=self.gui.revert, width=10)
        self.color_label = tk.Label(self)
        self.color_canvas = tk.Label(self, bg='#000000', width=3)
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

        self.first = True
        self.interaction = False
        self.color_slider.config(command=self.update_color)

    def discover(self):
        self.label.grid(pady=5)
        self.button_load.grid(pady=5)
        self.button_save.grid(pady=5)
        self.button_zoom.grid(pady=5)
        self.button_mirror.grid(pady=5)
        self.button_revert.grid(pady=5)
        self.color_label.grid(pady=5)
        self.color_canvas.grid(pady=3)
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

    def show_color(self, color, from_slider=False, from_original=False):
        c_in_hex = [hex(e)[-2:] for e in [x + 256 for x in color]]
        label_c = '#' + ''.join(c_in_hex)
        self.color_label.config(text=label_c)
        self.color_canvas.config(bg=label_c)
        if not from_slider and color[0] == color[1] == color[2]:
            self.interaction = False
            self.color_slider.set(color[0])

    def update_color(self, event):
        if self.first:
            self.first = False
            return
        if not self.interaction:
            self.interaction = True
            return
        c = self.color_slider.get()
        self.gui.image_manager.update_img_pixel(
            int(self.x_pixel), int(self.y_pixel), c)
        self.show_color((c, c, c), from_slider=True)
        self.gui.studio.show_image()
        if self.gui.zoom:
            self.gui.studio.zoom(
                self.gui.studio.x_zoom,
                self.gui.studio.y_zoom)

if __name__ == "__main__":
    gui = GUI()
    if (len(sys.argv) > 1 and sys.argv[1]):
        gui.load_file(sys.argv[1])
    gui.mainloop()
