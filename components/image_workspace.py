import tkinter as tk
import tkinter.messagebox
from PIL import ImageTk, Image

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
                self.x_selection, self.y_selection, x, y, outline='blue')
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

    def color_event(self, event):
        x = self.canvas_main.canvasx(event.x)
        y = self.canvas_main.canvasy(event.y)
        self.color_common_event(x, y)

    def color_common_event(self, x, y):
        self.gui.menu.x_pixel = x
        self.gui.menu.y_pixel = y
        self.gui.menu.show_color(self.get_pixel_color(x, y))

    def zoom(self, x, y):
        pass

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

