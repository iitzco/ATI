import tkinter as tk


class Menu(tk.Frame):
    def __init__(self, gui):
        tk.Frame.__init__(self, gui)
        self.gui = gui
        self.add_components()

    def add_components(self):
        self.main_label = tk.Label(self, text="Menu")
        self.button_save = tk.Button(
            self, text="Save", command=self.gui.save_file, width=10)
        self.button_undo = tk.Button(
            self, text="Undo", command=self.gui.undo, width=10)
        self.color_label = tk.Label(self)
        self.color_canvas = tk.Label(self, bg='#000000', width=3)
        self.color_slider = tk.Scale(
            self, from_=0, to=255, orient=tk.HORIZONTAL)
        self.button_zoom = tk.Button(
            self,
            text="Turn Zoom Mode ON",
            width=20,
            command=self.zoom_mode_trigger)
        self.button_mirror = tk.Button(
            self,
            text="Turn Mirror Mode ON",
            width=20,
            command=self.mirror_mode_trigger)
        self.button_unmark = tk.Button(
            self,
            text="Unmark",
            width=20,
            command=self.gui.unmark)
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
        self.main_label.grid(pady=5)
        self.button_save.grid(pady=5)
        self.button_undo.grid(pady=5)
        self.button_zoom.grid(pady=5)
        self.button_mirror.grid(pady=5)
        self.button_unmark.grid(pady=5)
        self.color_label.grid(pady=5)
        self.color_canvas.grid(pady=3)
        self.color_slider.grid(pady=5)

    def zoom_mode_trigger(self):
        self.gui.zoom_mode_trigger()
        self.button_zoom.config(
            text="Turn Zoom Mode {}".format("OFF" if self.gui.zoom else "ON"))

    def mirror_mode_trigger(self):
        self.gui.mirror_mode_trigger()
        self.button_mirror.config(text="Turn Mirror Mode {}".format(
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
            self.gui.studio.zoom(self.gui.studio.x_zoom,
                                 self.gui.studio.y_zoom)
