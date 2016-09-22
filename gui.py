import tkinter as tk
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.simpledialog import askinteger, askfloat
import tkinter.messagebox
from PIL import ImageTk, Image

from components.menubar import MenuBar
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
        self.create_menubar()
        self.create_subframes()

    def create_menubar(self):
        self.menubar = MenuBar(self)
        self.master.config(menu=self.menubar)

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

    def has_img(self):
        return self.image_manager.has_img()

    def load_rectangle(self):
        self._load_figure(self.image_manager.load_rectangle)

    def load_circle(self):
        self._load_figure(self.image_manager.load_circle)

    def load_black(self):
        self._load_figure(self.image_manager.load_black)

    def load_white(self):
        self._load_figure(self.image_manager.load_white)

    def _load_figure(self, f):
        w = askinteger("Size Specification", "Width?", minvalue=0)
        h = askinteger("Size Specification", "Heigth?", minvalue=0)

        if not w or not h:
            return
        f(w, h)
        self.load_images()

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

    def undo(self):
        try:
            self.image_manager.undo()
        except Exception:
            tkinter.messagebox.showinfo(
                'Alert', 'Already at latest change')
        self.studio.show_image()

    def to_bw(self):
        self.image_manager.to_bw()
        self.studio.show_image()

    def get_auxiliar_img_manager(self):
        fname = askopenfilename()
        if fname:
            img = Image.open(fname)
            aux_image_manager = ImageManager()
            try:
                aux_image_manager.load_image(img)
            except Exception:
                tkinter.messagebox.showinfo(
                    'Alert', 'Unsupported image format')
        return aux_image_manager

    def add_img(self):
        aux_image_manager = self.get_auxiliar_img_manager()
        self.image_manager.add_img(aux_image_manager)
        self.studio.show_image()

    def substract_img(self):
        aux_image_manager = self.get_auxiliar_img_manager()
        self.image_manager.substract_img(aux_image_manager)
        self.studio.show_image()

    def multiply_img(self):
        aux_image_manager = self.get_auxiliar_img_manager()
        self.image_manager.multiply_img(aux_image_manager)
        self.studio.show_image()

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

    def global_umbral(self):
        v, it = self.image_manager.get_global_umbral()
        ret = tkinter.messagebox.askyesno(
            "Global umbralization",
            "Best umbral value is {}, founded in {} iterations. Wish to umbralize?".format(v, it))
        if ret:
            self.image_manager.umbral(int(v))
            self.studio.show_image()

    def otsu_umbral(self):
        v = self.image_manager.get_otsu_umbral()
        ret = tkinter.messagebox.askyesno(
            "Otsu umbralization",
            "Best umbral value is {}. Wish to umbralize?".format(v))
        if ret:
            self.image_manager.umbral(int(v))
            self.studio.show_image()

    def power(self):
        value = askfloat("Power", "Power Value?", minvalue=0, maxvalue=2.5)
        try:
            self.image_manager.power(value)
            self.studio.show_image()
        except Exception:
            tkinter.messagebox.showinfo(
                'Alert', 'Unsupported operation')

    def product(self):
        value = askfloat("Product", "Product Value?", minvalue=0)
        try:
            self.image_manager.product(value)
            self.studio.show_image()
        except Exception:
            tkinter.messagebox.showinfo(
                'Alert', 'Unsupported operation')

    def compression(self):
        self.image_manager.compression()
        self.studio.show_image()

    def histogram(self):
        values = self.image_manager.get_histogram_values()

        plt.figure()
        if self.image_manager.image.is_bw():
            plt.xlim(0, 255)
            plt.hist(values, bins=256)
        else:
            plt.subplot(311)
            plt.hist([e for i, e in enumerate(values) if i %
                      3 == 0], bins=256, facecolor='red')
            plt.subplot(312)
            plt.hist([e for i, e in enumerate(values) if i %
                      3 == 1], bins=256, facecolor='green')
            plt.subplot(313)
            plt.hist([e for i, e in enumerate(values) if i %
                      3 == 2], bins=256, facecolor='blue')
        plt.show()

    def enhance_contrast(self):
        value1 = askinteger(
            "Contrast Enhancement",
            "r1 value?",
            minvalue=0,
            maxvalue=255)
        value2 = askinteger(
            "Contrast Enhancement",
            "r2 value?",
            minvalue=0,
            maxvalue=255)

        if not value1 or not value2:
            return

        if value2 < value1:
            value1, value2 = value2, value1

        self.image_manager.enhance_contrast(value1, value2)
        self.studio.show_image()

    def equalize(self):
        self.image_manager.equalize()
        self.studio.show_image()

    def input_multiplicative_noise(self, t):
        param = askfloat("{} Noise".format(t), "Parameter?")
        percentage = askfloat(
            "{} Noise".format(t),
            "Percentage?",
            minvalue=0,
            maxvalue=100)
        return (param, percentage)

    def exponential_noise(self):
        param, percentage = self.input_multiplicative_noise("Exponential")
        if any([x is None for x in [param, percentage]]):
            return
        self.image_manager.exponential_noise(param, percentage)
        self.studio.show_image()

    def rayleigh_noise(self):
        param, percentage = self.input_multiplicative_noise("Rayleigh")
        if any([x is None for x in [param, percentage]]):
            return
        self.image_manager.rayleigh_noise(param, percentage)
        self.studio.show_image()

    def gauss_noise(self):
        intensity = askfloat("Gauss Noise", "Intensity?")
        percentage = askfloat(
            "Gauss Noise",
            "Percentage?",
            minvalue=0,
            maxvalue=100)
        if any([x is None for x in [intensity, percentage]]):
            return
        self.image_manager.gauss_noise(intensity, percentage)
        self.studio.show_image()

    def salt_pepper_noise(self):
        p0 = askfloat("S&P Noise", "P0?", minvalue=0, maxvalue=255)
        p1 = askfloat("S&P Noise", "P1?", minvalue=0, maxvalue=255)
        percentage = askfloat(
            "Gauss Noise",
            "Percentage?",
            minvalue=0,
            maxvalue=100)
        if any([x is None for x in [p0, p1, percentage]]):
            return
        if p1 < p0:
            p0, p1 = p1, p0

        self.image_manager.salt_pepper_noise(p0, p1, percentage)
        self.studio.show_image()

    def _common_filter(self, f):
        size = askinteger("Window", "Window Size?", minvalue=0)
        if not size or not size % 2:
            return

        f(size)
        self.studio.show_image()

    def mean_filter(self):
        self._common_filter(self.image_manager.mean_filter)

    def median_filter(self):
        self._common_filter(self.image_manager.median_filter)

    def border_filter(self):
        self._common_filter(self.image_manager.border_filter)

    def gauss_filter(self):
        sigma = askfloat("Gauss", "Sigma?", minvalue=0)
        size = askinteger(
            "Gauss", "Window Size? Suggested {} (2*sigma + 1)".format(2 * sigma + 1), minvalue=0)
        if not sigma or not size or not size % 2:
            return
        self.image_manager.gauss_filter(size, sigma)
        self.studio.show_image()

    def anisotropic_diffusion(self, f):
        sigma = askfloat("Parameters", "Sigma?", minvalue=0)
        times = askinteger("Parameters", "Iterations?", minvalue=0)
        if not sigma or not times:
            return
        f(sigma, times)
        self.studio.show_image()

    def isotropic_diffusion(self):
        times = askinteger("Parameters", "Iterations?", minvalue=0)
        if not times:
            return
        self.image_manager.isotropic_diffusion(times)
        self.studio.show_image()

    def leclerc_anisotropic_diffusion(self):
        self.anisotropic_diffusion(self.image_manager.leclerc_anisotropic_diffusion)

    def lorentziano_anisotropic_diffusion(self):
        self.anisotropic_diffusion(self.image_manager.lorentziano_anisotropic_diffusion)

    def prewitt_method(self):
        self.image_manager.prewitt_method()
        self.studio.show_image()

    def sobel_method(self):
        self.image_manager.sobel_method()
        self.studio.show_image()

    def laplacian_mask(self):
        self.image_manager.laplacian_mask()
        self.studio.show_image()

    def LoG_mask(self):
        sigma = askfloat("Parameters", "Sigma?", minvalue=0)
        size = askinteger(
            "Parameters", "Window Size? Suggested {} (2*sigma + 1)".format(2 * sigma + 1), minvalue=0)
        if not sigma or not size or not size % 2:
            return
        self.image_manager.LoG_mask(size, sigma)
        self.studio.show_image()

    def laplacian_method(self):
        self.image_manager.laplacian_method()
        self.studio.show_image()

    def laplacian_pending_method(self):
        umbral = askinteger("Parameters", "Umbral?", minvalue=0, maxvalue=255)
        if not umbral:
            return
        self.image_manager.laplacian_pending_method(umbral)
        self.studio.show_image()

    def kirsh_directional_method(self):
        self.image_manager.kirsh_directional_method()
        self.studio.show_image()

    def prewitt_directional_method(self):
        self.image_manager.prewitt_directional_method()
        self.studio.show_image()

    def sobel_directional_method(self):
        self.image_manager.sobel_directional_method()
        self.studio.show_image()

    def alternative_directional_method(self):
        self.image_manager.alternative_directional_method()
        self.studio.show_image()

    def sobel_x_img(self):
        self.image_manager.sobel_x_img()
        self.studio.show_image()

    def sobel_y_img(self):
        self.image_manager.sobel_y_img()
        self.studio.show_image()


if __name__ == "__main__":
    gui = GUI()
    if (len(sys.argv) > 1 and sys.argv[1]):
        gui.load_file(sys.argv[1])
    gui.mainloop()
