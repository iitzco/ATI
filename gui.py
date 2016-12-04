import re
import os
import tkinter as tk
import cv2
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename, asksaveasfilename, askdirectory
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

        self.contour_detection = tk.BooleanVar()
        self.contour_detection.set(False)

        self.selection_for_video = False
        self.hsv_tracking = False

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
            self.master.winfo_screenwidth(), self.master.winfo_screenheight()))
        self.pack(fill=tk.BOTH, expand=tk.YES)

    def create_subframes(self):
        self.original = OriginalImageWorkspace(self, title="Original")
        self.studio = StudioImageWorkspace(self, title="Studio")

        self.original.others.append(self.studio)
        self.studio.others.append(self.original)

        self.menu = Menu(self)

        self.studio.pack(fill=tk.BOTH, expand=tk.YES, side=tk.LEFT)
        self.menu.pack(fill=tk.Y, expand=tk.NO, side=tk.RIGHT)

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
                tkinter.messagebox.showinfo('Alert',
                                            'Unsupported image format')
            self.load_images()

    def load_images(self):
        self.original.show_image()
        self.studio.show_image()

    def save_file(self):
        fname = asksaveasfilename()
        if fname:
            img = self.image_manager.get_image().save(fname)

    def save_file_with_marks(self):
        fname = asksaveasfilename()
        if fname:
            curr_img = self.image_manager.get_image()
            pixel_list = [each[1] for each in self.studio.pixel_list]
            img = self.image_manager.get_image_with_marks(curr_img, pixel_list)
            img.save(fname)

    def undo(self):
        try:
            self.image_manager.undo()
        except Exception:
            tkinter.messagebox.showinfo('Alert', 'Already at latest change')
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
                tkinter.messagebox.showinfo('Alert',
                                            'Unsupported image format')
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
            tkinter.messagebox.showinfo('Alert', 'Unsupported operation')

    def global_umbral(self):
        v, it = self.image_manager.get_global_umbral()
        ret = tkinter.messagebox.askyesno(
            "Global umbralization",
            "Best umbral value is {}, founded in {} iterations. Wish to umbralize?".
            format(v, it))
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
            tkinter.messagebox.showinfo('Alert', 'Unsupported operation')

    def product(self):
        value = askfloat("Product", "Product Value?", minvalue=0)
        try:
            self.image_manager.product(value)
            self.studio.show_image()
        except Exception:
            tkinter.messagebox.showinfo('Alert', 'Unsupported operation')

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
            plt.hist(
                [e for i, e in enumerate(values) if i % 3 == 0],
                bins=256,
                facecolor='red')
            plt.subplot(312)
            plt.hist(
                [e for i, e in enumerate(values) if i % 3 == 1],
                bins=256,
                facecolor='green')
            plt.subplot(313)
            plt.hist(
                [e for i, e in enumerate(values) if i % 3 == 2],
                bins=256,
                facecolor='blue')
        plt.show()

    def enhance_contrast(self):
        value1 = askinteger(
            "Contrast Enhancement", "r1 value?", minvalue=0, maxvalue=255)
        value2 = askinteger(
            "Contrast Enhancement", "r2 value?", minvalue=0, maxvalue=255)

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
            "{} Noise".format(t), "Percentage?", minvalue=0, maxvalue=100)
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
            "Gauss Noise", "Percentage?", minvalue=0, maxvalue=100)
        if any([x is None for x in [intensity, percentage]]):
            return
        self.image_manager.gauss_noise(intensity, percentage)
        self.studio.show_image()

    def salt_pepper_noise(self):
        p0 = askfloat("S&P Noise", "P0?", minvalue=0, maxvalue=255)
        p1 = askfloat("S&P Noise", "P1?", minvalue=0, maxvalue=255)
        percentage = askfloat(
            "Gauss Noise", "Percentage?", minvalue=0, maxvalue=100)
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
            "Gauss",
            "Window Size? Suggested {} (2*sigma + 1)".format(2 * sigma + 1),
            minvalue=0)
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
        self.anisotropic_diffusion(
            self.image_manager.leclerc_anisotropic_diffusion)

    def lorentziano_anisotropic_diffusion(self):
        self.anisotropic_diffusion(
            self.image_manager.lorentziano_anisotropic_diffusion)

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
            "Parameters",
            "Window Size? Suggested {} (2*sigma + 1)".format(2 * sigma + 1),
            minvalue=0)
        if not sigma or not size or not size % 2:
            return
        self.image_manager.LoG_mask(size, sigma)
        self.studio.show_image()

    def rotate(self):
        angle = askfloat("Parameters", "Angle?", minvalue=-360, maxvalue=360)
        if not angle:
            return
        self.image_manager.rotate(angle)
        self.load_images()

    def laplacian_method(self):
        self.image_manager.laplacian_method()
        self.studio.show_image()

    def laplacian_pending_method(self):
        umbral = askinteger("Parameters", "Umbral?", minvalue=0)
        if not umbral:
            return
        self.image_manager.laplacian_pending_method(umbral)
        self.studio.show_image()

    def LoG_method(self):
        sigma = askfloat("Parameters", "Sigma?", minvalue=0)
        size = askinteger(
            "Parameters",
            "Window Size? Suggested {} (2*sigma + 1)".format(2 * sigma + 1),
            minvalue=0)
        umbral = askinteger("Parameters", "Umbral?", minvalue=0)
        if not sigma or not size or not size % 2 or umbral is None or umbral is 0:
            return
        self.image_manager.LoG_method(size, sigma, umbral)
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

    def canny_method(self):
        self.image_manager.canny_method()
        self.studio.show_image()

    def canny_hysteresis_method(self):
        t1 = askfloat("Parameters", "t1?")
        t2 = askfloat("Parameters", "t2?")
        if t1 is None or t2 is None:
            return
        self.image_manager.canny_hysteresis_method(t1, t2)
        self.studio.show_image()

    def sobel_x_img(self):
        self.image_manager.sobel_x_img()
        self.studio.show_image()

    def sobel_y_img(self):
        self.image_manager.sobel_y_img()
        self.studio.show_image()

    def unmark(self):
        self.studio.unmark_pixels()
        self.studio.unmark_lines()
        self.studio.unmark_circles()
        self.menu.remove_unmark_button()

    def harris_method(self):
        umbral = askfloat("Parameters", "Percentage?", minvalue=0, maxvalue=1)
        if umbral is None:
            return
        p = self.image_manager.harris_method(umbral)
        self.studio.mark_pixels(p, 3)
        self.menu.show_unmark_button()

    def susan_method(self):
        umbral = askfloat("Parameters", "Umral (t)?", minvalue=0)
        reference = askfloat(
            "Parameters",
            "Reference? (~0.4 for border detection, ~0.5 for corner detection)",
            minvalue=0,
            maxvalue=1)
        if umbral is None or reference is None:
            return
        p = self.image_manager.susan_method(umbral, reference)
        self.studio.mark_pixels(p, 3)
        self.menu.show_unmark_button()

    def sift_method(self):
        fname = askopenfilename()
        if fname:
            img = cv2.imread(fname)
            sift_img = self.image_manager.sift_method(img)
            fname = asksaveasfilename()
            if fname:
                cv2.imwrite(fname, sift_img)

    def match_sift_method(self):
        fname1 = askopenfilename()
        fname2 = askopenfilename()
        if fname1 and fname2:
            img1 = cv2.imread(fname1, 0)  # queryImage
            img2 = cv2.imread(fname2, 0)  # trainImage

            sift_img = self.image_manager.match_sift_method(img1, img2)
            fname = asksaveasfilename()
            if fname:
                cv2.imwrite(fname, sift_img)

    def hough_for_lines(self):

        o_from = askfloat("Parameters", "Phi from? (from 0 to 360)")
        o_step = askfloat("Parameters", "Phi step? (from 0 to 360)")
        o_to = askfloat("Parameters", "Phi to? (from 0 to 360)")
        p_from = askfloat("Parameters", "P from? (from 0 to sqrt(2)*D)")
        p_step = askfloat("Parameters", "P step? (from 0 to sqrt(2)*D)")
        p_to = askfloat("Parameters", "P to? (from 0 to sqrt(2)*D)")
        epsilon = askfloat("Parameters", "Epsilon?")
        percentage = askfloat(
            "Parameters", "Percentage for max values?", minvalue=0, maxvalue=1)
        if o_step is None or p_step is None or epsilon is None or percentage is None:
            return
        l = self.image_manager.hugh_for_lines((o_from, o_step, o_to),
                                              (p_from, p_step, p_to), epsilon,
                                              percentage)
        self.studio.mark_lines(l)
        self.menu.show_unmark_button()

    def hough_for_circles(self):
        p_from = askfloat("Parameters", "a and b from?")
        p_step = askfloat("Parameters", "a and b step?")
        p_to = askfloat("Parameters", "a and b to?")
        r_from = askfloat("Parameters", "r from?")
        r_step = askfloat("Parameters", "r step?")
        r_to = askfloat("Parameters", "r to?")
        epsilon = askfloat("Parameters", "Epsilon?")
        if r_step is None or p_step is None or epsilon is None:
            return
        c = self.image_manager.hugh_for_circles(
            (p_from, p_step, p_to), (r_from, r_step, r_to), epsilon)
        self.studio.mark_circles(c)
        self.menu.show_unmark_button()

    def contour_detection_method(self, lin, lout):
        nmax = askinteger("Parameters", "Max iterations?")
        probability = askfloat("Parameters", "F Probability?")
        full_tracking = tkinter.messagebox.askyesno("Parameters",
                                                    "Apply second cycle?")
        if not nmax or not probability:
            return
        p = self.image_manager.contour_detection_method(
            lin, lout, nmax, probability, full_tracking)
        self.studio.mark_pixels(p, 1)
        self.menu.show_unmark_button()

    def contour_detection_video_method(self, lin, lout):
        def callback(lin):
            gui.studio.unmark_pixels()
            gui.studio.show_image()
            gui.studio.mark_pixels(lin, 3)
            gui.studio.update()

        self.selection_for_video = False
        nmax = askinteger("Parameters", "Max iterations?")
        probability = askfloat("Parameters", "F Probability?")
        if not nmax or not probability:
            return
        stats = self.image_manager.contour_detection_video_method(
            lin, lout, nmax, self.file_map, self.starting_number, callback,
            probability, self.full_tracking, self.hsv_tracking)
        avg = sum(stats) / len(stats)
        fps = int(1 / avg)
        self.menu.show_unmark_button()
        tkinter.messagebox.showinfo(
            'Info',
            'Average processing time for each frame: {}.\nFPS: {}'.format(avg,
                                                                          fps))

    def _video_tracking(self):
        dname = askdirectory()
        self.file_map = {}
        regex = re.compile('[a-zA-Z|0]+(?P<number>\d+)\.[a-zA-Z]+')
        if not dname:
            return
        self.starting_number = askinteger(
            "Parameters",
            "Starting number? (first appearance of object)",
            minvalue=1)
        if not self.starting_number:
            self.starting_number = 1
        for subdir, dirs, files in os.walk(dname):
            for file_name in files:
                full_path = os.path.join(subdir, file_name)
                mat = regex.match(file_name)
                if mat:
                    number = int(mat.groupdict()['number'])
                    self.file_map[number] = (full_path, file_name)

        img = Image.open(self.file_map[self.starting_number][0])
        try:
            self.image_manager.load_image(img)
        except Exception:
            tkinter.messagebox.showinfo('Alert', 'Unsupported image format')
        self.load_images()
        tkinter.messagebox.showinfo(
            'Info',
            'This is the first image of the video where the object is present. Select object region.')
        self.selection_for_video = True

    def video_tracking(self):
        self.full_tracking = False
        self._video_tracking()

    def video_tracking_hsv(self):
        self.full_tracking = False
        self.hsv_tracking = True
        self._video_tracking()

    def full_video_tracking(self):
        self.full_tracking = True
        self._video_tracking()


if __name__ == "__main__":
    gui = GUI()
    if (len(sys.argv) > 1 and sys.argv[1]):
        gui.load_file(sys.argv[1])
    gui.mainloop()
