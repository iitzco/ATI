import tkinter as tk


class MenuBar(tk.Menu):
    def __init__(self, gui):
        tk.Menu.__init__(self, gui)
        self.gui = gui
        self.add_components()

    def add_components(self):
        self.add_cascade(label="Load", menu=LoadMenuBar(self))
        self.add_cascade(label="Noise", menu=NoiseMenuBar(self))
        self.add_cascade(label="Filtering", menu=FilteringMenuBar(self))
        self.add_cascade(label="Operations", menu=OperationsMenuBar(self))
        self.add_cascade(label="Stats", menu=StatsMenuBar(self))
        self.add_cascade(label="Utils", menu=UtilsMenuBar(self))
        self.add_cascade(label="Borders", menu=BorderMenuBar(self))
        self.add_cascade(
            label="Characteristics", menu=CharacteristicsMenuBar(self))
        self.add_cascade(label="Shapes", menu=ShapesMenuBar(self))
        self.add_cascade(label="Contours", menu=ContourMenuBar(self))


class LoadMenuBar(tk.Menu):
    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.add_command(label="Load Image", command=parent.gui.load_file)
        self.add_command(
            label="Load Rectangle", command=parent.gui.load_rectangle)
        self.add_command(label="Load Circle", command=parent.gui.load_circle)
        self.add_command(
            label="Load (almost) Black Canvas", command=parent.gui.load_black)
        self.add_command(
            label="Load (almost) White Canvas", command=parent.gui.load_white)


class NoiseMenuBar(tk.Menu):
    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.add_command(
            label="Exponential", command=parent.gui.exponential_noise)
        self.add_command(label="Rayleigh", command=parent.gui.rayleigh_noise)
        self.add_command(label="Gauss", command=parent.gui.gauss_noise)
        self.add_command(
            label="Salt and Pepper", command=parent.gui.salt_pepper_noise)


class FilteringMenuBar(tk.Menu):
    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.add_command(label="Mean Filter", command=parent.gui.mean_filter)
        self.add_command(
            label="Median Filter", command=parent.gui.median_filter)
        self.add_command(label="Gauss Filter", command=parent.gui.gauss_filter)
        self.add_command(
            label="Border Filter", command=parent.gui.border_filter)
        self.add_command(
            label="Leclerc Anisotropic Diffusion",
            command=parent.gui.leclerc_anisotropic_diffusion)
        self.add_command(
            label="Lorentziano Anisotropic Diffusion",
            command=parent.gui.lorentziano_anisotropic_diffusion)
        self.add_command(
            label="Isotropic Diffusion",
            command=parent.gui.isotropic_diffusion)


class OperationsMenuBar(tk.Menu):
    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.add_command(label="Negative", command=parent.gui.negative)
        self.add_command(label="Umbralize", command=parent.gui.umbral)
        self.add_command(
            label="Global Umbralization", command=parent.gui.global_umbral)
        self.add_command(
            label="Otsu Umbralization", command=parent.gui.otsu_umbral)
        self.add_command(label="Power", command=parent.gui.power)
        self.add_command(label="Product", command=parent.gui.product)
        self.add_command(
            label="Range Compression", command=parent.gui.compression)
        self.add_command(
            label="Contrast Enhancement", command=parent.gui.enhance_contrast)
        self.add_command(label="Equalize", command=parent.gui.equalize)
        self.add_separator()
        self.add_command(label="Add Image", command=parent.gui.add_img)
        self.add_command(
            label="Substract image", command=parent.gui.substract_img)
        self.add_command(
            label="Multiply Image", command=parent.gui.multiply_img)


class StatsMenuBar(tk.Menu):
    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.add_command(label="Histogram", command=parent.gui.histogram)


class UtilsMenuBar(tk.Menu):
    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.add_command(label="To B&W", command=parent.gui.to_bw)
        self.add_command(label="Sobel X", command=parent.gui.sobel_x_img)
        self.add_command(label="Sobel Y", command=parent.gui.sobel_y_img)
        self.add_command(
            label="Laplacian Mask", command=parent.gui.laplacian_mask)
        self.add_command(label="LoG Mask", command=parent.gui.LoG_mask)
        self.add_command(label="Rotate", command=parent.gui.rotate)


class BorderMenuBar(tk.Menu):
    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.add_command(
            label="Prewitt Method", command=parent.gui.prewitt_method)
        self.add_command(label="Sobel Method", command=parent.gui.sobel_method)
        self.add_command(
            label="Laplacian Method", command=parent.gui.laplacian_method)
        self.add_command(
            label="Laplacian Pending Method",
            command=parent.gui.laplacian_pending_method)
        self.add_command(label="LoG Method", command=parent.gui.LoG_method)
        self.add_separator()
        self.add_command(
            label="Kirsh Directional Method",
            command=parent.gui.kirsh_directional_method)
        self.add_command(
            label="Prewitt Directional Method",
            command=parent.gui.prewitt_directional_method)
        self.add_command(
            label="Sobel Directional Method",
            command=parent.gui.sobel_directional_method)
        self.add_command(
            label="Alternative Directional Method",
            command=parent.gui.alternative_directional_method)
        self.add_separator()
        self.add_command(label="Canny Method", command=parent.gui.canny_method)
        self.add_command(
            label="Canny Method with hysteresis",
            command=parent.gui.canny_hysteresis_method)


class CharacteristicsMenuBar(tk.Menu):
    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.add_command(
            label="Harris Method", command=parent.gui.harris_method)
        self.add_command(label="Susan Method", command=parent.gui.susan_method)
        self.add_command(label="SIFT Method", command=parent.gui.sift_method)
        self.add_command(
            label="SIFT Matcher Method", command=parent.gui.match_sift_method)


class ShapesMenuBar(tk.Menu):
    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.add_command(
            label="Hough Method for lines", command=parent.gui.hough_for_lines)
        self.add_command(
            label="Hough Method for circles",
            command=parent.gui.hough_for_circles)


class ContourMenuBar(tk.Menu):
    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.add_command(
            label="Video Tracking", command=parent.gui.video_tracking)
        self.add_command(
            label="Video Tracking HSV", command=parent.gui.video_tracking_hsv)
        self.add_command(
            label="Full Video Tracking (With 2nd cycle)",
            command=parent.gui.full_video_tracking)
