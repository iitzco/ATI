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


class LoadMenuBar(tk.Menu):

    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.add_command(label="Load Image", command=parent.gui.load_file)
        self.add_command(
            label="Load Rectangle",
            command=parent.gui.load_rectangle)
        self.add_command(label="Load Circle", command=parent.gui.load_circle)
        self.add_command(
            label="Load (almost) Black Canvas",
            command=parent.gui.load_black)
        self.add_command(
            label="Load (almost) White Canvas",
            command=parent.gui.load_white)


class NoiseMenuBar(tk.Menu):

    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.add_command(
            label="Exponential",
            command=parent.gui.exponential_noise)
        self.add_command(label="Rayleigh", command=parent.gui.rayleigh_noise)
        self.add_command(label="Gauss", command=parent.gui.gauss_noise)
        self.add_command(
            label="Salt and Pepper",
            command=parent.gui.salt_pepper_noise)


class FilteringMenuBar(tk.Menu):

    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.add_command(
            label="Mean Filter",
            command=parent.gui.mean_filter)
        self.add_command(
            label="Median Filter",
            command=parent.gui.median_filter)
        self.add_command(label="Gauss Filter", command=parent.gui.gauss_filter)
        self.add_command(
            label="Border Filter",
            command=parent.gui.border_filter)


class OperationsMenuBar(tk.Menu):

    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.add_command(
            label="Negative",
            command=parent.gui.negative)
        self.add_command(
            label="Umbralize",
            command=parent.gui.umbral)
        self.add_command(label="Power", command=parent.gui.power)
        self.add_command(label="Product", command=parent.gui.product)
        self.add_command(label="Range Compression", command=parent.gui.compression)
        self.add_command(label="Contrast Enhancement", command=parent.gui.enhance_contrast)
        self.add_command(label="Equalize", command=parent.gui.equalize)
        self.add_separator()
        self.add_command(
            label="Add Image",
            command=parent.gui.add_img)
        self.add_command(
            label="Substract image",
            command=parent.gui.substract_img)
        self.add_command(label="Multiply Image", command=parent.gui.multiply_img)

class StatsMenuBar(tk.Menu):

    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.add_command(
            label="Histogram",
            command=parent.gui.histogram)

class UtilsMenuBar(tk.Menu):

    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.add_command(
            label="To B&W",
            command=parent.gui.to_bw)

class BorderMenuBar(tk.Menu):

    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.add_command(
            label="Prewitt Method",
            command=parent.gui.prewitt_method)
        self.add_command(
            label="Sobel Method",
            command=parent.gui.sobel_method)
