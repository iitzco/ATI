import tkinter as tk


class MenuBar(tk.Menu):

    def __init__(self, gui):
        tk.Menu.__init__(self, gui)
        self.gui = gui
        self.add_components()

    def add_components(self):
        self.add_cascade(label="Load", menu=LoadMenuBar(self))
        self.add_cascade(label="Noise", menu=NoiseMenuBar(self))


class LoadMenuBar(tk.Menu):

    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.add_command(label="Load Image", command=parent.gui.load_file)
        self.add_command(label="Load Rectangle", command=parent.gui.load_rectangle)
        self.add_command(label="Load Circle", command=parent.gui.load_circle)
        self.add_command(label="Load (almost) Black Canvas", command=parent.gui.load_black)
        self.add_command(label="Load (almost) White Canvas", command=parent.gui.load_white)


class NoiseMenuBar(tk.Menu):

    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.add_command(label="Exponential", command=parent.gui.exponential_noise)
        self.add_command(label="Rayleigh", command=parent.gui.rayleigh_noise)
        self.add_command(label="Gauss", command=parent.gui.gauss_noise)
        self.add_command(label="Salt and Pepper", command=parent.gui.salt_pepper_noise)

