import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import ImageTk, Image

import sys

class GUI(tk.Frame):

    def __init__(self):
        tk.Frame.__init__(self)
        self.master.title("GUI")

        # Full Windows Size
        self.master.geometry("{0}x{1}+0+0".format(
                        self.master.winfo_screenwidth(), 
                        self.master.winfo_screenheight()))

        self.pack(fill=tk.BOTH,expand=tk.YES)
        self.create_subframes()

    def create_subframes(self):
        
        self.original = ImageWorkspace(self, title="Original")
        self.studio = ImageWorkspace(self, title="Studio")
        self.menu = Menu(self) 

        self.original.pack(fill=tk.BOTH, expand=tk.YES, side=tk.LEFT)
        self.studio.pack(fill=tk.BOTH, expand=tk.YES, side=tk.LEFT)
        self.menu.pack(fill=tk.Y, expand=tk.YES,  side=tk.RIGHT)

        self.menu.discover()
        self.original.discover()
        self.studio.discover()

        # self.menu.zoom = tk.Canvas(self.menu, height=150, width=150)
        # self.menu.zoom.grid(row=0,column=0)
        # self.menu.zoom.create_text(10,10, anchor='nw',text="Click Image to zoom")
        

    def revert(self):
        self.img = self.img.rotate(180)
        self.imagerotated = ImageTk.PhotoImage(self.img)
        self.studio.canvas.itemconfig(self.id, image = self.imagerotated)

    def load_file(self):
        fname = askopenfilename()
        if fname:
            self.img = Image.open(fname)
            self.original.show_image(self.img)
            # self.original.canvas.create_image(0, 0, anchor='nw',image=self.image)

    def save_file(self):
        fname = asksaveasfilename()
        if fname:
            self.img.save(fname)

    def zoom(self, event):
        self.imgCropped = self.img.crop((event.x-20, event.y-20, event.x+20, event.y+20)).resize((150,150))
        self.imagetk = ImageTk.PhotoImage(self.imgCropped)
        self.menu.zoom.create_image(0, 0, anchor='nw',image=self.imagetk)
        self.menu.zoom2.create_image(0, 0, anchor='nw',image=self.imagetk)


class ImageWorkspace(tk.Frame):

    def __init__(self, gui, title):
        tk.Frame.__init__(self, gui)
        self.gui = gui
        self.label = tk.Label(self, text=title)
        self.canvas_main = tk.Canvas(self)
        self.canvas_zoom = tk.Canvas(self)

        self.canvas_main.bind("<B1-Motion>", self.zoom)
        self.canvas_main.bind("<Button-1>", self.zoom)

    def discover(self):
        self.label.pack(side=tk.TOP)
        self.canvas_main.pack(fill=tk.BOTH, expand=tk.YES)

    def discover_zoom_mode(self):
        self.canvas_zoom.pack(fill=tk.BOTH, expand=tk.YES)

    def show_image(self, img):
        imgtk = ImageTk.PhotoImage(img)
        self.image = imgtk
        self.canvas_main.create_image(0, 0, anchor='nw',image=self.image)

    def zoom(self, event):
        print("{} - {}".format(event.x, event.y))
        # self.imgCropped = self.img.crop((event.x-20, event.y-20, event.x+20, event.y+20)).resize((150,150))
        # self.imagetk = ImageTk.PhotoImage(self.imgCropped)
        # self.menu.zoom.create_image(0, 0, anchor='nw',image=self.imagetk)
        # self.menu.zoom2.create_image(0, 0, anchor='nw',image=self.imagetk)


class Menu(tk.Frame):

    def __init__(self, gui):
        tk.Frame.__init__(self, gui)
        self.gui = gui
        self.add_components()

    def add_components(self):
        self.label = tk.Label(self, text="Menu")
        self.button_load = tk.Button(self, text="Load Image", command=self.gui.load_file, width=10)
        self.button_save = tk.Button(self, text="Save", command=self.gui.save_file, width=10)
        self.button_revert = tk.Button(self, text="Revert", command=self.gui.revert, width=10)
        self.color_slider = tk.Scale(self, from_=0, to=255, orient=tk.HORIZONTAL)
        self.radio_zoom = tk.Checkbutton(self, text= "Zoom Mode", variable=self.gui.zoom)

    def discover(self):
        self.label.grid(pady=10)
        self.button_load.grid(pady=10)
        self.button_save.grid(pady=10)
        self.radio_zoom.grid(pady=10)
        self.button_revert.grid(pady=10)
        self.color_slider.grid(pady=10)


if __name__ == "__main__":
    GUI().mainloop()
    
